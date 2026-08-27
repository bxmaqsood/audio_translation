# Borrowed Voice

Forty minutes of a lecturer's Urdu, turned into the same talk delivered in English — same voice,
same cadence, someone else's language. Not a sound-alike reading a script: the actual speaker's
voice, cloned and then taught to speak a language it never has.

**[Read the full story →](https://claude.ai/code/artifact/4b1df2c4-6780-446f-9074-ae4f155661e9)**
— the dead ends, the bugs, and the checkpoint that quietly lied about being the best one in the
batch. What follows below is the technical reference: setup, exact commands, and every gotcha
hit along the way, for anyone reproducing or extending this.

**Where it landed:** zero-shot cloning with [Sooktam-2](#base-approach-sooktam-2-zero-shot-cloning-quick-start)
gets a correct, recognizable clone straight away, and is enough to start with. **Fine-tuning it
further on the speaker's own audio is the recommended path for production quality** — see
[below](#recommended-fine-tuning-sooktam-2-for-natural-delivery) — picked via held-out validation
loss rather than the last checkpoint on disk:
[checkpoint 700 won, the final one scored among the worst](#validation-loss-for-fine-tuning-no-built-in-support-in-f5-ttss-trainer).

## Base approach: Sooktam-2 zero-shot cloning (quick start)

**No fine-tuning required.** [Sooktam-2](https://huggingface.co/bharatgenai/sooktam2) (BharatGen)
is a TTS model actually pretrained on real Urdu (and Hindi, and 10 other Indian languages)
speech data, with zero-shot voice cloning built in — give it a short reference clip + its
transcript, and it clones that voice for any new text, including cross-lingual (Urdu reference
→ English output), which we've validated sounds both correct and close to the source voice.

**License note:** Sooktam-2's checkpoint is released under a BharatGen **non-commercial**
license — fine for this project's current (non-commercial) use, but re-check the license if
usage plans change.

Zero-shot is a good starting point to confirm the pipeline works end-to-end, but it only has a
few seconds of reference audio to draw from — correct, but flatter than the speaker actually
sounds. For real output, go straight to
[fine-tuning Sooktam-2](#recommended-fine-tuning-sooktam-2-for-natural-delivery) once the basic
pipeline below is confirmed working.

### Why not fine-tune Chatterbox? (earlier approach, abandoned)

We initially tried fine-tuning [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox)
(via the [chatterbox-finetuning](https://github.com/gokhaneraslan/chatterbox-finetuning) toolkit),
transliterating Urdu to Devanagari on the theory that Chatterbox already knows Hindi. That
theory turned out to be **false for the actual model weights**: Chatterbox's tokenizer *lists*
Hindi as one of 23 supported languages, but inspecting `resize_and_load_t3_weights` and the
actual token IDs our text mapped to (1671-1727, all above the 704-token "genuinely pretrained"
cutoff) confirmed those weights were never actually trained — just reserved, mean-initialized
placeholders. We were effectively teaching a brand-new script from near-scratch on very little
data, which produced a well-cloned voice but gibberish "words," even after scaling up training
data/epochs. Once we found a model (Sooktam-2) with genuine Urdu pretraining, zero-shot cloning
on it beat everything we got from hours of Chatterbox fine-tuning — so we switched. The old
Chatterbox scripts (`02_transliterate_ur_to_hi.py`, `03_prepare_dataset.py`) are kept in this
repo for reference but are no longer part of the active pipeline.

## Pipeline (current)

**1. Transcribe the Urdu audio** (with word-level timestamps — stage 4 needs per-word timing to
find real sentence breaks; see below for why):
```bash
python scripts/01_transcribe_urdu.py \
    --audio /mnt/extra/bxm0694/40_minutes_training_audio.m4a \
    --out transcript_ur.json \
    --model large-v3
```

**2. Group words into sentences by actual speaker pauses, then translate with IndicTrans2.**
Several approaches were tried here, in order, each fixing a real problem with the last:
1. Whisper's per-clip translate mode — prone to misdetecting the language on short clips.
2. IndicTrans2 translating each raw Whisper *segment* in isolation — better, but Whisper splits
   audio on *pauses*, not grammatical sentence boundaries, so many segments are incomplete
   clauses (e.g. "کے زمن میں" = "in the time of") that translate badly alone.
3. Batching segments through the Claude API with cross-segment context — fixed quality without
   merging, but costs API credits per run.
4. Merging segments using punctuation + a word/duration cap — free and simpler, big quality win,
   but still an approximation of real sentence boundaries.
5. **Current approach: merge using ONLY actual pause timing, at the word level.** We measured
   the real gap distribution between consecutive Whisper *segments* first — 90% were exactly
   zero, meaning segment boundaries don't reflect real pauses at all. Word-level timestamps do:
   in our data, 92% of word-to-word gaps were ~zero (continuous speech), with a sharp, clean
   boundary at 0.2s (zero gaps fell between 0.15s and 0.2s at all) separating "still talking"
   from "the speaker paused." `GAP_THRESHOLD_SEC = 0.2` in the script encodes this — **re-check
   the gap distribution (see the script's docstring) before assuming 0.2s holds for a
   differently-paced speaker/recording.** No other signal (punctuation, word count, duration) is
   used, by design — pure pause timing is both the truest signal for sentence meaning and what
   naturally lines up with audio-sync anchoring.

Each resulting sentence's start time is its first word's real start time, so stage 5's
timestamp-based audio-sync anchors to where the speaker actually began that sentence.
```bash
pip install IndicTransToolkit
python scripts/04_translate_ur_to_en.py \
    --transcript transcript_ur.json \
    --out transcript_en.json
```
Requires the same gated-model HF login as before (`huggingface-cli login`, after accepting the
model's terms at https://huggingface.co/ai4bharat/indictrans2-indic-en-1B). This writes a plain,
editable JSON file **before** any audio is generated, on purpose — review it and hand-fix any
translations that read oddly (edit the `"text"` field for that segment, save, and go straight to
stage 5 — no need to re-run translation).

**3. Pick a good reference clip.** You want a short (3-6s), clean, single-sentence clip with an
accurate transcript. We used `awk` on `dataset/metadata_debug.csv` (from the earlier Chatterbox
attempt) to find candidates by duration — or just listen to a few candidate segments from
`transcript_ur.json` and slice one out with `ffmpeg`:
```bash
ffmpeg -i /mnt/extra/bxm0694/40_minutes_training_audio.m4a -ss <start> -to <end> -ar 24000 -ac 1 reference.wav
```

**4. Generate the dub**, in the `sooktam` conda environment:
```bash
python scripts/05_generate_dub.py \
    --english transcript_en.json \
    --ref-file reference.wav \
    --ref-text "<accurate Urdu transcript of reference.wav>" \
    --out final_dub.wav
```
**Timing behavior (deliberate, see the script's docstring for full detail):** every segment's
audio is generated at its natural pace by default — never sped up or slowed down. Each segment
starts at the same timestamp its Urdu original started at, whenever possible; if English
finishes early, the gap is left as silence; if English runs long enough to bump into the next
segment's start time, the next segment starts immediately after instead (pushed later). Net
effect: the final dub is never shorter than the original, and can run longer if enough segments
overran — but every sentence's *start* stays as close to its original timing as the "no speed
changes" rule allows.

**Optional deliberate slowdown/speedup:** pass `--speed 0.65` (or any factor) to uniformly
change the pace of each *generated* segment before it's placed — pitch-preserving (ffmpeg
`atempo`, not a naive resample), so it sounds calmer/more measured, not deeper/robotic. This
changes segment *duration*, not the timing rules above — a slowed segment is just longer, so
the existing anchor/push-back logic handles it automatically:
```bash
python scripts/05_generate_dub.py \
    --english transcript_en.json \
    --ref-file reference.wav \
    --ref-text "<accurate Urdu transcript of reference.wav>" \
    --out final_dub_slow065.wav \
    --speed 0.65
```

Both stage 4 and stage 5 are reusable as-is for any future Urdu recording — just point them at
new `--audio`/`--transcript` files.

## Recommended: fine-tuning Sooktam-2 for natural delivery

Zero-shot cloning (above) works well for correctness/voice-match but can sound flat/"like
reading" — it only has a ~4s reference clip to learn from, which captures timbre but not how
this speaker actually varies pitch/rhythm/emphasis across real speech. Fine-tuning on the
speaker's full recording should fix this.

**This is not the same trap as the Chatterbox attempt.** That failed because Devanagari had
*zero* real pretraining in the checkpoint. Sooktam-2 already has working Urdu pretraining
(that's why zero-shot cloning already sounds correct) — fine-tuning here only deepens
voice-specific delivery, a well-precedented task.

Sooktam-2 ships inference-only (no training code), but its `config.json` confirms
`"model_name": "F5TTS_v1_Base"` — it's built on upstream [F5-TTS](https://github.com/SWivid/F5-TTS)'s
exact architecture, and its checkpoint filename (`model_1250000.pt`) matches F5TTS_v1_Base's own
default checkpoint step count, strongly suggesting Sooktam-2 was itself trained by continuing
from that base — so F5-TTS's own fine-tuning code should load it cleanly.

```bash
# 1. Get upstream F5-TTS (has the training code Sooktam-2's repo lacks)
cd /mnt/extra/bxm0694/
git clone https://github.com/SWivid/F5-TTS.git

# 2. Build the training dataset (real Urdu text, pause-grouped sentences -- same
#    grouping as stage 4, reused here for natural-sounding training clips)
python scripts/06_prepare_f5tts_dataset.py \
    --audio /mnt/extra/bxm0694/40_minutes_training_audio.m4a \
    --transcript transcript_ur.json \
    --out-dir /mnt/extra/bxm0694/f5tts_dataset

# 3. Convert to F5-TTS's internal training format
cd F5-TTS
python src/f5_tts/train/datasets/prepare_csv_wavs.py \
    /mnt/extra/bxm0694/f5tts_dataset/metadata.csv \
    data/urdu_speaker1_custom

# 4. Fine-tune, starting from Sooktam-2's checkpoint, using its own custom Indic
#    tokenizer (NOT the default pinyin one -- that's for the Chinese/English base
#    model and would mismatch Sooktam-2's actual vocabulary)
python src/f5_tts/train/finetune_cli.py \
    --finetune \
    --pretrain /mnt/extra/bxm0694/sooktam2/model_1250000.pt \
    --tokenizer custom \
    --tokenizer_path /mnt/extra/bxm0694/sooktam2/vocab.txt \
    --dataset_name urdu_speaker1 \
    --exp_name F5TTS_v1_Base \
    --learning_rate 1e-5 \
    --epochs 20 \
    --save_per_updates 500 \
    --last_per_updates 100
```
Notes:
- `--dataset_name` must match what you used in step 3 (`urdu_speaker1`) — the script looks for
  `data/{dataset_name}_{tokenizer}` (so `data/urdu_speaker1_custom`, matching step 3's output dir).
- Checkpoints are saved under `F5-TTS/ckpts/urdu_speaker1/`.
- Full fine-tune (not LoRA — the official script doesn't support it) on only ~40 minutes of data
  risks overfitting fast. Start with few epochs (`--epochs 20`, not the 100 default) and check
  intermediate checkpoints rather than assuming more training is better — same lesson as the
  Chatterbox loss-vs-generation-quality mismatch (see git history for that investigation).
- Batch size is in mel-spectrogram *frames* (`--batch_size_per_gpu`, default 3200), not sample
  count — reduce it if you hit CUDA OOM on the L40S.
- **Always `rm -rf ckpts/<dataset_name>` before a fresh attempt.** `load_checkpoint()` prioritizes
  any existing `model_last.pt`/`pretrained_*` files in that directory over `--pretrain`, so a
  stale file from a previous (possibly broken) attempt silently gets reused instead of your fix.

### Three real bugs we hit and fixed (all confirmed, not guesses)

1. **Training silently does nothing (no progress, no error) when fine-tuning a checkpoint with a
   high step count.** `trainer.train()`'s resume logic computes
   `skipped_epoch = start_update // batches_per_epoch`. Sooktam-2's checkpoint carries its own
   original step count (1,250,000) baked into an `"update"` key; with our small dataset's few
   batches-per-epoch, `skipped_epoch` comes out far larger than `--epochs`, so
   `for epoch in range(skipped_epoch, self.epochs)` is an empty range — nothing trains.
   **Fix:** strip the training-state keys, keeping only `ema_model_state_dict`, so
   `load_checkpoint()` treats it as fresh pretrained weights (`update=0`) instead of "resume this
   exact run":
   ```python
   import torch
   ckpt = torch.load('/mnt/extra/bxm0694/sooktam2/model_1250000.pt', map_location='cpu', weights_only=False)
   torch.save({'ema_model_state_dict': ckpt['ema_model_state_dict']},
              '/mnt/extra/bxm0694/sooktam2/model_1250000_clean_for_finetune.pt')
   ```
   Use that `_clean_for_finetune.pt` file as `--pretrain`, not the original.

2. **`--num_warmup_updates` defaults to 20,000**, far more than a small fine-tune dataset's total
   planned updates (e.g. ~800 for 20 epochs on ~280 clips) — the LR schedule's warmup phase never
   finishes, so the effective learning rate stays near-zero for the entire run. Set
   `--num_warmup_updates` to something proportional (we used 50).

3. **`RuntimeError: Function 'ScaledDotProductEfficientAttentionBackward0' returned nan values`**
   — loss is valid on the very first update, then NaN forever after (confirmed conclusively via
   `torch.autograd.set_detect_anomaly(True)`, which pinpoints this exact op; lowering the learning
   rate by 1000x made no difference, ruling out "gradient too large" as the cause). This is a
   known PyTorch numerical-stability issue in the memory-efficient/flash `scaled_dot_product_
   attention` backward pass, not a data or checkpoint problem. **Fix:** force the more stable
   "math" SDPA backend for training — add near the top of `finetune_cli.py` (after `import torch`):
   ```python
   torch.backends.cuda.enable_mem_efficient_sdp(False)
   torch.backends.cuda.enable_flash_sdp(False)
   torch.backends.cuda.enable_math_sdp(True)
   ```

## Validation loss for fine-tuning (no built-in support in F5-TTS's trainer)

F5-TTS's `Trainer`/`finetune_cli.py` has zero built-in validation/held-out-loss support — we
checked (`grep -n "val_dataset\|validation\|eval_dataset" trainer.py finetune_cli.py` returns
nothing). Rather than patch the training loop itself (real risk of new bugs, given how much
debugging the loop already needed), we hold out a validation split up front and evaluate saved
checkpoints against it with a separate, standalone script instead of live during training:
```bash
python3 -c "
import csv, random
random.seed(42)
with open('/mnt/extra/bxm0694/f5tts_dataset/metadata_clean.csv', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|'); header = next(reader); rows = list(reader)
random.shuffle(rows)
n_val = max(10, int(len(rows) * 0.1))
for name, subset in [('train', rows[n_val:]), ('val', rows[:n_val])]:
    with open(f'/mnt/extra/bxm0694/f5tts_dataset/metadata_{name}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='|'); w.writerow(header); w.writerows(subset)
"
```
Train only on `metadata_train.csv` (via `prepare_csv_wavs.py` + `finetune_cli.py` as above);
`metadata_val.csv` stays held out for the evaluation script below.

### Evaluating checkpoints against the held-out set

Prepare the validation set in F5-TTS's format once:
```bash
cd /mnt/extra/bxm0694/F5-TTS
python src/f5_tts/train/datasets/prepare_csv_wavs.py \
    /mnt/extra/bxm0694/f5tts_dataset/metadata_val.csv \
    data/urdu_speaker1_val_custom
```
Then run `scripts/07_evaluate_f5tts_checkpoints.py` (from this repo) against each saved
checkpoint — it replicates just enough of `finetune_cli.py`'s own model-construction and loss
code to run a forward-only pass (no gradient updates) over the held-out clips:
```bash
for ckpt in ckpts/urdu_speaker1/model_*.pt; do
    python /home/bxm0694/audio_translation/scripts/07_evaluate_f5tts_checkpoints.py \
        --checkpoint "$ckpt" \
        --val-dataset-name urdu_speaker1_val \
        --tokenizer-path /mnt/extra/bxm0694/sooktam2/vocab.txt
done
```
Pick the checkpoint with the lowest `val_loss` — **not** the one with the lowest training loss,
and not necessarily the last/final one (see git history / conversation for why: training loss
mostly measures memorization of the specific training clips, not generalization). Validation
loss narrows down candidates; still listen to the top 2-3 before committing to one, since even
validation loss is an imperfect proxy for perceived naturalness.

## Server setup for Sooktam-2

This environment has several real gotchas worth documenting — we hit all of them:

```bash
# 1. Separate conda env (avoid dependency conflicts with the Chatterbox env)
CONDA_NO_PLUGINS=true conda create -n sooktam -c conda-forge python=3.10 -y --solver classic
conda activate sooktam

# 2. Clone the model repo
git clone https://huggingface.co/bharatgenai/sooktam2
cd sooktam2

# 3. setup-cls.sh tries `apt-get install libcudnn9-cuda-12` which needs root we don't have.
#    Skip it — the pip-installed torch wheel already bundles a compatible cuDNN.
sed -i 's|    apt-get install libcudnn9-cuda-12 -y|    :  # skipped: no sudo, torch wheel bundles its own cudnn|' setup-cls.sh
bash setup-cls.sh

# 4. `git clone` doesn't fetch Git LFS content (git-lfs isn't installed, and we can't
#    apt-get it either) — the large checkpoint files download as ~135-byte pointer stubs.
#    Pull the real weights directly over HTTPS instead:
wget -O model_1250000.pt "https://huggingface.co/bharatgenai/sooktam2/resolve/main/model_1250000.pt"
wget -O sooktam.safetensors "https://huggingface.co/bharatgenai/sooktam2/resolve/main/sooktam.safetensors"

# 5. PyTorch 2.6 changed torch.load's default weights_only=True, which breaks loading this
#    (trusted, official) checkpoint. Patch the vendored loader:
sed -i 's/weights_only=True/weights_only=False/g' src/f5_tts/infer/utils_infer.py

# 6. ffmpeg (needed by pydub) isn't installed in this env either:
CONDA_NO_PLUGINS=true conda install -c conda-forge ffmpeg -y --solver classic
```

**Move it somewhere permanent** (its checkpoints are multi-GB and painful to re-download/re-patch
if lost — don't leave it in a temp/scratch location):
```bash
mv sooktam2 /mnt/extra/bxm0694/sooktam2
cd /mnt/extra/bxm0694/sooktam2 && pip install -e . --no-cache-dir   # editable installs point at
                                                                     # a fixed path -- re-run this
                                                                     # any time you move the folder
```

After that, `scripts/05_generate_dub.py` (run from anywhere, in the `sooktam` env) loads the
model from `/mnt/extra/bxm0694/sooktam2` (its default `--model-id`) — passing the local folder
path rather than the `bharatgenai/sooktam2` hub id avoids both re-downloading and a cwd-dependency
bug in the model's custom loading code (see git history on this file for the investigation).

### Notes on `model.infer()` parameters
- `cls_language`: `"urdu"` or `"english"` (also hindi, marathi, etc. — see the model card for
  the full list). Controls how the *target* text is read/pronounced; the reference clip's
  language doesn't need to match (we validated Urdu reference → English output works well).
- Don't pass `fix_duration` unless you have a specific reason to — it stretches/compresses
  the *same* amount of speech content to fit a fixed time rather than generating more content,
  which sounds unnatural. Let it auto-estimate duration from the actual target text length.

## Archived: Chatterbox fine-tuning pipeline

The scripts and setup below are kept for reference but are **not the active pipeline** —
see "Why not fine-tune Chatterbox?" above.

### Chatterbox server setup
```bash
conda create -n dub python=3.10 -y
conda activate dub
pip install torch --index-url https://download.pytorch.org/whl/cu124   # matches driver 575.57 / CUDA 12.9
pip install -r requirements.txt
pip install chatterbox-tts
git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git
```
`pydub` needs `ffmpeg` on PATH — install via conda if not present and you lack sudo:
`conda install -c conda-forge ffmpeg -y`.

### Get the audio onto the server
```bash
scp path/to/your_audio.wav bxm0694@oitrss-ardcdept-136.shost.uta.edu:/mnt/extra/bxm0694/
```

### 2. Transliterate Urdu → Devanagari
```bash
python scripts/02_transliterate_ur_to_hi.py --in transcript_ur.json --sample 8   # preview first
python scripts/02_transliterate_ur_to_hi.py --in transcript_ur.json --out transcript_hi.json --all
```
Requires `pip install indo-arabic-transliteration indic-nlp-library` — importing its offline
converter also needs a workaround for a broken transitive dependency chain (`urduhack` →
`tensorflow_addons` → incompatible Keras 3); the script stubs this out internally. See git
history on this file for the debugging trail if this comes up again with a different package.

### 3. Build the fine-tuning dataset
```bash
python scripts/03_prepare_dataset.py \
    --audio /mnt/extra/bxm0694/40_minutes_training_audio.m4a \
    --transcript transcript_hi.json \
    --out-dir dataset \
    --prefix rec1
```
Supports `--append`/`--prefix` for merging multiple source recordings into one dataset.

### Fine-tuning & known toolkit bugs
- Force single-GPU (`CUDA_VISIBLE_DEVICES=0`) — this toolkit's use of PyTorch's legacy
  `DataParallel` across multiple GPUs crashes on an internal assertion.
- `pip install tf-keras setuptools<70` — needed to work around `transformers` and `perth`
  (audio watermarking) import failures in this environment.
- `src/chatterbox_/models/t3/inference/alignment_stream_analyzer.py` has a real bug: its
  "3x repetition" check only actually checks the last 2 tokens, not 3.
- Root cause of poor output quality: see "Why not fine-tune Chatterbox?" above — this wasn't a
  bug we could fix, it was a fundamental data/pretraining gap.
