# Urdu Voice-Cloning Dubbing Pipeline

Take raw Urdu speech from one speaker and produce new audio (English dubbing) in that same
cloned voice.

## Current approach: Sooktam-2 zero-shot cloning

**No fine-tuning required.** [Sooktam-2](https://huggingface.co/bharatgenai/sooktam2) (BharatGen)
is a TTS model actually pretrained on real Urdu (and Hindi, and 10 other Indian languages)
speech data, with zero-shot voice cloning built in — give it a short reference clip + its
transcript, and it clones that voice for any new text, including cross-lingual (Urdu reference
→ English output), which we've validated sounds both correct and close to the source voice.

**License note:** Sooktam-2's checkpoint is released under a BharatGen **non-commercial**
license — fine for this project's current (non-commercial) use, but re-check the license if
usage plans change.

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

**1. Transcribe the Urdu audio** (for picking a good reference clip, and as a sanity check):
```bash
python scripts/01_transcribe_urdu.py \
    --audio /mnt/extra/bxm0694/40_minutes_training_audio.m4a \
    --out transcript_ur.json \
    --model large-v3
```

**2. Merge Whisper's segments into sentence-like units, then translate with IndicTrans2.**
Three approaches were tried here, in order:
1. Whisper's per-clip translate mode — prone to misdetecting the language on short clips.
2. IndicTrans2 translating each raw Whisper segment in isolation — better, but still broken on
   fragments, since Whisper splits on *pauses* in speech, not grammatical sentence boundaries
   (many segments are incomplete clauses, e.g. "کے زمن میں" = "in the time of").
3. **Batching segments through the Claude API with cross-segment context** — fixed quality
   without merging, but costs API credits per run; abandoned in favor of the free option below
   once we confirmed simple merging alone fixes most of the same problem.

**Current approach:** merge consecutive raw segments into more complete units first — using
Urdu sentence-ending punctuation when present, an unusually long pause before the next segment,
or a word/duration cap as a fallback — then translate each *merged* unit with IndicTrans2. Each
merged unit keeps its first sub-segment's original start time as its anchor, so stage 5's
timestamp-based audio-sync is unaffected; we're just anchoring at whole-sentence boundaries
instead of mid-sentence ones.
```bash
pip install IndicTransToolkit
python scripts/04_translate_ur_to_en.py \
    --transcript transcript_ur.json \
    --out transcript_en.json
```
Requires the same gated-model HF login as before (`huggingface-cli login`, after accepting the
model's terms at https://huggingface.co/ai4bharat/indictrans2-indic-en-1B). This writes a plain,
editable JSON file **before** any audio is generated, on purpose — review it and hand-fix any
translations that read oddly (edit the `"text"` field for that unit, save, and go straight to
stage 5 — no need to re-run translation). Each entry also carries `"source_ids"` (the original
Whisper segment ids it was merged from) for traceability back to `transcript_ur.json`.

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
audio is generated at its natural pace — never sped up or slowed down. Each segment starts at
the same timestamp its Urdu original started at, whenever possible; if English finishes early,
the gap is left as silence; if English runs long enough to bump into the next segment's start
time, the next segment starts immediately after instead (pushed later). Net effect: the final
dub is never shorter than the original, and can run longer if enough segments overran — but
every sentence's *start* stays as close to its original timing as the "no speed changes" rule
allows.

Both stage 4 and stage 5 are reusable as-is for any future Urdu recording — just point them at
new `--audio`/`--transcript` files.

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
