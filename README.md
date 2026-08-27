# Urdu Voice-Cloning Dubbing Pipeline

Take raw Urdu speech from one speaker and produce new audio (e.g. English dubbing) in that
same cloned voice, by fine-tuning [Chatterbox Multilingual TTS](https://github.com/resemble-ai/chatterbox)
on the speaker.

## Why this approach

Chatterbox has never seen Urdu's Perso-Arabic script, so fine-tuning it directly on raw Urdu
text with only ~40 minutes of audio fails: it learns the *voice* fine, but can't learn a brand
new alphabet from that little data and produces gibberish.

Chatterbox Multilingual **does** natively support Hindi (Devanagari script). Since Hindi and
Urdu are the same spoken language (Hindustani) with different scripts, transliterating the
Urdu transcript to Devanagari lets fine-tuning lean on language/script knowledge the model
already has — turning this into a much easier "adapt to a new voice" task instead of "learn a
new writing system" task.

Pipeline: **raw audio → Whisper transcript (Urdu) → transliterate to Devanagari → build
training clips → LoRA fine-tune Chatterbox → generate dubbed audio in the cloned voice**.

## Status

This repo is being built incrementally — see commit history for progress. Scripts are meant to
run on the GPU server (not here), one stage at a time, with manual sanity checks in between.

## Server setup

```bash
conda create -n dub python=3.10 -y
conda activate dub

# Match torch's CUDA build to the server's driver (575.57 / CUDA 12.9 — cu124 wheels work)
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install chatterbox-tts

# Third-party fine-tuning toolkit (not vendored in this repo)
git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git
```

`pydub` needs `ffmpeg` on PATH — check with `ffmpeg -version`; install via `sudo apt install
ffmpeg` if missing (or ask the admin, since this account isn't in sudoers).

## Get the audio onto the server

From your Windows machine:
```bash
scp path/to/your_audio.wav bxm0694@oitrss-ardcdept-136.shost.uta.edu:/mnt/extra/bxm0694/
```

## Pipeline steps

Detailed commands for each stage are added below as the corresponding script lands in this
repo. Run them in order from the server, in the `dub` conda environment.

### 1. Transcribe (Urdu)
```bash
python scripts/01_transcribe_urdu.py \
    --audio /mnt/extra/bxm0694/speaker.wav \
    --out transcript_ur.json \
    --model large-v3
```
Prints the first few segments so you can eyeball transcription quality immediately. Re-run
with `--model medium` if `large-v3` is too slow for a quick test, but use `large-v3` for the
real run — Urdu accuracy matters a lot here since every downstream stage depends on it.

### 2. Transliterate Urdu → Devanagari
**Always preview a sample first** — this is where the earlier attempt at this approach failed
silently with a bad converter:
```bash
python scripts/02_transliterate_ur_to_hi.py --in transcript_ur.json --sample 8
```
Read the printed pairs and confirm the Devanagari genuinely reads as the same Hindustani words.
Only once that looks right, run the full pass:
```bash
python scripts/02_transliterate_ur_to_hi.py --in transcript_ur.json --out transcript_hi.json --all
```
This caches results in `transliteration_cache.json` (safe to re-run/resume).

**Known issue:** the online API (`sangam.learnpunjabi.org`, an old academic service) is
currently returning HTTP 500 errors — it's disabled by default. The offline converter is used
instead; it occasionally produces stray non-Devanagari symbols or rare/unusual characters, which
the script strips or flags automatically. Any flagged segments are written to
`flagged_segments.json` for manual review before training — check that file and hand-fix or drop
those segments rather than feeding them into fine-tuning as-is.

Install its one missing dependency first if you haven't:
```bash
pip install indic-nlp-library
```

### 3. Build the fine-tuning dataset
Slices the source audio into per-segment clips (trimming silence at the edges) and writes an
LJSpeech-style `metadata.csv` using the Devanagari text:
```bash
python scripts/03_prepare_dataset.py \
    --audio /mnt/extra/bxm0694/40_minutes_training_audio.m4a \
    --transcript transcript_hi.json \
    --out-dir dataset \
    --prefix rec1
```
Prints how many clips were kept vs. skipped (too short/too long) and the total kept duration.
Check `dataset/metadata_debug.csv` afterwards — it has the Urdu + Devanagari text side by side
with timing, useful for spot-checking a few clips by ear against their text.

**Adding more recordings of the same speaker to grow the dataset:** run stages 1-3 again for
each new audio file, using a different `--prefix` (so filenames don't collide) and `--append`
on stage 3 (so you don't overwrite the existing metadata):
```bash
python scripts/01_transcribe_urdu.py --audio /path/to/recording2.wav --out transcript_ur_2.json
python scripts/02_transliterate_ur_to_hi.py --in transcript_ur_2.json --sample 8   # sanity check first
python scripts/02_transliterate_ur_to_hi.py --in transcript_ur_2.json --out transcript_hi_2.json --all
python scripts/03_prepare_dataset.py \
    --audio /path/to/recording2.wav \
    --transcript transcript_hi_2.json \
    --out-dir dataset \
    --prefix rec2 \
    --append
```

### 4. Fine-tune Chatterbox
_Pending._

### 5. Generate dubbed audio
_Pending._

## Important caveats

- The Urdu→Devanagari transliteration relies partly on an external API
  (`indo_arabic_transliteration.sangam_api`) for best accuracy. **Always sanity-check a small
  sample of the output by eye before running the full dataset through it** — a bad converter is
  exactly what broke the first attempt at this approach.
- `chatterbox-finetuning`'s exact config field names and inference API should be double-checked
  against the actual cloned repo/installed package version once you're on the server — this
  README documents the intended usage, but third-party APIs can drift from what's summarized
  here.
