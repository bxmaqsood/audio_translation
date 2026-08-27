"""
Stage 5: Generate the final English dub in the cloned voice, with each
sentence's audio anchored to start at the same timestamp its corresponding
Urdu sentence started at in the original recording.

Timing rules (do not change without re-reading this comment -- these were
specified deliberately, not arbitrary defaults):
  - Audio is NEVER sped up or slowed down to fit a time slot. Every segment
    plays at its natural generated pace, full length.
  - Each segment's audio starts at max(its original Urdu start time, the
    point where the previous segment's audio actually finished). i.e. we
    anchor to the original timestamp whenever possible, but never rewind.
  - If English finishes before the original Urdu segment would have ended,
    the gap up to the next segment's anchor point is left as silence.
  - If English runs long enough to overlap the next segment's anchor point,
    the next segment simply starts right after the current one finishes
    (its own anchor is skipped for that one segment) -- so the total output
    can end up longer than the original recording, but is never shorter and
    never has anything played faster/slower than it was actually generated.

Run this in the `sooktam` conda environment.

Usage:
    python 05_generate_dub.py \
        --english transcript_en.json \
        --ref-file reference.wav \
        --ref-text "<accurate Urdu transcript of reference.wav>" \
        --out final_dub.wav

Optional --speed (e.g. --speed 0.65) uniformly slows down (or speeds up) each
GENERATED segment's own audio before it's placed into the timeline -- pitch
is preserved (ffmpeg's atempo filter, not a naive resample), so a slowed
voice sounds calmer/more measured, not deeper. This does not change the
timing RULES above: a slowed segment is just longer, so the existing
"anchor to original timestamp, never rewind, push later segments if this one
overran" logic handles it automatically -- no special-casing needed.

Re-runnable pipeline note: point --english at any transcript_en.json produced
by stage 4 (from this recording or a future one) -- no code changes needed.
"""
import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from transformers import AutoModel


def apply_speed(wav: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Pitch-preserving speed change via ffmpeg's atempo filter (valid range 0.5-100.0)."""
    if speed == 1.0:
        return wav
    with tempfile.TemporaryDirectory() as tmp_dir:
        in_path = os.path.join(tmp_dir, "in.wav")
        out_path = os.path.join(tmp_dir, "out.wav")
        sf.write(in_path, wav, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", in_path, "-filter:a", f"atempo={speed}", out_path],
            check=True,
        )
        out_wav, _ = sf.read(out_path, dtype="float32")
        return out_wav


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", default="transcript_en.json", help="English segments JSON (from stage 4)")
    parser.add_argument("--ref-file", required=True, help="Reference voice clip (short, clean, single sentence)")
    parser.add_argument("--ref-text", required=True, help="Accurate transcript of the reference clip")
    parser.add_argument("--out", default="final_dub.wav")
    parser.add_argument(
        "--model-id",
        default="/mnt/extra/bxm0694/sooktam2",
        help="Local path to the downloaded Sooktam-2 repo (or a HF hub id, but the local path "
        "avoids re-downloading and works regardless of your current directory)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier applied to each generated segment's audio before placement "
        "(e.g. 0.65 = 65%% speed, slower/longer; pitch is preserved). Default 1.0 = unchanged.",
    )
    args = parser.parse_args()

    with open(args.english, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"Loading {args.model_id} (this can take a minute)...")
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)

    sample_rate = 24000
    timeline = np.zeros(0, dtype=np.float32)  # grows as we place segments
    current_pos_sec = 0.0
    overruns = 0

    def place_audio(start_sec: float, audio: np.ndarray, sr: float):
        """Extend `timeline` (padding with silence if needed) and write `audio` at start_sec."""
        nonlocal timeline
        start_sample = int(round(start_sec * sr))
        end_sample = start_sample + len(audio)
        if end_sample > len(timeline):
            timeline = np.concatenate([timeline, np.zeros(end_sample - len(timeline), dtype=np.float32)])
        timeline[start_sample:end_sample] = audio

    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        original_start = seg["start"]

        actual_start = max(original_start, current_pos_sec)
        if actual_start > original_start + 0.05:  # more than a rounding blip
            overruns += 1

        if not text:
            # Nothing to say for this segment -- just leave silence, don't advance
            # current_pos_sec past this segment's own original end, so later
            # segments can still anchor normally if there's a gap.
            current_pos_sec = max(current_pos_sec, seg["end"])
            continue

        print(f"[{i + 1}/{len(segments)}] @{actual_start:6.1f}s  {text[:60]}")
        wav, sr, _ = model.infer(
            ref_file=args.ref_file,
            ref_text=args.ref_text,
            gen_text=text,
            tokenizer="cls",
            cls_language="english",
        )
        sample_rate = sr
        wav = np.asarray(wav, dtype=np.float32)
        wav = apply_speed(wav, sr, args.speed)

        place_audio(actual_start, wav, sr)
        current_pos_sec = actual_start + len(wav) / sr

    sf.write(args.out, timeline, sample_rate)
    original_total = segments[-1]["end"] if segments else 0.0
    print(f"\nWrote {args.out}")
    print(f"Original duration: {original_total / 60:.1f} min | Final dub duration: {len(timeline) / sample_rate / 60:.1f} min")
    print(f"{overruns} segment(s) ran long and pushed later segments back")


if __name__ == "__main__":
    main()
