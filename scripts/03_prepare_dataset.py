"""
Stage 3: Slice the source audio into per-segment clips and build the
LJSpeech-style dataset that chatterbox-finetuning expects.

Usage:
    python 03_prepare_dataset.py \
        --audio /mnt/extra/bxm0694/40_minutes_training_audio.m4a \
        --transcript transcript_hi.json \
        --out-dir dataset

Produces:
    dataset/
      wavs/seg_0000.wav, seg_0001.wav, ...
      metadata.csv          # filename|raw_text|normalized_text (both = Devanagari)
      metadata_debug.csv    # filename|start|end|urdu_text|devanagari_text (for review)

Segments shorter than --min-dur or longer than --max-dur are skipped (their
audio is too short to be useful, or too long for stable TTS training) — counts
are printed at the end. The chatterbox-finetuning toolkit's own README notes
3-10s per clip as optimal; we keep a wider [1s, 15s] range by default since
this is a modest ~40-minute dataset and dropping too much data has its own
cost — tighten with --min-dur/--max-dur if you want to match their guidance
more strictly.
"""
import argparse
import csv
import os

from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def trim_silence(
    clip: AudioSegment, silence_thresh_offset: int = 16, min_silence_len: int = 100
) -> AudioSegment:
    """Trim leading/trailing silence from a clip based on where non-silent audio starts/ends."""
    thresh = clip.dBFS - silence_thresh_offset if clip.dBFS != float("-inf") else -40
    nonsilent_ranges = detect_nonsilent(clip, min_silence_len=min_silence_len, silence_thresh=thresh)
    if not nonsilent_ranges:
        return clip
    start, end = nonsilent_ranges[0][0], nonsilent_ranges[-1][1]
    return clip[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Path to the full source audio file")
    parser.add_argument("--transcript", default="transcript_hi.json")
    parser.add_argument("--out-dir", default="dataset")
    parser.add_argument("--min-dur", type=float, default=1.0, help="Skip clips shorter than this (s)")
    parser.add_argument("--max-dur", type=float, default=15.0, help="Skip clips longer than this (s)")
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args()

    import json

    with open(args.transcript, "r", encoding="utf-8") as f:
        segments = json.load(f)

    wavs_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    print(f"Loading source audio: {args.audio} (this can take a moment for long files)...")
    audio = AudioSegment.from_file(args.audio)
    audio = audio.set_channels(1).set_frame_rate(args.sample_rate)

    kept, too_short, too_long = 0, 0, 0
    metadata_rows = []
    debug_rows = []

    for seg in segments:
        duration = seg["end"] - seg["start"]
        if duration < args.min_dur:
            too_short += 1
            continue
        if duration > args.max_dur:
            too_long += 1
            continue

        clip = audio[int(seg["start"] * 1000) : int(seg["end"] * 1000)]
        clip = trim_silence(clip)
        if len(clip) < args.min_dur * 1000:
            too_short += 1
            continue

        filename = f"seg_{seg['id']:04d}.wav"
        clip.export(os.path.join(wavs_dir, filename), format="wav")

        text_hi = seg["text_hi"]
        metadata_rows.append((filename.replace(".wav", ""), text_hi, text_hi))
        debug_rows.append((filename, seg["start"], seg["end"], seg["text"], text_hi))
        kept += 1

    metadata_path = os.path.join(args.out_dir, "metadata.csv")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(metadata_rows)

    debug_path = os.path.join(args.out_dir, "metadata_debug.csv")
    with open(debug_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["filename", "start", "end", "urdu_text", "devanagari_text"])
        writer.writerows(debug_rows)

    total_kept_duration = sum(r[2] - r[1] for r in debug_rows) / 60
    print(f"\nKept {kept} clips ({total_kept_duration:.1f} minutes of audio)")
    print(f"Skipped {too_short} too-short, {too_long} too-long segments")
    print(f"Wrote {metadata_path} and {debug_path}")
    print(f"Wav clips in {wavs_dir}/")


if __name__ == "__main__":
    main()
