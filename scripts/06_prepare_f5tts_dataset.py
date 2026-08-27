"""
Stage 6: Build an F5-TTS-format training dataset (audio_file|text CSV, real
Urdu script, absolute wav paths) for fine-tuning Sooktam-2 on the speaker's
actual voice/prosody -- this is separate from stages 4-5's Devanagari/English
dubbing work; this dataset trains the model to speak like this speaker more
naturally (see README for why zero-shot cloning alone reads flat/"like
reading").

Reuses the same pause-based sentence grouping as stage 4 (see that script's
docstring for why pure pause timing, not punctuation/duration caps, is used)
so training clips are natural, complete-sounding sentences.

Usage:
    python 06_prepare_f5tts_dataset.py \
        --audio /mnt/extra/bxm0694/40_minutes_training_audio.m4a \
        --transcript transcript_ur.json \
        --out-dir /mnt/extra/bxm0694/f5tts_dataset

Produces:
    <out-dir>/wavs/seg_0000.wav, ...
    <out-dir>/metadata.csv     # "audio_file|text" header, absolute wav paths, real Urdu text

Feed metadata.csv into F5-TTS's own prepare_csv_wavs.py next (see README).
"""
import argparse
import csv
import os

from pydub import AudioSegment

GAP_THRESHOLD_SEC = 0.2  # same threshold validated in stage 4


def group_words_by_pause(words: list[dict], gap_threshold_sec: float) -> list[list[dict]]:
    groups = []
    current = []
    for i, w in enumerate(words):
        current.append(w)
        gap_to_next = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else None
        paused = gap_to_next is not None and gap_to_next >= gap_threshold_sec
        is_last = i + 1 == len(words)
        if paused or is_last:
            groups.append(current)
            current = []
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Path to the full source audio file")
    parser.add_argument("--transcript", default="transcript_ur.json", help="Output of stage 1 (needs word-level timing)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD_SEC)
    parser.add_argument("--min-dur", type=float, default=1.0, help="Skip clips shorter than this (s)")
    parser.add_argument("--max-dur", type=float, default=15.0, help="Skip clips longer than this (s)")
    parser.add_argument("--sample-rate", type=int, default=24000, help="F5-TTS expects 24kHz")
    args = parser.parse_args()

    import json

    with open(args.transcript, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = data["words"]

    groups = group_words_by_pause(words, args.gap_threshold)
    print(f"Grouped {len(words)} words into {len(groups)} pause-delimited sentences")

    wavs_dir = os.path.join(args.out_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    print(f"Loading source audio: {args.audio} (this can take a moment)...")
    audio = AudioSegment.from_file(args.audio)
    audio = audio.set_channels(1).set_frame_rate(args.sample_rate)

    kept, too_short, too_long = 0, 0, 0
    kept_duration_sec = 0.0
    rows = []

    for i, group in enumerate(groups):
        start, end = group[0]["start"], group[-1]["end"]
        duration = end - start
        if duration < args.min_dur:
            too_short += 1
            continue
        if duration > args.max_dur:
            too_long += 1
            continue

        text_ur = " ".join(w["word"] for w in group).strip()
        if not text_ur:
            continue

        clip = audio[int(start * 1000) : int(end * 1000)]
        filename = f"seg_{i:04d}.wav"
        abs_path = os.path.abspath(os.path.join(wavs_dir, filename))
        clip.export(abs_path, format="wav")

        rows.append((abs_path, text_ur))
        kept += 1
        kept_duration_sec += duration

    csv_path = os.path.join(args.out_dir, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text"])
        writer.writerows(rows)

    print(f"\nKept {kept} clips ({kept_duration_sec / 60:.1f} minutes), skipped {too_short} too-short / {too_long} too-long")
    print(f"Wrote {csv_path}")
    print(f"Wav clips in {wavs_dir}/")


if __name__ == "__main__":
    main()
