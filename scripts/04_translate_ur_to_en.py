"""
Stage 4: Translate each Urdu segment to English, one segment at a time, so
English segments map 1:1 to the original Urdu segments' timestamps (needed
for stage 5 to anchor each English sentence to when the corresponding Urdu
sentence started).

Translates the audio clip for each segment individually (Whisper's translate
task, audio -> English text) rather than translating the whole file at once
-- this keeps segment boundaries identical to transcript_ur.json's, at the
cost of losing a little cross-sentence context for the translation itself
(short/ambiguous segments may translate a bit more literally as a result --
review the output and hand-edit transcript_en.json if anything reads oddly).

This writes a plain, editable JSON file BEFORE any audio is generated, on
purpose: check it over (and fix any translation you don't like) before
running stage 5, since stage 5 just reads whatever text is in this file.

Usage:
    python 04_translate_ur_to_en.py \
        --audio /mnt/extra/bxm0694/40_minutes_training_audio.m4a \
        --transcript transcript_ur.json \
        --out transcript_en.json \
        --model large-v3

Output JSON shape (one entry per original Urdu segment, same id/start/end):
    [
      {"id": 0, "start": 0.0, "end": 4.44, "text_ur": "...", "text": "<English>"},
      ...
    ]

Re-runnable pipeline note: this script only needs --audio and a matching
--transcript (from stage 1) -- point it at a new recording's files and it
works the same way, so this whole pipeline (stages 1, 4, 5) is reusable for
future Urdu audio without any code changes.
"""
import argparse
import json

import whisper
from pydub import AudioSegment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Path to the source Urdu audio file")
    parser.add_argument("--transcript", default="transcript_ur.json", help="Urdu segments from stage 1")
    parser.add_argument("--out", default="transcript_en.json")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    args = parser.parse_args()

    with open(args.transcript, "r", encoding="utf-8") as f:
        ur_segments = json.load(f)

    print(f"Loading source audio: {args.audio}...")
    audio = AudioSegment.from_file(args.audio)

    print(f"Loading Whisper model: {args.model}...")
    model = whisper.load_model(args.model)

    en_segments = []
    for i, seg in enumerate(ur_segments):
        clip = audio[int(seg["start"] * 1000) : int(seg["end"] * 1000)]
        clip_path = "_translate_tmp_clip.wav"
        clip.export(clip_path, format="wav")

        result = model.transcribe(clip_path, task="translate", verbose=False)
        english_text = result["text"].strip()

        en_segments.append(
            {
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text_ur": seg["text"],
                "text": english_text,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(ur_segments)} segments translated")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(en_segments, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(en_segments)} segments to {args.out}")
    print("Spot-check a few below, then hand-edit the file if anything needs fixing:\n")
    for seg in en_segments[:5]:
        print(f"  UR: {seg['text_ur']}")
        print(f"  EN: {seg['text']}\n")


if __name__ == "__main__":
    main()
