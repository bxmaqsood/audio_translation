"""
Stage 1: Transcribe raw Urdu audio with Whisper, producing segment-level
timestamps + Urdu text. This is the input to the transliteration stage.

Usage:
    python 01_transcribe_urdu.py --audio /mnt/extra/bxm0694/speaker.wav \
        --out transcript_ur.json --model large-v3

Output JSON shape:
    [
      {"id": 0, "start": 0.0, "end": 4.32, "text": "..."},
      ...
    ]
"""
import argparse
import json

import whisper


def transcribe(audio_path: str, model_name: str) -> list[dict]:
    model = whisper.load_model(model_name)
    result = model.transcribe(
        audio_path,
        language="ur",
        task="transcribe",
        word_timestamps=False,
        verbose=False,
    )
    segments = [
        {
            "id": seg["id"],
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
    ]
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Path to the raw Urdu audio file")
    parser.add_argument("--out", default="transcript_ur.json", help="Output JSON path")
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model size (large-v3 recommended for Urdu accuracy)",
    )
    args = parser.parse_args()

    segments = transcribe(args.audio, args.model)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(segments)} segments to {args.out}")
    print("Spot-check a few segments below before moving to transliteration:")
    for seg in segments[:5]:
        print(f"  [{seg['start']:>7.2f}-{seg['end']:>7.2f}] {seg['text']}")


if __name__ == "__main__":
    main()
