"""
Stage 4: Translate the Urdu audio directly to English text using Whisper's
built-in translate mode (audio -> English text, in one step, no separate
text-translation model needed).

Usage:
    python 04_translate_ur_to_en.py --audio /mnt/extra/bxm0694/40_minutes_training_audio.m4a \
        --out transcript_en.json --model large-v3

Output JSON shape (same shape as stage 1, but text is English):
    [
      {"id": 0, "start": 0.0, "end": 4.32, "text": "..."},
      ...
    ]

Note: Whisper's translate task only translates INTO English (not other target
languages) — that's all we need for the English dubbing goal. Segment
boundaries here are independent of stage 1's Urdu segments (Whisper re-does
its own segmentation for the translation), which is fine since dubbing here
doesn't require matching original segment timing.
"""
import argparse
import json

import whisper


def translate(audio_path: str, model_name: str) -> list[dict]:
    model = whisper.load_model(model_name)
    result = model.transcribe(
        audio_path,
        task="translate",
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
    parser.add_argument("--audio", required=True, help="Path to the source Urdu audio file")
    parser.add_argument("--out", default="transcript_en.json", help="Output JSON path")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    args = parser.parse_args()

    segments = translate(args.audio, args.model)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(segments)} English segments to {args.out}")
    print("Spot-check a few segments below:")
    for seg in segments[:5]:
        print(f"  [{seg['start']:>7.2f}-{seg['end']:>7.2f}] {seg['text']}")


if __name__ == "__main__":
    main()
