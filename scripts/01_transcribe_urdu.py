"""
Stage 1: Transcribe raw Urdu audio with Whisper, producing WORD-level
timestamps -- this is what stage 4 uses to find real sentence breaks (see
its docstring): segment-level timestamps turned out to be nearly useless for
this (90% of segment-to-segment gaps are exactly zero -- Whisper snaps
segment boundaries together regardless of whether the speaker paused there),
so we need per-word timing to see the actual pauses.

Usage:
    python 01_transcribe_urdu.py --audio /mnt/extra/bxm0694/speaker.wav \
        --out transcript_ur.json --model large-v3

Output JSON shape:
    {
      "segments": [{"id": 0, "start": 0.0, "end": 4.32, "text": "..."}, ...],
      "words": [{"word": "...", "start": 0.0, "end": 0.34}, ...]
    }
"segments" is kept only for reference/debugging (e.g. spot-checking
transcription quality); "words" is what stage 4 actually uses.
"""
import argparse
import json

import whisper


def transcribe(audio_path: str, model_name: str) -> dict:
    model = whisper.load_model(model_name)
    result = model.transcribe(
        audio_path,
        language="ur",
        task="transcribe",
        word_timestamps=True,
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
    words = [
        {"word": w["word"].strip(), "start": round(w["start"], 3), "end": round(w["end"], 3)}
        for seg in result["segments"]
        for w in seg["words"]
    ]
    return {"segments": segments, "words": words}


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

    result = transcribe(args.audio, args.model)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(result['segments'])} segments / {len(result['words'])} words to {args.out}")
    print("Spot-check a few segments below before moving to transliteration:")
    for seg in result["segments"][:5]:
        print(f"  [{seg['start']:>7.2f}-{seg['end']:>7.2f}] {seg['text']}")


if __name__ == "__main__":
    main()
