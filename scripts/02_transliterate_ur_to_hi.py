"""
Stage 2: Transliterate each Urdu segment's text into Devanagari (Hindi script).

This is NOT translation — the words and meaning stay Urdu, we're just re-spelling
them in the script Chatterbox already knows, so training data still matches the
audio exactly.

Usage:
    # Sanity check a handful of segments first (recommended before --all):
    python 02_transliterate_ur_to_hi.py --in transcript_ur.json --sample 8

    # Full run once the sample output looks right:
    python 02_transliterate_ur_to_hi.py --in transcript_ur.json --out transcript_hi.json --all

Output JSON shape (adds "text_hi" to each segment):
    [
      {"id": 0, "start": 0.0, "end": 4.32, "text": "...", "text_hi": "..."},
      ...
    ]
"""
import argparse
import hashlib
import json
import os
import random
import time

CACHE_PATH = "transliteration_cache.json"


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def transliterate_one(text: str, cache: dict, retries: int = 2) -> tuple[str, str]:
    """Returns (devanagari_text, method_used). Caches by text hash."""
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]["text_hi"], cache[key]["method"]

    result, method = None, None

    # Preferred: online API (better accuracy, handles short vowels)
    try:
        from indo_arabic_transliteration.sangam_api import online_transliterate

        for attempt in range(retries + 1):
            try:
                result = online_transliterate(text, "ur-PK", "hi-IN")
                method = "online"
                break
            except Exception:
                if attempt < retries:
                    time.sleep(1.5)
    except ImportError:
        pass

    # Fallback: offline rule-based (faster, weaker on short vowels)
    if not result:
        from indo_arabic_transliteration.mapper import script_convert

        result = script_convert(text, "ur-PK", "hi-IN")
        method = "offline_fallback"

    cache[key] = {"text_hi": result, "method": method}
    return result, method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="infile", default="transcript_ur.json")
    parser.add_argument("--out", dest="outfile", default="transcript_hi.json")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Only transliterate N random segments and print them for manual review "
        "(does not write the output file). Use this before --all.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Process every segment and write --out"
    )
    args = parser.parse_args()

    with open(args.infile, "r", encoding="utf-8") as f:
        segments = json.load(f)

    cache = load_cache()

    if args.sample:
        chosen = random.sample(segments, min(args.sample, len(segments)))
        print(f"Sampling {len(chosen)} segments — check these read correctly as Hindustani:\n")
        for seg in chosen:
            hi, method = transliterate_one(seg["text"], cache)
            print(f"  UR [{method}]: {seg['text']}")
            print(f"  HI          : {hi}\n")
        save_cache(cache)
        print("Re-run with --all once this output looks correct.")
        return

    if not args.all:
        parser.error("Pass --sample N to preview first, or --all to process everything.")

    online_count, offline_count = 0, 0
    for i, seg in enumerate(segments):
        hi, method = transliterate_one(seg["text"], cache)
        seg["text_hi"] = hi
        online_count += method == "online"
        offline_count += method == "offline_fallback"
        if (i + 1) % 50 == 0:
            save_cache(cache)
            print(f"  ...{i + 1}/{len(segments)} done")

    save_cache(cache)
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(segments)} segments to {args.outfile} "
        f"({online_count} via online API, {offline_count} via offline fallback)"
    )
    if offline_count:
        print(
            f"WARNING: {offline_count} segments used the weaker offline fallback "
            "(short vowels may be missing) — spot-check those before training."
        )


if __name__ == "__main__":
    main()
