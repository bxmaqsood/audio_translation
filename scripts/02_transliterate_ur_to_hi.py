"""
Stage 2: Transliterate each Urdu segment's text into Devanagari (Hindi script).

This is NOT translation — the words and meaning stay Urdu, we're just re-spelling
them in the script Chatterbox already knows, so training data still matches the
audio exactly.

Notes on the underlying `indo_arabic_transliteration` package:
  - Its "online" mode calls a legacy academic API (learnpunjabi.org) that is
    currently returning server errors (HTTP 500) — dead, not something we can
    fix. Disabled by default; pass --try-online to attempt it anyway.
  - Its "offline" mode works, but importing it drags in an unrelated, broken
    dependency chain (urduhack -> tf2crf -> an abandoned tensorflow_addons
    build incompatible with the installed Keras). We stub out the two unused
    broken modules below so the import succeeds without needing any of that.
  - The offline converter occasionally emits stray non-Devanagari symbols
    (e.g. `ʿ` for the Arabic letter ain) or rare/unusual glyphs. We strip the
    known stray marks and flag any segment whose result still contains
    characters outside normal Hindi text, so those can be reviewed by hand
    instead of silently going into the training set.

Usage:
    # Sanity check a handful of segments first (recommended before --all):
    python 02_transliterate_ur_to_hi.py --in transcript_ur.json --sample 8

    # Full run once the sample output looks right:
    python 02_transliterate_ur_to_hi.py --in transcript_ur.json --out transcript_hi.json --all

Output JSON shape (adds "text_hi" to each segment):
    [
      {"id": 0, "start": 0.0, "end": 4.32, "text": "...", "text_hi": "...", "flagged": false},
      ...
    ]
Segments needing manual review are also written to flagged_segments.json.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import types

# --- Stub out the broken, unused NER dependency chain before importing the package ---
sys.modules.setdefault("tensorflow_addons", types.ModuleType("tensorflow_addons"))
_tf2crf_stub = types.ModuleType("tf2crf")
_tf2crf_stub.CRF = object
sys.modules.setdefault("tf2crf", _tf2crf_stub)

CACHE_PATH = "transliteration_cache.json"

# Stray marks the offline converter leaves behind that aren't real Devanagari
STRAY_MARKS = re.compile(r"[ʿʾ]")  # ʿ (ain) and ʾ (hamza) modifier letters

# Characters considered normal in transliterated Hindustani text: the commonly-used
# Devanagari subset (U+0900-U+096F: vowels, consonants, nukta letters, matras,
# digits, danda) plus whitespace and basic ASCII. Deliberately excludes U+0970+,
# which is rare additions for other languages (e.g. U+0979 "ॹ" for Dogri/Kashmiri)
# that a correct Hindustani transliteration should never actually produce.
ALLOWED_CHARS = re.compile(r"^[ऀ-९ -~\s]*$")


def sanitize(text_hi: str) -> tuple[str, bool]:
    """Strip known stray marks; return (cleaned_text, needs_review)."""
    cleaned = STRAY_MARKS.sub("", text_hi)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    needs_review = not bool(ALLOWED_CHARS.match(cleaned))
    return cleaned, needs_review


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def transliterate_one(text: str, cache: dict, try_online: bool, retries: int = 2) -> dict:
    """Returns {"text_hi": ..., "method": ..., "needs_review": bool}. Cached by text hash."""
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if key in cache:
        return cache[key]

    result, method = None, None

    if try_online:
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

    if not result:
        from indo_arabic_transliteration.mapper import script_convert

        result = script_convert(text, "ur-PK", "hi-IN")
        method = "offline"

    cleaned, needs_review = sanitize(result)
    entry = {"text_hi": cleaned, "method": method, "needs_review": needs_review}
    cache[key] = entry
    return entry


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
    parser.add_argument(
        "--try-online",
        action="store_true",
        help="Attempt the (currently broken) online API before falling back offline",
    )
    args = parser.parse_args()

    with open(args.infile, "r", encoding="utf-8") as f:
        segments = json.load(f)

    cache = load_cache()

    if args.sample:
        chosen = random.sample(segments, min(args.sample, len(segments)))
        print(f"Sampling {len(chosen)} segments — check these read correctly as Hindustani:\n")
        for seg in chosen:
            r = transliterate_one(seg["text"], cache, args.try_online)
            flag = "  [FLAGGED - unusual chars]" if r["needs_review"] else ""
            print(f"  UR [{r['method']}]: {seg['text']}")
            print(f"  HI          : {r['text_hi']}{flag}\n")
        save_cache(cache)
        print("Re-run with --all once this output looks correct.")
        return

    if not args.all:
        parser.error("Pass --sample N to preview first, or --all to process everything.")

    flagged = []
    for i, seg in enumerate(segments):
        r = transliterate_one(seg["text"], cache, args.try_online)
        seg["text_hi"] = r["text_hi"]
        seg["flagged"] = r["needs_review"]
        if r["needs_review"]:
            flagged.append(seg)
        if (i + 1) % 50 == 0:
            save_cache(cache)
            print(f"  ...{i + 1}/{len(segments)} done")

    save_cache(cache)
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    if flagged:
        with open("flagged_segments.json", "w", encoding="utf-8") as f:
            json.dump(flagged, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(segments)} segments to {args.outfile}")
    print(
        f"{len(flagged)}/{len(segments)} segments flagged for manual review "
        f"-> flagged_segments.json"
        if flagged
        else "No segments flagged — all output looked like clean Devanagari."
    )


if __name__ == "__main__":
    main()
