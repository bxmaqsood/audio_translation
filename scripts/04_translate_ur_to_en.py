"""
Stage 4: Merge Whisper's pause-based segments into more complete sentence-like
units, then translate each unit to English with IndicTrans2 (free, local,
dedicated Urdu-to-English MT model).

Why merge first: Whisper splits audio on PAUSES in speech, not grammatical
sentence boundaries, so many raw segments are incomplete clauses (e.g.
"کے زمن میں" = "in the time of"). Translating those in isolation produces
broken or misleading fragments no matter how good the translator is. This
merges consecutive segments into units using, in priority order: (1) Urdu
sentence-ending punctuation if the segment has any, (2) an unusually long
pause before the next segment (a real breath/topic break), (3) a fallback
word/duration cap if neither signal shows up. Each merged unit still carries
the ORIGINAL start time of its first sub-segment as its anchor point, so
stage 5's timestamp-based audio-sync anchoring is unaffected -- we're just
choosing better (whole-sentence) anchor points instead of mid-sentence ones.

This writes a plain, editable JSON file BEFORE any audio is generated, on
purpose: check it over (and hand-fix any translation you don't like) before
running stage 5, since stage 5 just reads whatever text is in this file.

Usage:
    python 04_translate_ur_to_en.py \
        --transcript transcript_ur.json \
        --out transcript_en.json

Output JSON shape (one entry per MERGED unit, not per raw Whisper segment):
    [
      {"id": 0, "start": 0.0, "end": 12.12, "text_ur": "...", "text": "<English>",
       "source_ids": [0, 1, 2]},
      ...
    ]

Re-runnable pipeline note: point --transcript at any transcript_ur.json
produced by stage 1 (from this recording or a future one) -- no code changes
needed.

Requires: pip install IndicTransToolkit
"""
import argparse
import json

import torch
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"
SRC_LANG = "urd_Arab"
TGT_LANG = "eng_Latn"

# Urdu (and Latin, for any mixed-script text) sentence-ending punctuation.
SENTENCE_END_CHARS = "۔؟!.?"
GAP_THRESHOLD_SEC = 0.3   # a pause at least this long before the next segment
                          # is treated as a likely natural sentence break
MAX_WORDS = 20            # fallback cap: force a break if a unit grows past this
MAX_DURATION_SEC = 12.0   # fallback cap: force a break past this much audio


def merge_segments(segments: list[dict]) -> list[list[dict]]:
    """Group raw Whisper segments into sentence-like units. See module docstring."""
    groups = []
    current = []

    for i, seg in enumerate(segments):
        current.append(seg)
        text = seg["text"].strip()

        ends_with_punctuation = bool(text) and text[-1] in SENTENCE_END_CHARS

        gap_to_next = segments[i + 1]["start"] - seg["end"] if i + 1 < len(segments) else None
        long_pause_next = gap_to_next is not None and gap_to_next >= GAP_THRESHOLD_SEC

        word_count = sum(len(s["text"].split()) for s in current)
        duration = current[-1]["end"] - current[0]["start"]
        at_cap = word_count >= MAX_WORDS or duration >= MAX_DURATION_SEC

        is_last = i + 1 == len(segments)

        if ends_with_punctuation or long_pause_next or at_cap or is_last:
            groups.append(current)
            current = []

    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", default="transcript_ur.json", help="Urdu segments from stage 1")
    parser.add_argument("--out", default="transcript_en.json")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    with open(args.transcript, "r", encoding="utf-8") as f:
        ur_segments = json.load(f)

    groups = merge_segments(ur_segments)
    print(f"Merged {len(ur_segments)} raw segments into {len(groups)} sentence-like units")

    merged = [
        {
            "id": i,
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text_ur": " ".join(s["text"].strip() for s in group),
            "source_ids": [s["id"] for s in group],
        }
        for i, group in enumerate(groups)
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device} (this can take a minute)...")
    ip = IndicProcessor(inference=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    if device == "cuda":
        model.half()
    model.eval()

    texts = [m["text_ur"] for m in merged]
    translations = [""] * len(texts)

    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start : start + args.batch_size]
        batch = ip.preprocess_batch(batch_texts, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
        inputs = tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True
        ).to(device)

        with torch.no_grad():
            tokens = model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5)

        decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        batch_translations = ip.postprocess_batch(decoded, lang=TGT_LANG)
        translations[start : start + len(batch_translations)] = batch_translations

        done = min(start + args.batch_size, len(texts))
        print(f"  ...{done}/{len(texts)} units translated")

    for m, translation in zip(merged, translations):
        m["text"] = translation.strip()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(merged)} segments to {args.out}")
    print("Spot-check a few below, then hand-edit the file if anything needs fixing:\n")
    for seg in merged[:8]:
        print(f"  UR: {seg['text_ur']}")
        print(f"  EN: {seg['text']}\n")


if __name__ == "__main__":
    main()
