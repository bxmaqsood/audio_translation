"""
Stage 4: Translate each Urdu segment's TEXT to English using IndicTrans2
(ai4bharat/indictrans2-indic-en-1B) -- a dedicated Urdu/Indic-to-English
machine translation model, not Whisper's general-purpose translate mode.

We translate the already-accurate Urdu TEXT from stage 1 (transcript_ur.json),
not the raw audio -- this is both higher quality (IndicTrans2 is a purpose-
built MT model for exactly this language pair) and avoids Whisper's translate
mode randomly misdetecting the language on very short clips.

Segments map 1:1 to transcript_ur.json's ids/timestamps, which stage 5 needs
to anchor English audio to the original Urdu timing.

This writes a plain, editable JSON file BEFORE any audio is generated, on
purpose: check it over (and hand-fix any translation you don't like) before
running stage 5, since stage 5 just reads whatever text is in this file.

Usage:
    python 04_translate_ur_to_en.py \
        --transcript transcript_ur.json \
        --out transcript_en.json

Output JSON shape (one entry per Urdu segment, same id/start/end):
    [
      {"id": 0, "start": 0.0, "end": 4.44, "text_ur": "...", "text": "<English>"},
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", default="transcript_ur.json", help="Urdu segments from stage 1")
    parser.add_argument("--out", default="transcript_en.json")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    with open(args.transcript, "r", encoding="utf-8") as f:
        ur_segments = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device} (this can take a minute)...")
    ip = IndicProcessor(inference=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    if device == "cuda":
        model.half()
    model.eval()

    texts = [seg["text"] for seg in ur_segments]
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
        print(f"  ...{done}/{len(texts)} segments translated")

    en_segments = [
        {
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text_ur": seg["text"],
            "text": translation.strip(),
        }
        for seg, translation in zip(ur_segments, translations)
    ]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(en_segments, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(en_segments)} segments to {args.out}")
    print("Spot-check a few below, then hand-edit the file if anything needs fixing:\n")
    for seg in en_segments[:5]:
        print(f"  UR: {seg['text_ur']}")
        print(f"  EN: {seg['text']}\n")


if __name__ == "__main__":
    main()
