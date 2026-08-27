"""
Stage 4: Group words into sentences based ONLY on where the speaker actually
paused (word-to-word timing gaps), then translate each sentence with
IndicTrans2 (free, local, dedicated Urdu-to-English MT model).

Why word-level pauses, and only that signal: Whisper's segment-level
timestamps turned out to be nearly useless for finding real sentence breaks
-- 90% of segment-to-segment gaps are exactly zero regardless of whether the
speaker paused there (Whisper snaps segment boundaries together). Word-level
timestamps reveal the real pauses: across a sample recording, 92% of
word-to-word gaps were ~zero (continuous speech) and there was a sharp,
clean boundary at 0.2s with NO gaps falling between 0.15s and 0.2s at all --
a natural dividing line between "still talking" and "the speaker paused."
GAP_THRESHOLD_SEC below should be re-checked (see stage 1's word list) if
this is ever run on a very differently-paced speaker/recording, but no other
signal (punctuation, word count, duration caps) is used -- pure pause timing
only, since that's the truest signal for both meaning AND natural audio
pacing/sync.

Each resulting sentence's start time is its first word's actual start time,
so stage 5's timestamp-based audio-sync anchoring lines up with where the
speaker really began speaking that sentence.

This writes a plain, editable JSON file BEFORE any audio is generated, on
purpose: check it over (and hand-fix any translation you don't like) before
running stage 5, since stage 5 just reads whatever text is in this file.

Usage:
    python 04_translate_ur_to_en.py \
        --transcript transcript_ur.json \
        --out transcript_en.json

Output JSON shape (one entry per pause-delimited sentence):
    [
      {"id": 0, "start": 0.0, "end": 4.44, "text_ur": "...", "text": "<English>"},
      ...
    ]

Re-runnable pipeline note: point --transcript at any transcript_ur.json
produced by stage 1 (from this recording or a future one) -- no code changes
needed. If a new recording has a noticeably different speaking pace, re-run
the gap-distribution check (see git history / README) before assuming 0.2s
still separates real pauses from normal word spacing.

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

GAP_THRESHOLD_SEC = 0.2  # see module docstring for how this was derived


def group_words_by_pause(words: list[dict]) -> list[dict]:
    """Split the word list into sentences wherever the pause before the next
    word is >= GAP_THRESHOLD_SEC. No other signal (punctuation, word count,
    duration) is used."""
    groups = []
    current = []

    for i, w in enumerate(words):
        current.append(w)
        gap_to_next = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else None
        paused = gap_to_next is not None and gap_to_next >= GAP_THRESHOLD_SEC
        is_last = i + 1 == len(words)

        if paused or is_last:
            groups.append(current)
            current = []

    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", default="transcript_ur.json", help="Output of stage 1")
    parser.add_argument("--out", default="transcript_en.json")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    with open(args.transcript, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = data["words"]

    groups = group_words_by_pause(words)
    print(f"Grouped {len(words)} words into {len(groups)} pause-delimited sentences")

    sentences = [
        {
            "id": i,
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text_ur": " ".join(w["word"] for w in group).strip(),
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

    texts = [s["text_ur"] for s in sentences]
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
        print(f"  ...{done}/{len(texts)} sentences translated")

    for s, translation in zip(sentences, translations):
        s["text"] = translation.strip()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(sentences)} segments to {args.out}")
    print("Spot-check a few below, then hand-edit the file if anything needs fixing:\n")
    for seg in sentences[:8]:
        print(f"  UR: {seg['text_ur']}")
        print(f"  EN: {seg['text']}\n")


if __name__ == "__main__":
    main()
