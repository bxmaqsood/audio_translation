"""
Stage 4: Translate each Urdu segment to English using the Claude API, with
full surrounding context -- not IndicTrans2 or Whisper's translate mode.

Why: Whisper's segments are split on PAUSES in speech, not grammatical
sentence boundaries, so many segments are incomplete clauses (e.g.
"کے زمن میں" = "in the time of"). Translating each one in total isolation
(what IndicTrans2 and Whisper's translate mode both do) produces broken or
misleading fragments. This script instead sends a BATCH of consecutive
segments at once (plus a little trailing context from the previous batch)
and asks Claude to translate each one so that, concatenated in order, they
read as one fluent, natural English passage -- while still returning exactly
one translation per input segment ID, so segment count and the original
Urdu timestamps are completely unchanged. This is what preserves per-segment
timestamp anchoring for stage 5's audio-sync requirement while still fixing
the translation-quality problem.

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

Requires: pip install anthropic
Requires: ANTHROPIC_API_KEY set in the environment (or `ant auth login`).
"""
import argparse
import json
import re
import time

import anthropic

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You are translating Urdu speech-to-text segments into English for a \
dubbing project. The segments come from an automatic speech recognizer that splits audio \
on PAUSES, not on grammatical sentence boundaries -- so many segments are incomplete \
clauses or sentence fragments, not full sentences.

Your job: translate each segment into English such that, when all segments are \
concatenated IN ORDER, the result reads as one fluent, natural, grammatically correct \
English passage -- as if written by a native English speaker, not a word-for-word \
fragment-by-fragment translation. Use the surrounding segments as context to resolve \
ambiguity and produce natural phrasing, but you MUST return exactly one translation per \
input segment id -- never merge segments together or split one segment into multiple \
outputs, even if the natural English wording would flow better merged. Some "context" \
segments are included only so you understand what comes immediately before the segments \
you need to translate -- do NOT include translations for context-only ids in your output.

Return ONLY a JSON array, no other text, no markdown code fences, in this exact shape:
[{"id": 0, "text": "..."}, {"id": 1, "text": "..."}, ...]
"""


def build_user_prompt(context_segments: list[dict], target_segments: list[dict]) -> str:
    lines = []
    if context_segments:
        lines.append("CONTEXT ONLY (do not translate these, just use for context):")
        for seg in context_segments:
            lines.append(f'  id {seg["id"]}: {seg["text"]}')
        lines.append("")
    lines.append("TRANSLATE THESE (return exactly one output per id, in this list):")
    for seg in target_segments:
        lines.append(f'  id {seg["id"]}: {seg["text"]}')
    return "\n".join(lines)


def extract_json_array(raw_text: str) -> list[dict]:
    raw_text = raw_text.strip()
    # Strip markdown fences if Claude adds them despite instructions
    raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip())
    return json.loads(raw_text)


def translate_batch(
    client: anthropic.Anthropic, context_segments: list[dict], target_segments: list[dict], retries: int = 3
) -> dict[int, str]:
    user_prompt = build_user_prompt(context_segments, target_segments)

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = next((b.text for b in response.content if b.type == "text"), "")
            parsed = extract_json_array(text)
            result = {item["id"]: item["text"] for item in parsed}

            target_ids = {seg["id"] for seg in target_segments}
            if set(result.keys()) != target_ids:
                raise ValueError(f"ID mismatch: expected {target_ids}, got {set(result.keys())}")
            return result
        except (json.JSONDecodeError, ValueError, KeyError, anthropic.APIStatusError) as e:
            print(f"  Batch attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", default="transcript_ur.json", help="Urdu segments from stage 1")
    parser.add_argument("--out", default="transcript_en.json")
    parser.add_argument("--batch-size", type=int, default=25, help="Segments translated per API call")
    parser.add_argument("--context-size", type=int, default=3, help="Trailing context segments per batch")
    args = parser.parse_args()

    with open(args.transcript, "r", encoding="utf-8") as f:
        ur_segments = json.load(f)

    client = anthropic.Anthropic()
    translations: dict[int, str] = {}

    for start in range(0, len(ur_segments), args.batch_size):
        target_segments = ur_segments[start : start + args.batch_size]
        context_start = max(0, start - args.context_size)
        context_segments = ur_segments[context_start:start]

        result = translate_batch(client, context_segments, target_segments)
        translations.update(result)

        done = min(start + args.batch_size, len(ur_segments))
        print(f"  ...{done}/{len(ur_segments)} segments translated")

    en_segments = [
        {
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text_ur": seg["text"],
            "text": translations[seg["id"]].strip(),
        }
        for seg in ur_segments
    ]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(en_segments, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(en_segments)} segments to {args.out}")
    print("Spot-check a few below, then hand-edit the file if anything needs fixing:\n")
    for seg in en_segments[:8]:
        print(f"  UR: {seg['text_ur']}")
        print(f"  EN: {seg['text']}\n")


if __name__ == "__main__":
    main()
