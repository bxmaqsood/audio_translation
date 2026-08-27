"""
Stage 5: Generate the final English dub in the cloned voice, using Sooktam-2's
zero-shot voice cloning (no fine-tuning required — this replaced the earlier
Chatterbox fine-tuning approach after Sooktam-2's native Urdu pretraining
proved good enough for zero-shot cloning; see README for why).

Run this in the `sooktam` conda environment, from anywhere (the model is
loaded via `transformers.AutoModel` + `trust_remote_code`, no need to be
inside the sooktam2 repo directory).

Usage:
    python 05_generate_dub.py \
        --english transcript_en.json \
        --ref-file /home/bxm0694/audio_translation/dataset/wavs/rec1_seg_0007.wav \
        --ref-text "کہ دین اسلام کی تمام" \
        --out final_dub.wav

--ref-file/--ref-text should be a short (3-6s), clean, single-sentence clip
of the speaker with its accurate transcript — this is the "voice sample" the
whole dub is cloned from, so pick a good one (see README for how we picked
ours). Segments are generated one at a time, with a short pause inserted
between them, then concatenated into one output file.
"""
import argparse
import json

import numpy as np
import soundfile as sf
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", default="transcript_en.json", help="English segments JSON (from stage 4)")
    parser.add_argument("--ref-file", required=True, help="Reference voice clip (short, clean, single sentence)")
    parser.add_argument("--ref-text", required=True, help="Accurate transcript of the reference clip")
    parser.add_argument("--out", default="final_dub.wav")
    parser.add_argument("--pause-seconds", type=float, default=0.3)
    parser.add_argument(
        "--model-id", default="bharatgenai/sooktam2", help="HF model id (or local path if already downloaded)"
    )
    args = parser.parse_args()

    with open(args.english, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"Loading {args.model_id} (this can take a minute)...")
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)

    chunks = []
    sample_rate = 24000

    for i, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue
        print(f"[{i + 1}/{len(segments)}] {text[:60]}")
        wav, sr, _ = model.infer(
            ref_file=args.ref_file,
            ref_text=args.ref_text,
            gen_text=text,
            tokenizer="cls",
            cls_language="english",
        )
        sample_rate = sr
        chunks.append(np.asarray(wav, dtype=np.float32))
        chunks.append(np.zeros(int(args.pause_seconds * sr), dtype=np.float32))

    if not chunks:
        print("No segments to synthesize — nothing written.")
        return

    final_audio = np.concatenate(chunks)
    sf.write(args.out, final_audio, sample_rate)
    print(f"\nWrote {args.out} ({len(final_audio) / sample_rate:.1f} seconds)")


if __name__ == "__main__":
    main()
