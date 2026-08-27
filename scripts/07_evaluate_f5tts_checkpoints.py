"""
Stage 7: Evaluate saved F5-TTS fine-tuning checkpoints against a held-out
validation set (the 30 clips split off in metadata_val.csv, never seen during
training). Reports average validation loss per checkpoint, forward-pass only
(no gradient updates) -- so we can pick the checkpoint that actually
generalizes best, rather than the one with the lowest TRAINING loss (which
would likely just be the most overfit; see README for why).

F5-TTS's own trainer/finetune_cli.py has no built-in validation support at
all (confirmed by grepping for val/eval in trainer.py and finetune_cli.py --
nothing found), so this replicates just enough of its own model-construction
and loss-call logic (copied from finetune_cli.py's F5TTS_v1_Base branch) to
run the same loss function in eval mode.

Must be run from inside the F5-TTS repo directory (uses its `f5_tts` package
and its `data/` dataset convention).

Usage (run once per checkpoint you want to compare):
    cd /mnt/extra/bxm0694/F5-TTS
    for ckpt in ckpts/urdu_speaker1/model_*.pt; do
        python /home/bxm0694/audio_translation/scripts/07_evaluate_f5tts_checkpoints.py \
            --checkpoint "$ckpt" \
            --val-dataset-name urdu_speaker1_val \
            --tokenizer-path /mnt/extra/bxm0694/sooktam2/vocab.txt
    done
"""
import argparse

import torch
from torch.utils.data import DataLoader

# Same numerical-stability fix as finetune_cli.py -- without this, the
# forward/backward pass can hit the SDPA NaN bug even in eval mode.
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from f5_tts.model import CFM, DiT  # noqa: E402
from f5_tts.model.dataset import collate_fn, load_dataset  # noqa: E402
from f5_tts.model.utils import get_tokenizer  # noqa: E402

# Copied from finetune_cli.py -- must match exactly or the checkpoint won't load
TARGET_SAMPLE_RATE = 24000
N_MEL_CHANNELS = 100
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024
MEL_SPEC_TYPE = "vocos"
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a saved .pt checkpoint")
    parser.add_argument(
        "--val-dataset-name",
        default="urdu_speaker1_val",
        help="Dataset name matching data/{name}_custom (from prepare_csv_wavs.py)",
    )
    parser.add_argument("--tokenizer-path", required=True, help="Path to vocab.txt")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_char_map, vocab_size = get_tokenizer(args.tokenizer_path, "custom")

    mel_spec_kwargs = dict(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mel_channels=N_MEL_CHANNELS,
        target_sample_rate=TARGET_SAMPLE_RATE,
        mel_spec_type=MEL_SPEC_TYPE,
    )

    model = CFM(
        transformer=DiT(**MODEL_CFG, text_num_embeds=vocab_size, mel_dim=N_MEL_CHANNELS),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Prefer EMA weights (what's actually used at inference time) if present
    if "ema_model_state_dict" in checkpoint:
        state_dict = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "update", "step"]
        }
    else:
        state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    val_dataset = load_dataset(args.val_dataset_name, "custom", mel_spec_kwargs=mel_spec_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            mel_spec = batch["mel"].permute(0, 2, 1).to(device)
            mel_lengths = batch["mel_lengths"].to(device)
            text_inputs = batch["text"]
            loss, _, _ = model(mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=None)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    print(f"{args.checkpoint}\tval_loss={avg_loss:.4f}\t({n_batches} batches)")


if __name__ == "__main__":
    main()
