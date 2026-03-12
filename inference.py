import os
import time
import argparse
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from dataset import SegDataset


ROOT = os.getcwd()
DATA_DIR = "data"
DATASET_CRACKS_DIR = "cracks.v1i.yolov8"
DATASET_DRYWALL_DIR = "Drywall-Join-Detect.v1i.yolov8"
PROMPT_DIR = "prompts/augmented_prompts.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/train__CLIPSeg__seed42__1773259795_best.pt",
                        help="Path to trained checkpoint (.pt file)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference timing (default: 1)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--warmup-batches", type=int, default=3,
                        help="Number of warmup batches before timing starts")
    return parser.parse_args()


def collate_fn(batch):
    return {
        "image":    [b["image"] for b in batch],
        "mask":     torch.stack([b["mask"] for b in batch]),
        "prompt":   [b["prompt"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }


def load_model(checkpoint_path, device):
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, processor


@torch.no_grad()
def time_loader(model, processor, loader, device, label, warmup_batches):
    """Run inference over all batches in loader and return per-batch times in ms."""

    # ── Warmup ───────────────────────────────────────────────────────────────
    print(f"  Warming up ({warmup_batches} batches)...")
    for i, batch in enumerate(loader):
        if i >= warmup_batches:
            break
        inputs = processor(
            text=batch["prompt"], images=batch["image"],
            return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _ = model(**inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # ── Timed loop ────────────────────────────────────────────────────────────
    batch_times = []

    for batch in loader:
        inputs = processor(
            text=batch["prompt"], images=batch["image"],
            return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if device.type == "cuda":
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        outputs = model(**inputs)
        logits = outputs.logits
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        probs = torch.sigmoid(logits)
        _ = (probs > 0.5).float()

        if device.type == "cuda":
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        batch_times.append((t_end - t_start) * 1000.0)

    times = np.array(batch_times)
    bs = loader.batch_size

    print(f"\n  [{label}]")
    print(f"  Batch size      : {bs}")
    print(f"  Batches timed   : {len(times)}")
    print(f"  Mean / batch    : {times.mean():.1f} ms")
    print(f"  Median / batch  : {np.median(times):.1f} ms")
    print(f"  Std dev         : {times.std():.1f} ms")
    print(f"  Min / Max       : {times.min():.1f} ms / {times.max():.1f} ms")
    print(f"  Mean / image    : {times.mean() / bs:.1f} ms")

    return times


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"\nDevice    : {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Batch size: {args.batch_size}")

    model, processor = load_model(args.checkpoint, device)

    prompts_path = os.path.join(ROOT, DATA_DIR, PROMPT_DIR)

    # ── Test datasets (fixed canonical prompts, no augmentation) ─────────────
    cracks_test_dataset = SegDataset(
        os.path.join(ROOT, DATA_DIR, DATASET_CRACKS_DIR, "test"),
        prompts_path=prompts_path,
        fixed_prompt="segment crack",
    )

    cracks_test_loader = DataLoader(
        cracks_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    print("\n── Inference Timing ──────────────────────────────────────────────")

    cracks_times = time_loader(
        model, processor, cracks_test_loader, device,
        label="Cracks  (prompt: 'segment crack')",
        warmup_batches=args.warmup_batches,
    )

    all_times = cracks_times
    bs = args.batch_size

    print("\n  [Overall]")
    print(f"  Total batches   : {len(all_times)}")
    print(f"  Mean / batch    : {all_times.mean():.1f} ms")
    print(f"  Mean / image    : {all_times.mean() / bs:.1f} ms  (batch_size={bs})")
    print("──────────────────────────────────────────────────────────────────\n")