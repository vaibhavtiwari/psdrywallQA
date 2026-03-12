import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


VALID_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default= "checkpoints/train__CLIPSeg__seed42__1773259795_best.pt", 
                        help="Path to trained checkpoint")
    parser.add_argument("--input-dir", type=str, default="data/Drywall-Join-Detect.v1i.yolov8/valid/images", 
                        help="Directory containing input images")
    parser.add_argument("--output-dir", type=str, default="predictions",
                        help="Directory to save PNG masks")
    parser.add_argument("--prompt", type=str, default="segment taping area",
                        help='Prompt, e.g. "segment crack"')
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def sanitize_prompt(prompt: str) -> str:
    return (
        prompt.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def load_model(checkpoint_path: str, device: torch.device):
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, processor


def list_images(input_dir: str):
    paths = sorted(
        p for p in Path(input_dir).glob("*")
        if p.suffix.lower() in VALID_EXTS
    )
    if not paths:
        raise ValueError(f"No images found in {input_dir}")
    return paths


@torch.no_grad()
def predict_mask(model, processor, image_rgb: np.ndarray, prompt: str, device: torch.device, threshold: float):
    h, w = image_rgb.shape[:2]

    inputs = processor(
        text=[prompt],
        images=[image_rgb],
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits

    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    elif logits.dim() != 4:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()

    pred = F.interpolate(
        pred,
        size=(h, w),
        mode="nearest"
    )

    mask = pred[0, 0].cpu().numpy().astype(np.uint8) * 255
    return mask


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    model, processor = load_model(args.checkpoint, device)
    image_paths = list_images(args.input_dir)
    prompt_slug = sanitize_prompt(args.prompt)

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = predict_mask(model, processor, image_rgb, args.prompt, device, args.threshold)

        out_name = f"{image_path.stem}__{prompt_slug}.png"
        out_path = Path(args.output_dir) / out_name

        # single-channel PNG with values {0,255}
        cv2.imwrite(str(out_path), mask)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()