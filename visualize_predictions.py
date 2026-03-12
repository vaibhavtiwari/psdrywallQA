import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


VALID_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, default= "data/Drywall-Join-Detect.v1i.yolov8/valid/images", 
                        help="Directory with source images")
    parser.add_argument("--labels-dir", type=str, default="data/Drywall-Join-Detect.v1i.yolov8/valid/labels", 
                        help="Directory with YOLO label txt files (bbox or polygon)")
    parser.add_argument("--pred-dir", type=str,  default="predictions",
                        help="Directory with predicted PNG masks")
    parser.add_argument("--output-dir", type=str,  default="visulaize",
                        help="Directory to save report figures")
    parser.add_argument("--prompt", type=str,  default="segment taping area",
                        help='Prompt used for prediction, e.g. "segment crack"')
    parser.add_argument("--max-images", type=int, default=4, 
                        help="Number of figures to save")
    parser.add_argument("--overlay-alpha", type=float, default=0.45,
                        help="Opacity of the predicted mask overlay (0.0 = invisible, 1.0 = opaque)")
    parser.add_argument("--overlay-color", type=str, default="red",
                        choices=["red", "green", "blue", "yellow"],
                        help="Color of the predicted mask overlay")
    return parser.parse_args()


def sanitize_prompt(prompt: str) -> str:
    return (
        prompt.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def label_to_mask(label_path: Path, image_shape):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not label_path.exists():
        return mask

    with open(label_path, "r") as f:
        for line in f:
            values = list(map(float, line.split()))
            if not values:
                continue

            # YOLO detection bbox format: class cx cy bw bh
            if len(values) == 5:
                _, cx, cy, bw, bh = values

                x_center = cx * w
                y_center = cy * h
                box_w = bw * w
                box_h = bh * h

                x1 = int(max(0, round(x_center - box_w / 2)))
                y1 = int(max(0, round(y_center - box_h / 2)))
                x2 = int(min(w, round(x_center + box_w / 2)))
                y2 = int(min(h, round(y_center + box_h / 2)))

                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 255

            # YOLO segmentation polygon format: class x1 y1 x2 y2 ...
            elif len(values) >= 7:
                coords = values[1:]
                if len(coords) % 2 != 0:
                    continue

                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i + 1] * h)
                    points.append([x, y])

                if len(points) >= 3:
                    points = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [points], 255)

    return mask


def make_overlay(image_rgb: np.ndarray, pred_mask: np.ndarray, alpha: float, color: str) -> np.ndarray:
    """
    Blend a translucent colored mask over the original image.

    Args:
        image_rgb:  H x W x 3 uint8 image.
        pred_mask:  H x W uint8 mask with values in {0, 255}.
        alpha:      Opacity of the mask layer (0.0 – 1.0).
        color:      One of 'red', 'green', 'blue', 'yellow'.

    Returns:
        H x W x 3 uint8 overlay image.
    """
    color_map = {
        "red":    (255,   0,   0),
        "green":  (  0, 255,   0),
        "blue":   (  0,   0, 255),
        "yellow": (255, 255,   0),
    }
    rgb = color_map[color]

    # Resize mask to match image if needed (e.g. CLIPSeg outputs smaller logits)
    if pred_mask.shape != image_rgb.shape[:2]:
        pred_mask = cv2.resize(pred_mask, (image_rgb.shape[1], image_rgb.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    # Boolean mask of predicted foreground pixels
    binary = pred_mask > 127

    overlay = image_rgb.copy().astype(np.float32)
    for c, val in enumerate(rgb):
        overlay[binary, c] = (1 - alpha) * overlay[binary, c] + alpha * val

    return np.clip(overlay, 0, 255).astype(np.uint8)


def list_images(images_dir: str):
    paths = sorted(
        p for p in Path(images_dir).glob("*")
        if p.suffix.lower() in VALID_EXTS
    )
    if not paths:
        raise ValueError(f"No images found in {images_dir}")
    return paths


def save_quad_figure(image_rgb, gt_mask, pred_mask, overlay, save_path: Path, title: str):
    """Save a 4-panel figure: Image | GT | Prediction | Overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = list_images(args.images_dir)
    prompt_slug = sanitize_prompt(args.prompt)

    saved = 0
    for image_path in image_paths:
        if saved >= args.max_images:
            break

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        label_path = Path(args.labels_dir) / f"{image_path.stem}.txt"
        gt_mask = label_to_mask(label_path, image_rgb.shape)

        pred_path = Path(args.pred_dir) / f"{image_path.stem}__{prompt_slug}.png"
        if not pred_path.exists():
            print(f"Skipping missing prediction: {pred_path}")
            continue

        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            print(f"Skipping unreadable prediction: {pred_path}")
            continue

        overlay = make_overlay(image_rgb, pred_mask, alpha=args.overlay_alpha, color=args.overlay_color)

        save_path = Path(args.output_dir) / f"{image_path.stem}__viz.png"
        title = f"{image_path.stem} | {args.prompt}"
        save_quad_figure(image_rgb, gt_mask, pred_mask, overlay, save_path, title)

        print(f"Saved figure: {save_path}")
        saved += 1


if __name__ == "__main__":
    main()