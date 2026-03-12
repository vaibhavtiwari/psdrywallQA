import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SegDataset(Dataset):
    def __init__(self, data_path, prompts_path, fixed_prompt=None):
        self.data_path = Path(data_path)
        self.fixed_prompt = fixed_prompt

        self.images = sorted(
            p for p in (self.data_path / "images").glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not self.images:
            raise ValueError(f"No images found in {self.data_path / 'images'}")

        with open(prompts_path) as f:
            data = json.load(f)

        data_path_lower = str(self.data_path).lower()
        if "cracks" in data_path_lower:
            self.prompts = data["cracks"]
        elif "drywall" in data_path_lower:
            self.prompts = data["drywall"]
        else:
            raise ValueError(f"Could not infer prompt type from path: {self.data_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = self.data_path / "labels" / f"{image_path.stem}.txt"
        mask = self._label_to_mask(label_path, image.shape)

        mask = torch.from_numpy(mask).unsqueeze(0).float()
        prompt = self.fixed_prompt if self.fixed_prompt is not None else random.choice(self.prompts)

        return {
            "image": image,
            "prompt": prompt,
            "mask": mask,
            "image_id": image_path.stem,
        }

    def _label_to_mask(self, label_path, image_shape):
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if not label_path.exists():
            return mask

        with open(label_path) as f:
            for line in f:
                values = list(map(float, line.split()))
                if not values:
                    continue

                # YOLO detection bbox: class cx cy w h
                if len(values) == 5:
                    _, cx, cy, bw, bh = values

                    x_center = cx * width
                    y_center = cy * height
                    box_w = bw * width
                    box_h = bh * height

                    x1 = int(max(0, round(x_center - box_w / 2)))
                    y1 = int(max(0, round(y_center - box_h / 2)))
                    x2 = int(min(width, round(x_center + box_w / 2)))
                    y2 = int(min(height, round(y_center + box_h / 2)))

                    if x2 > x1 and y2 > y1:
                        mask[y1:y2, x1:x2] = 1

                # YOLO segmentation polygon: class x1 y1 x2 y2 ...
                elif len(values) >= 7:
                    coords = values[1:]
                    if len(coords) % 2 != 0:
                        continue

                    points = []
                    for i in range(0, len(coords), 2):
                        x = int(coords[i] * width)
                        y = int(coords[i + 1] * height)
                        points.append([x, y])

                    if len(points) >= 3:
                        points = np.array(points, dtype=np.int32)
                        cv2.fillPoly(mask, [points], 1)

        return mask