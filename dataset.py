import os
import glob
from pathlib import Path
import cv2
import json

from torch.utils.data import Dataset
import numpy as np
import torch

import random

class SegDataset(Dataset):

    def __init__(self, data_path, prompts_path):

        self.images = list(Path(data_path).glob("images/*.jpg"))

        with open(prompts_path) as f:
            data = json.load(f)

        if "cracks" in data_path:
            self.prompts = data["cracks"]
        elif "drywall" in data_path:
            self.prompts = data["drywall"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = str(image_path).replace("/images/", "/labels/").replace(".jpg", ".txt")

        mask = self._polygon_to_mask(label_path, image.shape)

        mask = torch.from_numpy(mask).unsqueeze(0).float()

        prompt = random.choice(self.prompts)

        return {
            "image": image,
            "prompt": prompt,
            "mask": mask
        }

    def _polygon_to_mask(self, label_path, image_shape):

        height, width = image_shape[:2]

        mask = np.zeros((height, width), dtype=np.uint8)

        with open(label_path) as f:
            for line in f:

                values = list(map(float, line.split()))
                coords = values[1:]

                points = []

                for i in range(0, len(coords), 2):
                    x = int(coords[i] * width)
                    y = int(coords[i + 1] * height)
                    points.append([x, y])

                points = np.array(points, dtype=np.int32)

                cv2.fillPoly(mask, [points], 1)

        return mask


"""
path = str("/home/vaibhav/Playground/git/psdrywallQA/data/cracks.v1i.yolov8/train/images/3_jpg.rf.20d82b882b32fcd953b4a0bc53d93c19.jpg")    
image = cv2.imread(path)
label_path = str(path).replace("images", "labels").replace(".jpg", ".txt")
mask = polygon_to_mask(label_path, image.shape)

cv2.imshow("Mask", mask*255)
cv2.waitKey(0)

print(type(image))
"""