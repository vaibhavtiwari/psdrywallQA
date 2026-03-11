import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.load_state_dict(
    torch.load("checkpoints/train__CLIPSeg__1773160622_best.pt", map_location=device)
)
model = model.to(device)
model.eval()

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

image_path = "/home/vaibhav/Playground/git/psdrywallQA/data/cracks.v1i.yolov8/valid/images/images27_jpg.rf.2dbc5d4631b5e5f07dfe0dddeecde1b9.jpg"
prompt = "ssegment drywall crack"

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]

inputs = processor(
    text=[prompt],
    images=[image],
    return_tensors="pt",
    padding=True
)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

    if logits.dim() == 3:
        logits = logits.unsqueeze(1)

    probs = torch.sigmoid(logits)
    pred_mask = (probs > 0.5).float()

    pred_mask = F.interpolate(
        pred_mask,
        size=(h, w),
        mode="nearest"
    )

mask_np = pred_mask[0, 0].cpu().numpy().astype("uint8") * 255

out_path = "2056__segment_crack.png"
cv2.imwrite(out_path, mask_np)

print("Saved mask to:", out_path)
print("Mask shape:", mask_np.shape)
print("Unique values:", set(mask_np.flatten().tolist()))