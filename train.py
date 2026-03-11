import os
import argparse
import time
from distutils.util import strtobool

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, DataLoader

from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from dataset import SegDataset


ROOT = os.getcwd()
DATA_DIR = "data"
DATASET_CRACKS_DIR = "cracks.v1i.yolov8"
DATASET_DRYWALL_DIR = "Drywall-Join-Detect.v1i.yolov8"
PROMPT_DIR = "prompts/augmented_prompts.json"


def collate_fn(batch):
    images = [b["image"] for b in batch]
    masks = torch.stack([b["mask"] for b in batch])
    prompts = [b["prompt"] for b in batch]

    return {
        "image": images,
        "mask": masks,
        "prompt": prompts
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--arch", type=str, default="CLIPSeg")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--wandb-project-name", type=str, default="PromptSeg")
    parser.add_argument("--wandb-entity", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="checkpoints")

    return parser.parse_args()


def compute_iou_and_dice(logits, masks, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    dice_denom = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (dice_denom + eps)

    return iou.mean().item(), dice.mean().item()


def prepare_batch(batch, processor, device):
    inputs = processor(
        text=batch["prompt"],
        images=batch["image"],
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    masks = batch["mask"].to(device)
    return inputs, masks


def forward_step(model, processor, batch, device, criterion):
    inputs, masks = prepare_batch(batch, processor, device)

    outputs = model(**inputs)
    logits = outputs.logits

    if logits.dim() == 3:
        logits = logits.unsqueeze(1)

    target = F.interpolate(
        masks,
        size=logits.shape[-2:],
        mode="nearest"
    )

    loss = criterion(logits, target)
    iou, dice = compute_iou_and_dice(logits, target)

    return loss, iou, dice, logits, target


@torch.no_grad()
def evaluate(model, processor, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_batches = 0

    sample_batch = None
    sample_logits = None
    sample_target = None

    for batch in loader:
        loss, iou, dice, logits, target = forward_step(model, processor, batch, device, criterion)

        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        total_batches += 1

        if sample_batch is None:
            sample_batch = batch
            sample_logits = logits.detach().cpu()
            sample_target = target.detach().cpu()

    return {
        "loss": total_loss / max(total_batches, 1),
        "iou": total_iou / max(total_batches, 1),
        "dice": total_dice / max(total_batches, 1),
        "sample_batch": sample_batch,
        "sample_logits": sample_logits,
        "sample_target": sample_target,
    }


if __name__ == "__main__":
    args = parse_args()

    run_name = f"{args.exp_name}__{args.arch}__{int(time.time())}"
    os.makedirs(args.save_dir, exist_ok=True)

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Using device:", device)

    prompts_path = os.path.join(ROOT, DATA_DIR, PROMPT_DIR)

    cracks_train_dataset = SegDataset(
        os.path.join(ROOT, DATA_DIR, DATASET_CRACKS_DIR, "train"),
        prompts_path=prompts_path
    )
    drywall_train_dataset = SegDataset(
        os.path.join(ROOT, DATA_DIR, DATASET_DRYWALL_DIR, "train"),
        prompts_path=prompts_path
    )

    cracks_test_dataset = SegDataset(
        os.path.join(ROOT, DATA_DIR, DATASET_CRACKS_DIR, "test"),
        prompts_path=prompts_path
    )
    drywall_test_dataset = SegDataset(
        os.path.join(ROOT, DATA_DIR, DATASET_DRYWALL_DIR, "test"),
        prompts_path=prompts_path
    )

    train_dataset = ConcatDataset([cracks_train_dataset, drywall_train_dataset])
    test_dataset = ConcatDataset([cracks_test_dataset, drywall_test_dataset])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    global_step = 0
    best_test_loss = float("inf")

    for epoch in range(args.num_epochs):
        model.train()

        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            loss, iou, dice, logits, target = forward_step(model, processor, batch, device, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_iou += iou
            running_dice += dice
            num_batches += 1

            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar("train/batch_iou", iou, global_step)
            writer.add_scalar("train/batch_dice", dice, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            if batch_idx % args.log_every == 0:
                print(
                    f"Epoch [{epoch+1}/{args.num_epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"IoU: {iou:.4f} "
                    f"Dice: {dice:.4f}"
                )

            global_step += 1

        train_loss = running_loss / max(num_batches, 1)
        train_iou = running_iou / max(num_batches, 1)
        train_dice = running_dice / max(num_batches, 1)

        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("train/epoch_iou", train_iou, epoch)
        writer.add_scalar("train/epoch_dice", train_dice, epoch)

        eval_stats = evaluate(model, processor, test_loader, device, criterion)

        writer.add_scalar("test/loss", eval_stats["loss"], epoch)
        writer.add_scalar("test/iou", eval_stats["iou"], epoch)
        writer.add_scalar("test/dice", eval_stats["dice"], epoch)

        print(
            f"[Epoch {epoch+1}] "
            f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train Dice: {train_dice:.4f} | "
            f"Test Loss: {eval_stats['loss']:.4f}, Test IoU: {eval_stats['iou']:.4f}, Test Dice: {eval_stats['dice']:.4f}"
        )

        sample_batch = eval_stats["sample_batch"]
        sample_logits = eval_stats["sample_logits"]
        sample_target = eval_stats["sample_target"]

        if sample_batch is not None:
            sample_image = torch.tensor(sample_batch["image"][0]).permute(2, 0, 1).float() / 255.0
            sample_pred = (torch.sigmoid(sample_logits[0]) > 0.5).float()
            sample_gt = sample_target[0]

            writer.add_image("samples/image", sample_image, epoch)
            writer.add_image("samples/pred_mask", sample_pred, epoch)
            writer.add_image("samples/gt_mask", sample_gt, epoch)
            writer.add_text("samples/prompt", sample_batch["prompt"][0], epoch)

        if eval_stats["loss"] < best_test_loss:
            best_test_loss = eval_stats["loss"]
            save_path = os.path.join(args.save_dir, f"{run_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    writer.close()