import os
import argparse
import time
import random
from distutils.util import strtobool

import numpy as np
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    images = [b["image"] for b in batch]
    masks = torch.stack([b["mask"] for b in batch])
    prompts = [b["prompt"] for b in batch]
    image_ids = [b["image_id"] for b in batch]

    return {
        "image": images,
        "mask": masks,
        "prompt": prompts,
        "image_id": image_ids,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--arch", type=str, default="CLIPSeg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--wandb-project-name", type=str, default="PromptSeg")
    parser.add_argument("--wandb-entity", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=32)
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
    masks = batch["mask"].to(device).float()
    return inputs, masks


def forward_step(model, processor, batch, device, criterion):
    inputs, masks = prepare_batch(batch, processor, device)

    outputs = model(**inputs)
    logits = outputs.logits

    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    elif logits.dim() != 4:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

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
    set_seed(args.seed)

    run_name = f"{args.exp_name}__{args.arch}__seed{args.seed}__{int(time.time())}"
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

    # Train datasets: prompt augmentation allowed
    cracks_train_dataset = SegDataset(
        os.path.join(ROOT, DATA_DIR, DATASET_CRACKS_DIR, "train"),
        prompts_path=prompts_path
    )
    drywall_train_dataset = SegDataset(
        os.path.join(ROOT, DATA_DIR, DATASET_DRYWALL_DIR, "train"),
        prompts_path=prompts_path
    )

    # Validation datasets: deterministic canonical prompts
    cracks_valid_dataset = SegDataset(
        os.path.join(ROOT, DATA_DIR, DATASET_CRACKS_DIR, "valid"),
        prompts_path=prompts_path,
        fixed_prompt="segment crack"
    )
    drywall_valid_dataset = SegDataset(
        os.path.join(ROOT, DATA_DIR, DATASET_DRYWALL_DIR, "valid"),
        prompts_path=prompts_path,
        fixed_prompt="segment taping area"
    )

    train_dataset = ConcatDataset([cracks_train_dataset, drywall_train_dataset])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available() and args.cuda,
    )

    cracks_valid_loader = DataLoader(
        cracks_valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available() and args.cuda,
    )

    drywall_valid_loader = DataLoader(
        drywall_valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available() and args.cuda,
    )

    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    global_step = 0
    best_macro_dice = -1.0

    train_start_time = time.time()

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

        cracks_stats = evaluate(model, processor, cracks_valid_loader, device, criterion)
        drywall_stats = evaluate(model, processor, drywall_valid_loader, device, criterion)

        macro_iou = 0.5 * (cracks_stats["iou"] + drywall_stats["iou"])
        macro_dice = 0.5 * (cracks_stats["dice"] + drywall_stats["dice"])
        macro_loss = 0.5 * (cracks_stats["loss"] + drywall_stats["loss"])

        writer.add_scalar("val/cracks_loss", cracks_stats["loss"], epoch)
        writer.add_scalar("val/cracks_iou", cracks_stats["iou"], epoch)
        writer.add_scalar("val/cracks_dice", cracks_stats["dice"], epoch)

        writer.add_scalar("val/drywall_loss", drywall_stats["loss"], epoch)
        writer.add_scalar("val/drywall_iou", drywall_stats["iou"], epoch)
        writer.add_scalar("val/drywall_dice", drywall_stats["dice"], epoch)

        writer.add_scalar("val/macro_loss", macro_loss, epoch)
        writer.add_scalar("val/macro_iou", macro_iou, epoch)
        writer.add_scalar("val/macro_dice", macro_dice, epoch)

        print(
            f"[Epoch {epoch+1}] "
            f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train Dice: {train_dice:.4f} | "
            f"Cracks Val Loss: {cracks_stats['loss']:.4f}, IoU: {cracks_stats['iou']:.4f}, Dice: {cracks_stats['dice']:.4f} | "
            f"Drywall Val Loss: {drywall_stats['loss']:.4f}, IoU: {drywall_stats['iou']:.4f}, Dice: {drywall_stats['dice']:.4f} | "
            f"Macro IoU: {macro_iou:.4f}, Macro Dice: {macro_dice:.4f}"
        )

        # Log one example per task
        for tag, stats in [("cracks", cracks_stats), ("drywall", drywall_stats)]:
            sample_batch = stats["sample_batch"]
            sample_logits = stats["sample_logits"]
            sample_target = stats["sample_target"]

            if sample_batch is not None:
                sample_image = torch.from_numpy(sample_batch["image"][0]).permute(2, 0, 1).float() / 255.0
                sample_pred = (torch.sigmoid(sample_logits[0]) > 0.5).float()
                sample_gt = sample_target[0].float()

                writer.add_image(f"samples/{tag}_image", sample_image, epoch)
                writer.add_image(f"samples/{tag}_pred_mask", sample_pred, epoch)
                writer.add_image(f"samples/{tag}_gt_mask", sample_gt, epoch)
                writer.add_text(f"samples/{tag}_prompt", sample_batch["prompt"][0], epoch)

        if macro_dice > best_macro_dice:
            best_macro_dice = macro_dice
            save_path = os.path.join(args.save_dir, f"{run_name}_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_dice": best_macro_dice,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"Saved best model to {save_path}")

    total_train_time = time.time() - train_start_time
    print(f"Total training time: {total_train_time:.2f} seconds")

    writer.add_text("runtime/train_time_seconds", f"{total_train_time:.2f}")
    writer.close()