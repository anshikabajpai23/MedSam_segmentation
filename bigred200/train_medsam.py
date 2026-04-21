from pathlib import Path
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
from tqdm import tqdm


class MedSAMDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row["npz_path"], allow_pickle=True)

        image = data["image"].astype(np.float32)   # (3, H, W)
        mask = data["mask"].astype(np.float32)     # (H, W)
        box = data["box"].astype(np.float32)       # (4,)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask[None, ...], dtype=torch.float32)  # (1, H, W)
        box = torch.tensor(box, dtype=torch.float32)

        return {
            "image": image,
            "mask": mask,
            "box": box,
            "case_id": str(row["case_id"]),
            "slice_idx": int(row["slice_idx"]),
        }


def dice_loss(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    dice = (2.0 * inter) / denom
    return 1.0 - dice.mean()


@torch.no_grad()
def validate_one_epoch(model, loader, bce_loss, device):
    model.eval()
    total_loss = 0.0
    dice_scores = []

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        boxes = batch["box"].to(device)

        image_embeddings = model.image_encoder(model.preprocess(images))

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=boxes[:, None, :],   # [B,1,4]
            masks=None,
        )

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        pred_masks = F.interpolate(
            low_res_masks,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = bce_loss(pred_masks, masks) + dice_loss(pred_masks, masks)
        total_loss += loss.item()

        pred_bin = (torch.sigmoid(pred_masks) > 0.5).float()
        inter = (pred_bin * masks).sum(dim=(1, 2, 3))
        denom = pred_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6
        dice = (2.0 * inter / denom).detach().cpu().numpy()
        dice_scores.extend(dice.tolist())

    mean_loss = total_loss / max(len(loader), 1)
    mean_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    return mean_loss, mean_dice


def train_one_epoch(model, loader, optimizer, bce_loss, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        boxes = batch["box"].to(device)

        image_embeddings = model.image_encoder(model.preprocess(images))

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=boxes[:, None, :],   # [B,1,4]
                masks=None,
            )

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        pred_masks = F.interpolate(
            low_res_masks,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = bce_loss(pred_masks, masks) + dice_loss(pred_masks, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_per_slice(model, loader, device):
    model.eval()
    rows = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        boxes = batch["box"].to(device)

        image_embeddings = model.image_encoder(model.preprocess(images))

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=None,
            boxes=boxes[:, None, :],
            masks=None,
        )

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        pred_masks = F.interpolate(
            low_res_masks,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        pred_bin = (torch.sigmoid(pred_masks) > 0.5).float()
        inter = (pred_bin * masks).sum(dim=(1, 2, 3))
        denom = pred_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6
        dice = (2.0 * inter / denom).detach().cpu().numpy()

        batch_size = images.shape[0]
        for i in range(batch_size):
            rows.append({
                "case_id": batch["case_id"][i],
                "slice_idx": int(batch["slice_idx"][i]),
                "dice": float(dice[i]),
            })

    return pd.DataFrame(rows)


def load_model(checkpoint_path: str, device: str):
    checkpoint_path = str(checkpoint_path)

    if device == "cpu":
        sam = sam_model_registry["vit_b"](checkpoint=None)
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        sam.load_state_dict(state_dict)
    else:
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)

    sam = sam.to(device)
    return sam


def freeze_for_decoder_only(model):
    for p in model.image_encoder.parameters():
        p.requires_grad = False

    for p in model.prompt_encoder.parameters():
        p.requires_grad = False

    for p in model.mask_decoder.parameters():
        p.requires_grad = True


def freeze_for_last_block_plus_decoder(model):
    for p in model.image_encoder.parameters():
        p.requires_grad = False

    for p in model.prompt_encoder.parameters():
        p.requires_grad = False

    for p in model.mask_decoder.parameters():
        p.requires_grad = True

    for p in model.image_encoder.blocks[-1].parameters():
        p.requires_grad = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--outdir", type=str, default="./outputs_medsam")
    parser.add_argument(
        "--train_mode",
        type=str,
        default="decoder_only",
        choices=["decoder_only", "last_block_plus_decoder"],
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_ds = MedSAMDataset(args.train_csv)
    val_ds = MedSAMDataset(args.val_csv)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    print("train samples:", len(train_ds))
    print("val samples:", len(val_ds))

    sam = load_model(args.checkpoint, device)

    if args.train_mode == "decoder_only":
        freeze_for_decoder_only(sam)
    else:
        freeze_for_last_block_plus_decoder(sam)

    num_trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in sam.parameters())
    print("train mode:", args.train_mode)
    print("trainable params:", num_trainable)
    print("total params:", num_total)

    optimizer = torch.optim.AdamW(
        [p for p in sam.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_dice = -1.0
    history = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(sam, train_loader, optimizer, bce_loss, device)
        val_loss, val_dice = validate_one_epoch(sam, val_loader, bce_loss, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
        }
        history.append(row)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  train_loss = {train_loss:.4f}")
        print(f"  val_loss   = {val_loss:.4f}")
        print(f"  val_dice   = {val_dice:.4f}")

        history_path = Path(args.outdir) / "history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)

        latest_path = Path(args.outdir) / "latest_model.pth"
        torch.save(sam.state_dict(), latest_path)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = Path(args.outdir) / "best_model.pth"
            torch.save(sam.state_dict(), best_path)
            print(f"  saved best model to {best_path}")

    results_df = evaluate_per_slice(sam, val_loader, device)
    results_df.to_csv(Path(args.outdir) / "val_per_slice.csv", index=False)

    case_scores = results_df.groupby("case_id")["dice"].mean().reset_index()
    case_scores.to_csv(Path(args.outdir) / "val_per_case.csv", index=False)

    print("final mean slice dice:", results_df["dice"].mean())
    print("final mean case dice :", case_scores["dice"].mean())
    print("training complete")


if __name__ == "__main__":
    main()