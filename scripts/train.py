"""
Script de entrenamiento con argumentos.
Ejemplo:
  python scripts/train.py --data_dir data/processed --epochs 20 --batch 32 --lr 3e-5
"""
import argparse
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from utils.transforms import get_transforms
from models_src.vit_model import get_vit_model
from utils.train_utils import train_one_epoch, validate, save_checkpoint
from utils.early_stopping import EarlyStopping

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed", type=str)
    p.add_argument("--epochs", default=20, type=int)
    p.add_argument("--batch", default=32, type=int)
    p.add_argument("--lr", default=3e-5, type=float)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--model_name", default="vit_base_patch16_224", type=str)
    p.add_argument("--out", default="models/best_vit.pth", type=str)
    p.add_argument("--num_workers", default=4, type=int)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_t, val_t = get_transforms(224)

    train_dataset = datasets.ImageFolder(Path(args.data_dir) / "train", transform=train_t)
    val_dataset   = datasets.ImageFolder(Path(args.data_dir) / "val", transform=val_t)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    model = get_vit_model(num_classes=len(train_dataset.classes), model_name=args.model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    early_stopper = EarlyStopping(patience=4, min_delta=0.001)

    best_val_acc = 0.0
    out_path = Path(args.out)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} | Train loss: {train_loss:.4f} acc: {train_acc:.4f} | Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc
            }
            save_checkpoint(checkpoint, str(out_path))
            print("Best model saved ->", out_path)

        if early_stopper.step(val_loss):
            print("Early stopping triggered.")
            break

    print("Training finished.")

if __name__ == "__main__":
    main()
