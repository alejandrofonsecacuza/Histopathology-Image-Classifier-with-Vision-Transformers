"""
Carga el mejor modelo y calcula matriz de confusión y reporte.
Ejemplo:
  python scripts/evaluate.py --data_dir data/processed --model models/best_vit.pth --out_dir docs/
"""
import argparse
from pathlib import Path
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.transforms import get_transforms
from models_src.vit_model import get_vit_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import json

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed", type=str)
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--out_dir", default="docs", type=str)
    return p.parse_args()

def plot_cm(cm, classes, out_path):
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_t = get_transforms(224)
    test_dataset = datasets.ImageFolder(Path(args.data_dir) / "test", transform=val_t)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = get_vit_model(num_classes=len(test_dataset.classes))
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt) )
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=test_dataset.classes, output_dict=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plot_cm(cm, test_dataset.classes, out_dir / "confusion_matrix.png")
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Evaluation saved to", out_dir)

if __name__ == "__main__":
    main()
