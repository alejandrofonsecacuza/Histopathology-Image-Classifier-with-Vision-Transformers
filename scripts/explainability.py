"""
Aplicar Grad-CAM sobre el modelo guardado y salvar mapas de atención.
Ejemplo:
  python scripts/explainability.py --model models/best_vit.pth --data_dir data/processed --out docs/attention
"""
import argparse
from pathlib import Path
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.transforms import get_transforms
from models_src.vit_model import get_vit_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--out", default="docs/attention")
    p.add_argument("--num_images", default=8, type=int)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_t, val_t = get_transforms(224)
    test_dataset = datasets.ImageFolder(Path(args.data_dir) / "test", transform=val_t)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = get_vit_model(num_classes=len(test_dataset.classes))
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.to(device)
    model.eval()

    # seleccionar capa target (ViT típico)
    target_layer = model.patch_embed.proj
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type=="cuda"))

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = outputs.max(1)

    count = min(args.num_images, images.size(0))
    for i in range(count):
        img_tensor = images[i].cpu()
        target_class = int(preds[i].cpu().numpy())
        grayscale_cam = cam(input_tensor=images[i].unsqueeze(0), targets=[ClassifierOutputTarget(target_class)])[0]

        # denormalizar imagen
        img_numpy = img_tensor.permute(1,2,0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img_numpy = np.clip(std * img_numpy + mean, 0, 1)

        visualization = show_cam_on_image(img_numpy, grayscale_cam, use_rgb=True)
        plt.imsave(out_dir / f"gradcam_{i}_gt{int(labels[i])}_pred{target_class}.png", visualization)

    print("Grad-CAMs guardados en:", out_dir)

if __name__ == "__main__":
    main()
