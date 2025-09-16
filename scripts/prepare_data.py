"""
Flatten + limpieza de carpetas y split train/val/test.
Ejemplo:
    python scripts/prepare_data.py --raw_dir data/raw/ --out_dir data/processed/ --train 0.7 --val 0.15 --test 0.15
"""
import argparse
from pathlib import Path
import shutil
import os
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

def is_image(fname):
    return fname.lower().endswith(IMAGE_EXTENSIONS)

def move_images_to_parent(src_dir: Path, dest_dir: Path):
    """Mueve imágenes de subdirectorios hacia dest_dir y renombra duplicados."""
    if not src_dir.exists():
        return
    for root, _, files in os.walk(src_dir):
        for f in files:
            if is_image(f):
                src = Path(root) / f
                dest = dest_dir / f
                counter = 1
                while dest.exists():
                    name, ext = dest.stem, dest.suffix
                    dest = dest_dir / f"{name}_{counter}{ext}"
                    counter += 1
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))

    # intentar eliminar subcarpetas vacías
    for p, dirs, _ in os.walk(src_dir, topdown=False):
        for d in dirs:
            try:
                Path(p).joinpath(d).rmdir()
            except OSError:
                pass

def flatten_special_folders(raw_root: Path):
    """
    Casos del dataset: NE/Menstrual, NE/Luteal, NE/Follicular -> NE
    EH/Simple, EH/Complex -> EH
    """
    ne_subs = ["Menstrual", "Luteal", "Follicular"]
    eh_subs = ["Simple", "Complex"]
    for sub in ne_subs:
        s = raw_root / "NE" / sub
        if s.exists():
            move_images_to_parent(s, raw_root / "NE")
    for sub in eh_subs:
        s = raw_root / "EH" / sub
        if s.exists():
            move_images_to_parent(s, raw_root / "EH")

def split_and_copy(raw_root: Path, out_root: Path, train_ratio, val_ratio, test_ratio, seed=42):
    out_train = out_root / "train"
    out_val = out_root / "val"
    out_test = out_root / "test"
    for out in [out_train, out_val, out_test]:
        out.mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in raw_root.iterdir() if d.is_dir()]
    for cls in classes:
        cls_dir = raw_root / cls
        imgs = [f for f in cls_dir.iterdir() if f.is_file() and is_image(f.name)]
        imgs = sorted(imgs)
        if not imgs:
            continue

        # split per-class to preserve balance
        train_val, test = train_test_split(imgs, test_size=test_ratio, random_state=seed)
        # ahora entre train y val
        val_rel = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(train_val, test_size=val_rel, random_state=seed)

        for p in train:
            dest = out_train / cls
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / p.name)
        for p in val:
            dest = out_val / cls
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / p.name)
        for p in test:
            dest = out_test / cls
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest / p.name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw/", type=str)
    parser.add_argument("--out_dir", default="data/processed/", type=str)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    raw_root = Path(args.raw_dir)
    out_root = Path(args.out_dir)
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "Train+Val+Test must sum 1.0"

    flatten_special_folders(raw_root)
    split_and_copy(raw_root, out_root, args.train, args.val, args.test)
    print("Preparación finalizada. Salida en:", out_root)

if __name__ == "__main__":
    main()
