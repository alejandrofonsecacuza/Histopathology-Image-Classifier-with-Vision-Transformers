from pathlib import Path
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

def count_images_in_dir(directory: Path):
    total = 0
    for p in directory.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            total += 1
    return total

def list_classes(directory: Path):
    return sorted([d.name for d in directory.iterdir() if d.is_dir()])
