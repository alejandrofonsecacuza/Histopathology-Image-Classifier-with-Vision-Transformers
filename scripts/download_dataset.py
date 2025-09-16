"""
Descarga el archivo desde Figshare u otra URL y lo guarda en data/raw/.
Uso:
    python scripts/download_dataset.py --url https://figshare.com/ndownloader/files/13496366
"""
import argparse
import os
from pathlib import Path
import requests
from tqdm import tqdm
import shutil
import zipfile

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download(url: str, out_path: Path):
    out_tmp = out_path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_tmp, "wb") as f, tqdm(total=total, unit='B', unit_scale=True, desc=out_path.name) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    out_tmp.replace(out_path)
    return out_path

def maybe_unzip(path: Path, extract_to: Path):
    try:
        if zipfile.is_zipfile(path):
            print("Descomprimiendo:", path)
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(extract_to)
            print("Descompresión finalizada.")
        else:
            print("No es un ZIP o no es necesario descomprimir.")
    except Exception as e:
        print("Error al descomprimir:", e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="URL del archivo a descargar")
    parser.add_argument("--out", default=str(RAW_DIR / "dataset.zip"))
    parser.add_argument("--extract", action="store_true", help="Extraer si es zip")
    args = parser.parse_args()

    out_path = Path(args.out)
    print("Descargando a:", out_path)
    download(args.url, out_path)
    if args.extract:
        maybe_unzip(out_path, RAW_DIR)

if __name__ == "__main__":
    main()
