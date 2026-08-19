"""Baixa e extrai os arquivos COCO 2017 necessários para pose.

Uso recomendado para validar o pipeline com um conjunto menor:
    python -m scripts.download_coco --split val

Para baixar o conjunto de treino completo (grande):
    python -m scripts.download_coco --split train
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


URLS = {
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val": "http://images.cocodataset.org/zips/val2017.zip",
}


def download(url: str, destination: Path) -> None:
    """Baixa um arquivo mostrando progresso simples no terminal."""
    if destination.exists():
        print(f"Já existe, pulando download: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    print(f"Baixando {url}")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break

            output.write(chunk)
            downloaded += len(chunk)

            if total > 0:
                percent = downloaded * 100.0 / total
                print(
                    f"\r{downloaded / 1024**2:8.1f} MB / "
                    f"{total / 1024**2:8.1f} MB ({percent:5.1f}%)",
                    end="",
                )
            else:
                print(f"\r{downloaded / 1024**2:8.1f} MB", end="")

    print()
    temporary.replace(destination)


def extract(archive: Path, destination: Path) -> None:
    """Extrai um ZIP do COCO."""
    print(f"Extraindo {archive.name}...")
    with zipfile.ZipFile(archive, "r") as zip_file:
        zip_file.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa COCO 2017 para pose.")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="val é menor para validar o pipeline; train é o conjunto de treino real.",
    )
    parser.add_argument("--root", default="data/coco")
    parser.add_argument("--keep-zips", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    downloads = root / "downloads"
    root.mkdir(parents=True, exist_ok=True)

    archives = [
        ("annotations", downloads / "annotations_trainval2017.zip"),
        (args.split, downloads / f"{args.split}2017.zip"),
    ]

    for key, archive in archives:
        download(URLS[key], archive)
        extract(archive, root)

        if not args.keep_zips:
            archive.unlink(missing_ok=True)

    if downloads.exists() and not any(downloads.iterdir()):
        shutil.rmtree(downloads)

    print("COCO preparado em:", root.resolve())
    print("Imagens:", (root / f"{args.split}2017").resolve())
    print("Anotações:", (root / "annotations").resolve())


if __name__ == "__main__":
    main()
