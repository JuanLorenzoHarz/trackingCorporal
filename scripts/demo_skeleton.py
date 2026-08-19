"""Gera uma imagem de demonstração do esqueleto sem usar webcam ou IA.

Execute a partir da raiz do projeto:

    python -m scripts.demo_skeleton

Use --show para também abrir uma janela com o resultado:

    python -m scripts.demo_skeleton --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.pose.demo_pose import build_demo_pose
from src.visualization.renderer import draw_pose


OUTPUT_PATH = Path("data/processed/skeleton_demo.png")


def create_demo_image(width: int = 960, height: int = 720) -> np.ndarray:
    """Cria uma imagem clara e desenha nela uma pose artificial."""
    frame = np.full((height, width, 3), 245, dtype=np.uint8)
    pose = build_demo_pose()
    return draw_pose(frame, pose)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstração do esqueleto sem webcam.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="abre uma janela com o resultado além de salvar a imagem",
    )
    args = parser.parse_args()

    image = create_demo_image()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), image)
    print(f"Demo salvo em: {OUTPUT_PATH.resolve()}")

    if args.show:
        cv2.imshow("trackingCorporal - skeleton demo", image)
        print("Pressione qualquer tecla na janela para fechar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
