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

from src.core.types import Keypoint, Pose
from src.visualization.renderer import draw_pose


OUTPUT_PATH = Path("data/processed/skeleton_demo.png")


def build_demo_pose() -> Pose:
    """Cria uma pose artificial simples com os 17 keypoints do projeto."""
    coordinates = (
        (0.50, 0.16),  # NOSE
        (0.47, 0.14),  # LEFT_EYE
        (0.53, 0.14),  # RIGHT_EYE
        (0.43, 0.16),  # LEFT_EAR
        (0.57, 0.16),  # RIGHT_EAR
        (0.39, 0.30),  # LEFT_SHOULDER
        (0.61, 0.30),  # RIGHT_SHOULDER
        (0.32, 0.45),  # LEFT_ELBOW
        (0.68, 0.45),  # RIGHT_ELBOW
        (0.27, 0.60),  # LEFT_WRIST
        (0.73, 0.60),  # RIGHT_WRIST
        (0.43, 0.57),  # LEFT_HIP
        (0.57, 0.57),  # RIGHT_HIP
        (0.41, 0.75),  # LEFT_KNEE
        (0.59, 0.75),  # RIGHT_KNEE
        (0.39, 0.94),  # LEFT_ANKLE
        (0.61, 0.94),  # RIGHT_ANKLE
    )

    return Pose(
        [
            Keypoint(x=x, y=y, confidence=1.0)
            for x, y in coordinates
        ]
    )


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
