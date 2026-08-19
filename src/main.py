"""Ponto de entrada atual do trackingCorporal.

Nesta etapa o programa mostra a webcam ao vivo e desenha uma pose ARTIFICIAL
fixa por cima da imagem. Isso valida captura + loop + renderer antes da CNN.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.capture.camera import Camera
from src.pose.demo_pose import build_demo_pose
from src.visualization.renderer import draw_pose


WINDOW_NAME = "trackingCorporal"
CAMERA_INDEX = 0
REQUESTED_WIDTH = 640
REQUESTED_HEIGHT = 480


def draw_status(frame: np.ndarray, fps: float) -> np.ndarray:
    """Exibe informações úteis da etapa atual sobre o frame."""
    cv2.putText(
        frame,
        "POSE DEMO - NOT TRACKING",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Q ou ESC para sair",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    """Executa o primeiro loop de vídeo em tempo real do projeto."""
    demo_pose = build_demo_pose()
    previous_time = time.perf_counter()

    try:
        with Camera(
            index=CAMERA_INDEX,
            width=REQUESTED_WIDTH,
            height=REQUESTED_HEIGHT,
        ) as camera:
            width, height = camera.resolution
            print(f"Câmera {CAMERA_INDEX} aberta em {width}x{height}.")
            print("A pose exibida é artificial e ainda NÃO acompanha o corpo.")
            print("Pressione Q ou ESC na janela para encerrar.")

            while True:
                frame = camera.read()

                # Etapa temporária: o frame é real, mas a pose ainda é fixa.
                draw_pose(frame, demo_pose)

                current_time = time.perf_counter()
                elapsed = current_time - previous_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                previous_time = current_time

                draw_status(frame, fps)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

    except RuntimeError as error:
        raise SystemExit(f"Erro de câmera: {error}") from error
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
