"""Renderização simples de poses e esqueletos com OpenCV."""

import cv2
import numpy as np

from src.core.types import Pose
from src.pose.keypoints import SKELETON_CONNECTIONS


def normalized_to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Converte coordenadas normalizadas para coordenadas em pixels."""
    px = int(round(x * (width - 1)))
    py = int(round(y * (height - 1)))
    return px, py


def draw_pose(
    frame: np.ndarray,
    pose: Pose,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Desenha keypoints válidos e conexões do esqueleto sobre um frame."""
    height, width = frame.shape[:2]

    # Desenha primeiro as conexões, para os pontos ficarem por cima.
    for start_index, end_index in SKELETON_CONNECTIONS:
        if start_index >= len(pose) or end_index >= len(pose):
            continue

        start = pose[start_index]
        end = pose[end_index]

        if not start.is_valid(confidence_threshold):
            continue
        if not end.is_valid(confidence_threshold):
            continue

        start_pixel = normalized_to_pixel(start.x, start.y, width, height)
        end_pixel = normalized_to_pixel(end.x, end.y, width, height)

        cv2.line(frame, start_pixel, end_pixel, (60, 180, 75), 3)

    for point in pose.keypoints:
        if not point.is_valid(confidence_threshold):
            continue

        pixel = normalized_to_pixel(point.x, point.y, width, height)
        cv2.circle(frame, pixel, 6, (40, 40, 230), -1)

    return frame
