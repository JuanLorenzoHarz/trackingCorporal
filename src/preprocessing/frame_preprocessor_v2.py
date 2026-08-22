"""Letterbox full-frame para a inferência multi-pessoa da PoseNet V2."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from src.core.types import Keypoint, Pose


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    original_width: int
    original_height: int
    input_size: int
    scale: float
    pad_x: int
    pad_y: int

    def pose_to_original(self, pose: Pose) -> Pose:
        points: list[Keypoint] = []
        for point in pose.keypoints:
            input_x = point.x * max(self.input_size - 1, 1)
            input_y = point.y * max(self.input_size - 1, 1)
            original_x = (input_x - self.pad_x) / max(self.scale, 1e-8)
            original_y = (input_y - self.pad_y) / max(self.scale, 1e-8)
            normalized_x = original_x / max(self.original_width - 1, 1)
            normalized_y = original_y / max(self.original_height - 1, 1)
            points.append(
                Keypoint(
                    x=min(1.0, max(0.0, normalized_x)),
                    y=min(1.0, max(0.0, normalized_y)),
                    confidence=point.confidence,
                )
            )
        return Pose(points)


def preprocess_frame_v2(
    frame: np.ndarray,
    input_size: int = 256,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, LetterboxTransform]:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame deve ser uma imagem BGR com 3 canais.")
    if input_size <= 0:
        raise ValueError("input_size deve ser positivo.")

    height, width = frame.shape[:2]
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
    )
    if device is not None:
        tensor = tensor.to(device)

    transform = LetterboxTransform(
        original_width=width,
        original_height=height,
        input_size=input_size,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return tensor, transform
