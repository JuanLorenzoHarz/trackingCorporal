"""Pré-processamento compartilhado pela inferência de pose.

O treino usa recortes quadrados de pessoas. Na webcam, portanto, não devemos
esticar um frame 4:3 diretamente para 1:1: isso altera proporções corporais e
prejudica principalmente quadril, joelhos e tornozelos.

Este módulo usa um recorte quadrado central e mantém uma transformação capaz de
remapear os keypoints normalizados da CNN de volta para o frame original.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from src.core.types import Keypoint, Pose


@dataclass(frozen=True, slots=True)
class FrameTransform:
    """Transformação entre o recorte quadrado da CNN e o frame original."""

    original_width: int
    original_height: int
    crop_left: int
    crop_top: int
    crop_size: int

    def pose_to_original(self, pose: Pose) -> Pose:
        """Converte coordenadas 0..1 do recorte para coordenadas 0..1 do frame."""
        points: list[Keypoint] = []

        for point in pose.keypoints:
            pixel_x = self.crop_left + point.x * self.crop_size
            pixel_y = self.crop_top + point.y * self.crop_size

            normalized_x = pixel_x / max(self.original_width, 1)
            normalized_y = pixel_y / max(self.original_height, 1)

            points.append(
                Keypoint(
                    x=min(1.0, max(0.0, normalized_x)),
                    y=min(1.0, max(0.0, normalized_y)),
                    confidence=point.confidence,
                )
            )

        return Pose(points)


def _validate_frame(frame: np.ndarray) -> None:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame deve ser uma imagem BGR com 3 canais.")


def _to_tensor(
    image: np.ndarray,
    input_size: int,
    device: torch.device | str | None,
) -> torch.Tensor:
    resized = cv2.resize(
        image,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
    )

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def preprocess_frame_with_transform(
    frame: np.ndarray,
    input_size: int = 256,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, FrameTransform]:
    """Recorta o maior quadrado central sem distorção e retorna sua transformação."""
    _validate_frame(frame)

    height, width = frame.shape[:2]
    crop_size = min(width, height)
    crop_left = (width - crop_size) // 2
    crop_top = (height - crop_size) // 2

    crop = frame[
        crop_top : crop_top + crop_size,
        crop_left : crop_left + crop_size,
    ]

    transform = FrameTransform(
        original_width=width,
        original_height=height,
        crop_left=crop_left,
        crop_top=crop_top,
        crop_size=crop_size,
    )

    return _to_tensor(crop, input_size, device), transform


def preprocess_frame(
    frame: np.ndarray,
    input_size: int = 256,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compatibilidade: retorna apenas o tensor usando o recorte sem distorção."""
    tensor, _ = preprocess_frame_with_transform(
        frame,
        input_size=input_size,
        device=device,
    )
    return tensor
