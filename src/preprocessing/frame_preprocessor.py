"""Pré-processamento compartilhado pela inferência de pose."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def preprocess_frame(
    frame: np.ndarray,
    input_size: int = 256,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Converte um frame BGR do OpenCV em tensor RGB [1,3,H,W] entre 0 e 1."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame deve ser uma imagem BGR com 3 canais.")

    resized = cv2.resize(
        frame,
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
