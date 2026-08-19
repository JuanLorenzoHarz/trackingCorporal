"""Decodificação de heatmaps em coordenadas normalizadas de keypoints."""

from __future__ import annotations

import torch

from src.core.types import Keypoint, Pose
from src.pose.keypoints import NUM_KEYPOINTS


def decode_heatmaps(heatmaps: torch.Tensor) -> Pose:
    """Converte heatmaps [17,H,W] ou [1,17,H,W] em uma Pose.

    A posição de cada keypoint é o máximo do respectivo heatmap. A confiança é
    limitada ao intervalo 0..1 para continuar compatível com o restante do
    pipeline.
    """
    if heatmaps.ndim == 4:
        if heatmaps.shape[0] != 1:
            raise ValueError("decode_heatmaps aceita apenas batch de tamanho 1.")
        heatmaps = heatmaps[0]

    if heatmaps.ndim != 3:
        raise ValueError("heatmaps deve possuir formato [K,H,W] ou [1,K,H,W].")
    if heatmaps.shape[0] != NUM_KEYPOINTS:
        raise ValueError(
            f"Esperados {NUM_KEYPOINTS} heatmaps, recebidos {heatmaps.shape[0]}."
        )

    heatmaps = heatmaps.detach().float().cpu()
    _, height, width = heatmaps.shape
    points: list[Keypoint] = []

    for heatmap in heatmaps:
        flat_index = int(torch.argmax(heatmap).item())
        y = flat_index // width
        x = flat_index % width
        confidence = float(torch.clamp(heatmap[y, x], 0.0, 1.0).item())

        normalized_x = x / (width - 1) if width > 1 else 0.0
        normalized_y = y / (height - 1) if height > 1 else 0.0

        points.append(
            Keypoint(
                x=normalized_x,
                y=normalized_y,
                confidence=confidence,
            )
        )

    return Pose(points)
