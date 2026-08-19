"""Geração dos heatmaps usados como alvo da rede de pose."""

from __future__ import annotations

import torch


def generate_heatmaps(
    keypoints: torch.Tensor,
    visibility: torch.Tensor,
    heatmap_size: int = 64,
    sigma: float = 1.8,
) -> torch.Tensor:
    """Gera um heatmap gaussiano para cada keypoint.

    Args:
        keypoints: tensor [K, 2] com coordenadas normalizadas entre 0 e 1.
        visibility: tensor [K] onde valores > 0 indicam pontos anotados/visíveis.
        heatmap_size: largura e altura do heatmap quadrado.
        sigma: desvio padrão da gaussiana em pixels do heatmap.
    """
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("keypoints deve possuir formato [K, 2].")

    device = keypoints.device
    dtype = keypoints.dtype
    keypoint_count = keypoints.shape[0]

    ys = torch.arange(heatmap_size, device=device, dtype=dtype)
    xs = torch.arange(heatmap_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    heatmaps = torch.zeros(
        (keypoint_count, heatmap_size, heatmap_size),
        device=device,
        dtype=dtype,
    )

    for index in range(keypoint_count):
        if visibility[index] <= 0:
            continue

        x = keypoints[index, 0]
        y = keypoints[index, 1]

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue

        center_x = x * (heatmap_size - 1)
        center_y = y * (heatmap_size - 1)
        squared_distance = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2

        heatmaps[index] = torch.exp(
            -squared_distance / (2.0 * sigma * sigma)
        )

    return heatmaps
