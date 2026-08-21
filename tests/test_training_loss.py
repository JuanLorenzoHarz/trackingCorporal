"""Testes da loss de heatmap usada no treinamento."""

import torch

from scripts.train_pose import masked_heatmap_loss


def test_positive_heatmap_weight_emphasizes_keypoint_peak():
    prediction = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    target = torch.zeros_like(prediction)
    target[0, 0, 4, 4] = 1.0
    visibility = torch.ones((1, 1, 1, 1), dtype=torch.float32)

    old_style = masked_heatmap_loss(
        prediction,
        target,
        visibility,
        positive_weight=0.0,
    )
    weighted = masked_heatmap_loss(
        prediction,
        target,
        visibility,
        positive_weight=8.0,
    )

    assert weighted > old_style


def test_invisible_keypoint_does_not_affect_loss():
    prediction = torch.ones((1, 1, 8, 8), dtype=torch.float32)
    target = torch.zeros_like(prediction)
    visibility = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

    loss = masked_heatmap_loss(
        prediction,
        target,
        visibility,
        positive_weight=8.0,
    )

    assert float(loss.item()) == 0.0
