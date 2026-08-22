"""Testes das losses da PoseNet V2."""

import torch

from scripts.train_pose_v2 import focal_heatmap_loss, masked_offset_loss


def test_negative_frame_penalizes_false_person_centers():
    target = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    hallucinating = torch.zeros_like(target)
    quiet = torch.full_like(target, -8.0)

    hallucination_loss = focal_heatmap_loss(hallucinating, target)
    quiet_loss = focal_heatmap_loss(quiet, target)

    assert hallucination_loss > quiet_loss
    assert quiet_loss >= 0.0


def test_offset_loss_only_uses_supervised_cells():
    prediction = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    target = torch.zeros_like(prediction)
    mask = torch.zeros((1, 2, 8, 8), dtype=torch.float32)

    # Apenas o vetor do keypoint 0 em (3,4) é supervisionado.
    target[0, 0, 3, 4] = 2.0
    target[0, 1, 3, 4] = -1.0
    mask[0, 0, 3, 4] = 1.0

    supervised_loss = masked_offset_loss(prediction, target, mask)

    # Um erro enorme em célula sem máscara não pode alterar a loss.
    prediction[0, 2, 1, 1] = 1000.0
    ignored_loss = masked_offset_loss(prediction, target, mask)

    assert supervised_loss > 0.0
    assert torch.allclose(supervised_loss, ignored_loss)
