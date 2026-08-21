"""Testes da loss de heatmap usada no treinamento."""

import torch

from scripts.train_pose import masked_heatmap_loss
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS


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


def test_leg_weight_penalizes_missing_knee_more_than_missing_nose():
    target = torch.zeros((1, NUM_KEYPOINTS, 8, 8), dtype=torch.float32)
    visibility = torch.zeros((1, NUM_KEYPOINTS, 1, 1), dtype=torch.float32)

    nose = int(BodyKeypoint.NOSE)
    knee = int(BodyKeypoint.LEFT_KNEE)
    target[0, nose, 4, 4] = 1.0
    target[0, knee, 4, 4] = 1.0
    visibility[0, nose] = 1.0
    visibility[0, knee] = 1.0

    misses_nose = target.clone()
    misses_nose[0, nose, 4, 4] = 0.0

    misses_knee = target.clone()
    misses_knee[0, knee, 4, 4] = 0.0

    nose_loss = masked_heatmap_loss(
        misses_nose,
        target,
        visibility,
        positive_weight=8.0,
        leg_keypoint_weight=2.0,
    )
    knee_loss = masked_heatmap_loss(
        misses_knee,
        target,
        visibility,
        positive_weight=8.0,
        leg_keypoint_weight=2.0,
    )

    assert knee_loss > nose_loss
