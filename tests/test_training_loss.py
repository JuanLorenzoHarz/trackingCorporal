"""Testes da loss de heatmap usada no treinamento."""

import torch

from scripts.train_pose import bilateral_cross_peak_loss, masked_heatmap_loss
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


def test_bilateral_loss_penalizes_both_knees_on_same_target():
    target = torch.zeros((1, NUM_KEYPOINTS, 16, 16), dtype=torch.float32)
    visibility = torch.zeros((1, NUM_KEYPOINTS, 1, 1), dtype=torch.float32)
    left_knee = int(BodyKeypoint.LEFT_KNEE)
    right_knee = int(BodyKeypoint.RIGHT_KNEE)

    target[0, left_knee, 10, 4] = 1.0
    target[0, right_knee, 10, 12] = 1.0
    visibility[0, left_knee] = 1.0
    visibility[0, right_knee] = 1.0

    correct = target.clone()
    collapsed = target.clone()
    collapsed[0, right_knee].zero_()
    collapsed[0, right_knee, 10, 4] = 1.0

    correct_loss = bilateral_cross_peak_loss(
        correct,
        target,
        visibility,
        minimum_target_distance=3.0,
    )
    collapsed_loss = bilateral_cross_peak_loss(
        collapsed,
        target,
        visibility,
        minimum_target_distance=3.0,
    )

    assert float(correct_loss.item()) == 0.0
    assert collapsed_loss > correct_loss


def test_bilateral_loss_ignores_overlapping_ground_truth():
    target = torch.zeros((1, NUM_KEYPOINTS, 16, 16), dtype=torch.float32)
    visibility = torch.zeros((1, NUM_KEYPOINTS, 1, 1), dtype=torch.float32)
    left_ankle = int(BodyKeypoint.LEFT_ANKLE)
    right_ankle = int(BodyKeypoint.RIGHT_ANKLE)

    target[0, left_ankle, 12, 8] = 1.0
    target[0, right_ankle, 12, 8] = 1.0
    visibility[0, left_ankle] = 1.0
    visibility[0, right_ankle] = 1.0

    prediction = target.clone()
    loss = bilateral_cross_peak_loss(
        prediction,
        target,
        visibility,
        minimum_target_distance=3.0,
    )

    assert float(loss.item()) == 0.0
