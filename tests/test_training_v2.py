"""Testes das losses da PoseNet V2."""

import torch

from scripts.train_pose_v2 import (
    build_keypoint_weights,
    focal_heatmap_loss,
    masked_offset_loss,
)
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS


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


def test_extremity_weights_prioritize_wrists_and_ankles():
    weights = build_keypoint_weights(NUM_KEYPOINTS)

    hip = int(BodyKeypoint.LEFT_HIP)
    shoulder = int(BodyKeypoint.LEFT_SHOULDER)
    elbow = int(BodyKeypoint.LEFT_ELBOW)
    wrist = int(BodyKeypoint.LEFT_WRIST)
    knee = int(BodyKeypoint.LEFT_KNEE)
    ankle = int(BodyKeypoint.LEFT_ANKLE)

    assert weights[hip] == 1.0
    assert weights[shoulder] > weights[hip]
    assert weights[elbow] > weights[shoulder]
    assert weights[wrist] > weights[elbow]
    assert weights[knee] > weights[hip]
    assert weights[ankle] > weights[knee]
    assert weights[ankle] > weights[wrist]


def test_weighted_heatmap_loss_penalizes_missed_ankle_more_than_hip():
    target = torch.zeros((1, NUM_KEYPOINTS, 8, 8), dtype=torch.float32)
    hip = int(BodyKeypoint.LEFT_HIP)
    ankle = int(BodyKeypoint.LEFT_ANKLE)
    target[0, hip, 4, 2] = 1.0
    target[0, ankle, 4, 6] = 1.0

    weights = build_keypoint_weights(NUM_KEYPOINTS)

    # Logits altos apenas no target que NÃO foi perdido; o outro fica muito negativo.
    misses_hip = torch.full_like(target, -8.0)
    misses_hip[0, ankle, 4, 6] = 8.0

    misses_ankle = torch.full_like(target, -8.0)
    misses_ankle[0, hip, 4, 2] = 8.0

    hip_loss = focal_heatmap_loss(
        misses_hip,
        target,
        channel_weights=weights,
    )
    ankle_loss = focal_heatmap_loss(
        misses_ankle,
        target,
        channel_weights=weights,
    )

    assert ankle_loss > hip_loss


def test_weighted_offset_loss_prioritizes_extremity_vector():
    prediction = torch.zeros((1, NUM_KEYPOINTS * 2, 8, 8), dtype=torch.float32)
    target = torch.zeros_like(prediction)
    mask = torch.zeros((1, NUM_KEYPOINTS, 8, 8), dtype=torch.float32)

    hip = int(BodyKeypoint.LEFT_HIP)
    ankle = int(BodyKeypoint.LEFT_ANKLE)
    weights = build_keypoint_weights(NUM_KEYPOINTS)

    target[0, 2 * hip, 3, 3] = 2.0
    mask[0, hip, 3, 3] = 1.0
    hip_loss = masked_offset_loss(
        prediction,
        target,
        mask,
        keypoint_weights=weights,
    )

    target.zero_()
    mask.zero_()
    target[0, 2 * ankle, 3, 3] = 2.0
    mask[0, ankle, 3, 3] = 1.0
    ankle_loss = masked_offset_loss(
        prediction,
        target,
        mask,
        keypoint_weights=weights,
    )

    assert ankle_loss > hip_loss
