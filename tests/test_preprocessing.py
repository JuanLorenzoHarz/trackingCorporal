"""Testes do pré-processamento sem distorção da webcam."""

import numpy as np

from src.core.types import Keypoint, Pose
from src.preprocessing.frame_preprocessor import (
    preprocess_frame,
    preprocess_frame_with_transform,
)


def test_preprocess_frame_keeps_expected_tensor_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tensor = preprocess_frame(frame, input_size=256)

    assert tensor.shape == (1, 3, 256, 256)


def test_widescreen_frame_uses_center_square_crop_and_remaps_pose():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tensor, transform = preprocess_frame_with_transform(frame, input_size=256)

    assert tensor.shape == (1, 3, 256, 256)
    assert transform.crop_size == 480
    assert transform.crop_left == 80
    assert transform.crop_top == 0

    crop_pose = Pose(
        [
            Keypoint(0.0, 0.0, 1.0),
            Keypoint(0.5, 0.5, 1.0),
            Keypoint(1.0, 1.0, 1.0),
        ]
    )
    original = transform.pose_to_original(crop_pose)

    assert abs(original[0].x - 0.125) < 1e-6
    assert abs(original[0].y - 0.0) < 1e-6
    assert abs(original[1].x - 0.5) < 1e-6
    assert abs(original[1].y - 0.5) < 1e-6
    assert abs(original[2].x - 0.875) < 1e-6
    assert abs(original[2].y - 1.0) < 1e-6


def test_portrait_frame_remaps_vertical_crop():
    frame = np.zeros((640, 480, 3), dtype=np.uint8)

    _, transform = preprocess_frame_with_transform(frame, input_size=256)

    assert transform.crop_size == 480
    assert transform.crop_left == 0
    assert transform.crop_top == 80

    center = transform.pose_to_original(Pose([Keypoint(0.5, 0.5, 1.0)]))
    assert abs(center[0].x - 0.5) < 1e-6
    assert abs(center[0].y - 0.5) < 1e-6
