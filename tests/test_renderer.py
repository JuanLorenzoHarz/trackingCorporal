"""Testes básicos da renderização do esqueleto."""

import numpy as np

from scripts.demo_skeleton import build_demo_pose
from src.visualization.renderer import draw_pose, normalized_to_pixel


def test_normalized_to_pixel_center():
    assert normalized_to_pixel(0.5, 0.5, 101, 101) == (50, 50)


def test_draw_pose_changes_frame():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    pose = build_demo_pose()

    result = draw_pose(frame, pose)

    assert result.shape == frame.shape
    assert np.any(result != 0)
