"""Testes da primeira CNN e da representação por heatmaps."""

import torch

from src.pose.decoder import decode_heatmaps
from src.pose.heatmaps import generate_heatmaps
from src.pose.keypoints import NUM_KEYPOINTS
from src.pose.model import PoseNet


def test_pose_model_output_shape():
    model = PoseNet()
    model.eval()

    image = torch.zeros((1, 3, 256, 256), dtype=torch.float32)

    with torch.inference_mode():
        output = model(image)

    assert output.shape == (1, NUM_KEYPOINTS, 64, 64)


def test_generate_heatmaps():
    keypoints = torch.zeros((NUM_KEYPOINTS, 2), dtype=torch.float32)
    visibility = torch.zeros(NUM_KEYPOINTS, dtype=torch.float32)

    keypoints[0] = torch.tensor([0.5, 0.5])
    visibility[0] = 1.0

    heatmaps = generate_heatmaps(
        keypoints,
        visibility,
        heatmap_size=64,
        sigma=1.8,
    )

    assert heatmaps.shape == (NUM_KEYPOINTS, 64, 64)
    assert float(heatmaps[0].max()) > 0.9
    assert float(heatmaps[1].max()) == 0.0


def test_decode_heatmap_peak():
    heatmaps = torch.zeros((1, NUM_KEYPOINTS, 64, 64), dtype=torch.float32)
    heatmaps[0, 0, 21, 42] = 0.95

    pose = decode_heatmaps(heatmaps)
    nose = pose[0]

    assert abs(nose.x - (42 / 63)) < 1e-6
    assert abs(nose.y - (21 / 63)) < 1e-6
    assert abs(nose.confidence - 0.95) < 1e-5
