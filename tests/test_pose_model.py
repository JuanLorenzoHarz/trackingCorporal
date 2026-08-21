"""Testes da primeira CNN e da representação por heatmaps."""

import torch

from src.pose.decoder import decode_heatmaps, decode_heatmaps_bilateral
from src.pose.heatmaps import generate_heatmaps
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS
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


def test_bilateral_decoder_uses_second_peak_when_legs_collapse():
    heatmaps = torch.zeros((1, NUM_KEYPOINTS, 64, 64), dtype=torch.float32)
    left_knee = int(BodyKeypoint.LEFT_KNEE)
    right_knee = int(BodyKeypoint.RIGHT_KNEE)

    # Os dois canais preferem inicialmente a perna da esquerda.
    heatmaps[0, left_knee, 40, 20] = 0.90
    heatmaps[0, right_knee, 40, 20] = 0.86

    # O canal direito também possui uma segunda hipótese quase tão boa na perna direita.
    heatmaps[0, right_knee, 40, 44] = 0.82

    pose, report = decode_heatmaps_bilateral(
        heatmaps,
        top_k=3,
        suppression_radius=3,
        minimum_separation_pixels=3.0,
        minimum_alternative_ratio=0.65,
    )

    assert report.corrected_pairs == 1
    assert abs(pose[left_knee].x - (20 / 63)) < 1e-6
    assert abs(pose[right_knee].x - (44 / 63)) < 1e-6
    assert abs(pose[right_knee].confidence - 0.82) < 1e-5


def test_bilateral_decoder_does_not_promote_weak_noise():
    heatmaps = torch.zeros((1, NUM_KEYPOINTS, 64, 64), dtype=torch.float32)
    left_ankle = int(BodyKeypoint.LEFT_ANKLE)
    right_ankle = int(BodyKeypoint.RIGHT_ANKLE)

    heatmaps[0, left_ankle, 50, 25] = 0.90
    heatmaps[0, right_ankle, 50, 25] = 0.88
    heatmaps[0, right_ankle, 50, 46] = 0.20

    pose, report = decode_heatmaps_bilateral(
        heatmaps,
        minimum_alternative_ratio=0.65,
    )

    assert report.corrected_pairs == 0
    assert abs(pose[right_ankle].x - (25 / 63)) < 1e-6
