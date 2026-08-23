"""Testes da arquitetura e decoder multi-pessoa da PoseNet V2."""

import torch

from src.pose.decoder_v2 import decode_pose_v2
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS
from src.pose.model import PoseNet
from src.pose.model_v2 import PoseNetV2


def _empty_output(size: int = 64) -> dict[str, torch.Tensor]:
    return {
        "center": torch.full((1, 1, size, size), -10.0),
        "keypoints": torch.full((1, NUM_KEYPOINTS, size, size), -10.0),
        "center_offsets": torch.zeros((1, NUM_KEYPOINTS * 2, size, size)),
        "parent_offsets": torch.zeros((1, NUM_KEYPOINTS * 2, size, size)),
    }


def _put_keypoint(
    output: dict[str, torch.Tensor],
    keypoint: BodyKeypoint,
    x: int,
    y: int,
    center_x: int,
    center_y: int,
    logit: float = 8.0,
    parent_x: int | None = None,
    parent_y: int | None = None,
) -> None:
    index = int(keypoint)
    output["keypoints"][0, index, y, x] = logit
    output["center_offsets"][0, 2 * index, y, x] = center_x - x
    output["center_offsets"][0, 2 * index + 1, y, x] = center_y - y
    if parent_x is not None and parent_y is not None:
        output["parent_offsets"][0, 2 * index, y, x] = parent_x - x
        output["parent_offsets"][0, 2 * index + 1, y, x] = parent_y - y


def test_pose_v2_output_shapes():
    model = PoseNetV2().eval()
    image = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    with torch.inference_mode():
        output = model(image)

    assert output["center"].shape == (1, 1, 64, 64)
    assert output["keypoints"].shape == (1, NUM_KEYPOINTS, 64, 64)
    assert output["center_offsets"].shape == (1, NUM_KEYPOINTS * 2, 64, 64)
    assert output["parent_offsets"].shape == (1, NUM_KEYPOINTS * 2, 64, 64)


def test_pose_v2_reuses_compatible_v1_backbone(tmp_path):
    v1 = PoseNet()
    with torch.no_grad():
        v1.encoder1[0].weight.fill_(0.123)
    checkpoint = tmp_path / "v1.pt"
    torch.save({"model_state": v1.state_dict()}, checkpoint)

    v2 = PoseNetV2()
    copied = v2.initialize_from_v1(checkpoint)

    assert copied > 0
    assert torch.allclose(
        v2.encoder1[0].weight,
        torch.full_like(v2.encoder1[0].weight, 0.123),
    )


def test_v2_decoder_returns_no_person_without_center():
    poses, report = decode_pose_v2(_empty_output())
    assert poses == []
    assert report.person_count == 0


def test_v2_decoder_separates_two_people_by_center_votes():
    output = _empty_output()
    centers = ((18, 32), (46, 32))
    for cx, cy in centers:
        output["center"][0, 0, cy, cx] = 8.0
        _put_keypoint(output, BodyKeypoint.NOSE, cx, 20, cx, cy)
        _put_keypoint(output, BodyKeypoint.LEFT_SHOULDER, cx - 4, 28, cx, cy)
        _put_keypoint(output, BodyKeypoint.RIGHT_SHOULDER, cx + 4, 28, cx, cy)

    poses, report = decode_pose_v2(
        output,
        center_threshold=0.5,
        keypoint_threshold=0.2,
        association_radius=3.0,
    )

    assert report.person_count == 2
    assert len(poses) == 2
    centers_x = sorted(
        (pose[int(BodyKeypoint.LEFT_SHOULDER)].x + pose[int(BodyKeypoint.RIGHT_SHOULDER)].x) / 2
        for pose in poses
    )
    assert centers_x[0] < 0.5 < centers_x[1]


def test_v2_parent_offsets_prevent_crossed_wrist_x():
    output = _empty_output()
    cx, cy = 32, 32
    output["center"][0, 0, cy, cx] = 8.0

    # Âncoras para materializar a pessoa.
    _put_keypoint(output, BodyKeypoint.NOSE, 32, 18, cx, cy)
    _put_keypoint(output, BodyKeypoint.LEFT_SHOULDER, 23, 26, cx, cy)
    _put_keypoint(output, BodyKeypoint.RIGHT_SHOULDER, 41, 26, cx, cy)

    # Cotovelos corretos.
    _put_keypoint(
        output, BodyKeypoint.LEFT_ELBOW, 19, 34, cx, cy,
        parent_x=23, parent_y=26,
    )
    _put_keypoint(
        output, BodyKeypoint.RIGHT_ELBOW, 45, 34, cx, cy,
        parent_x=41, parent_y=26,
    )

    # LEFT_WRIST possui um pico MAIS FORTE no lado errado, mas esse pico aponta
    # para o cotovelo direito. O pico correto é menor e aponta para o cotovelo esquerdo.
    _put_keypoint(
        output, BodyKeypoint.LEFT_WRIST, 48, 42, cx, cy,
        logit=9.0, parent_x=45, parent_y=34,
    )
    _put_keypoint(
        output, BodyKeypoint.LEFT_WRIST, 16, 42, cx, cy,
        logit=7.0, parent_x=19, parent_y=34,
    )
    _put_keypoint(
        output, BodyKeypoint.RIGHT_WRIST, 48, 42, cx, cy,
        logit=8.0, parent_x=45, parent_y=34,
    )

    poses, report = decode_pose_v2(
        output,
        center_threshold=0.5,
        keypoint_threshold=0.2,
        association_radius=3.0,
        extremity_parent_sigma=2.0,
    )

    assert report.person_count == 1
    pose = poses[0]
    left_wrist = pose[int(BodyKeypoint.LEFT_WRIST)]
    left_elbow = pose[int(BodyKeypoint.LEFT_ELBOW)]
    right_wrist = pose[int(BodyKeypoint.RIGHT_WRIST)]

    assert left_wrist.confidence > 0.0
    assert left_wrist.x < 0.5
    assert left_wrist.x < left_elbow.x + 0.05
    assert right_wrist.x > 0.5


def test_v2_profile_keeps_close_left_right_pairs():
    """Perfil real pode projetar L/R quase no mesmo X sem ser um erro."""
    output = _empty_output()
    cx, cy = 32, 32
    output["center"][0, 0, cy, cx] = 8.0

    _put_keypoint(output, BodyKeypoint.NOSE, 32, 18, cx, cy)

    # Ombros quase sobrepostos, situação comum em perfil.
    _put_keypoint(output, BodyKeypoint.LEFT_SHOULDER, 31, 27, cx, cy)
    _put_keypoint(output, BodyKeypoint.RIGHT_SHOULDER, 32, 27, cx, cy)

    # Cotovelos também próximos, mas cada um aponta corretamente para seu ombro.
    _put_keypoint(
        output,
        BodyKeypoint.LEFT_ELBOW,
        31,
        36,
        cx,
        cy,
        parent_x=31,
        parent_y=27,
    )
    _put_keypoint(
        output,
        BodyKeypoint.RIGHT_ELBOW,
        32,
        36,
        cx,
        cy,
        parent_x=32,
        parent_y=27,
    )

    poses, report = decode_pose_v2(
        output,
        center_threshold=0.5,
        keypoint_threshold=0.2,
        association_radius=3.0,
        bilateral_min_separation=2.0,
    )

    assert report.person_count == 1
    assert report.rejected_bilateral_points == 0
    pose = poses[0]
    assert pose[int(BodyKeypoint.LEFT_SHOULDER)].confidence > 0.0
    assert pose[int(BodyKeypoint.RIGHT_SHOULDER)].confidence > 0.0
    assert pose[int(BodyKeypoint.LEFT_ELBOW)].confidence > 0.0
    assert pose[int(BodyKeypoint.RIGHT_ELBOW)].confidence > 0.0
