"""Testes do gate de presença corporal."""

from src.core.types import Keypoint, Pose
from src.pose.demo_pose import build_demo_pose
from src.pose.keypoints import NUM_KEYPOINTS
from src.tracking.presence import BodyPresenceGate


def empty_pose() -> Pose:
    return Pose([Keypoint(0.0, 0.0, 0.0) for _ in range(NUM_KEYPOINTS)])


def test_presence_requires_multiple_consistent_frames_to_acquire():
    gate = BodyPresenceGate(
        detection_threshold=0.5,
        acquire_frames=3,
        release_frames=2,
    )
    pose = build_demo_pose()

    first = gate.update(pose)
    second = gate.update(pose)
    third = gate.update(pose)

    assert not first.active
    assert not second.active
    assert third.active
    assert third.acquired


def test_presence_releases_after_several_empty_frames():
    gate = BodyPresenceGate(
        detection_threshold=0.5,
        acquire_frames=1,
        release_frames=2,
    )
    assert gate.update(build_demo_pose()).active

    first_empty = gate.update(empty_pose())
    second_empty = gate.update(empty_pose())

    assert first_empty.active
    assert not second_empty.active
    assert second_empty.lost


def test_presence_rejects_scattered_points_without_torso():
    gate = BodyPresenceGate(
        detection_threshold=0.5,
        acquire_frames=1,
    )
    points = [Keypoint(0.1 + i * 0.02, 0.1, 1.0) for i in range(NUM_KEYPOINTS)]
    # Remove toda evidência do tronco, mesmo mantendo vários picos fortes.
    for index in (5, 6, 11, 12):
        points[index] = Keypoint(0.0, 0.0, 0.0)

    report = gate.update(Pose(points))
    assert not report.active
    assert report.torso_keypoints == 0
