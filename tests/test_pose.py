"""Testes das definições básicas de pose."""

from src.core.types import Keypoint, Pose
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS, SKELETON_CONNECTIONS


def test_number_of_keypoints():
    assert NUM_KEYPOINTS == 17


def test_keypoint_indices():
    assert BodyKeypoint.NOSE == 0
    assert BodyKeypoint.LEFT_WRIST == 9
    assert BodyKeypoint.RIGHT_ANKLE == 16


def test_skeleton_has_connections():
    assert len(SKELETON_CONNECTIONS) > 0


def test_valid_keypoint():
    point = Keypoint(x=0.5, y=0.5, confidence=0.9)
    assert point.is_valid()


def test_low_confidence_keypoint():
    point = Keypoint(x=0.5, y=0.5, confidence=0.2)
    assert not point.is_valid()


def test_keypoint_outside_image():
    point = Keypoint(x=1.2, y=0.5, confidence=1.0)
    assert not point.is_valid()


def test_pose_size_and_index_access():
    points = [Keypoint(x=0.5, y=0.5, confidence=1.0) for _ in range(NUM_KEYPOINTS)]
    pose = Pose(points)

    assert len(pose) == 17
    assert pose[BodyKeypoint.LEFT_WRIST] is points[9]
