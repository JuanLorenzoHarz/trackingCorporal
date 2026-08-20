"""Testes do tracking temporal e suavização de pose."""

from src.core.types import Keypoint, Pose
from src.pose.demo_pose import build_demo_pose
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS
from src.tracking.smoothing import ExponentialPoseSmoother
from src.tracking.temporal_tracker import TemporalPoseTracker


def pose_with_hidden_keypoint(pose: Pose, index: int) -> Pose:
    points = [
        Keypoint(point.x, point.y, point.confidence)
        for point in pose.keypoints
    ]
    points[index] = Keypoint(0.0, 0.0, 0.0)
    return Pose(points)


def test_tracker_predicts_temporarily_hidden_shoulder_from_neighbors():
    tracker = TemporalPoseTracker(
        detection_threshold=0.5,
        max_missing_frames=3,
        confidence_decay=0.8,
        anatomy_weight=1.0,
    )
    visible_pose = build_demo_pose()
    tracker.update(visible_pose)

    shoulder_index = int(BodyKeypoint.LEFT_SHOULDER)
    hidden_pose = pose_with_hidden_keypoint(visible_pose, shoulder_index)
    tracked = tracker.update(hidden_pose)

    expected = visible_pose[shoulder_index]
    predicted = tracked[shoulder_index]

    assert tracker.predicted_count == 1
    assert shoulder_index in tracker.predicted_indices
    assert abs(predicted.x - expected.x) < 1e-6
    assert abs(predicted.y - expected.y) < 1e-6
    assert abs(predicted.confidence - 0.8) < 1e-6


def test_tracker_expires_prediction_after_missing_limit():
    tracker = TemporalPoseTracker(
        detection_threshold=0.5,
        max_missing_frames=2,
        confidence_decay=0.8,
    )
    visible_pose = build_demo_pose()
    tracker.update(visible_pose)

    shoulder_index = int(BodyKeypoint.LEFT_SHOULDER)
    hidden_pose = pose_with_hidden_keypoint(visible_pose, shoulder_index)

    first = tracker.update(hidden_pose)
    second = tracker.update(hidden_pose)
    third = tracker.update(hidden_pose)

    assert first[shoulder_index].confidence > 0.0
    assert second[shoulder_index].confidence > 0.0
    assert third[shoulder_index].confidence == 0.0
    assert shoulder_index not in tracker.predicted_indices


def test_smoother_reduces_coordinate_jump():
    smoother = ExponentialPoseSmoother(alpha=0.5)

    first_pose = Pose(
        [Keypoint(0.2, 0.2, 1.0) for _ in range(NUM_KEYPOINTS)]
    )
    second_pose = Pose(
        [Keypoint(0.8, 0.8, 1.0) for _ in range(NUM_KEYPOINTS)]
    )

    smoother.update(first_pose)
    result = smoother.update(second_pose)

    assert abs(result[0].x - 0.5) < 1e-6
    assert abs(result[0].y - 0.5) < 1e-6
    assert result[0].confidence == 1.0
