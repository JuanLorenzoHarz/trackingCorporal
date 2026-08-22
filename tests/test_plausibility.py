"""Testes da métrica de plausibilidade anatômica e temporal."""

from src.core.types import Keypoint, Pose
from src.pose.demo_pose import build_demo_pose
from src.pose.keypoints import BodyKeypoint
from src.tracking.plausibility import PosePlausibilityEvaluator


def copy_pose(pose: Pose) -> Pose:
    return Pose(
        [Keypoint(point.x, point.y, point.confidence) for point in pose.keypoints]
    )


def calibrate(evaluator: PosePlausibilityEvaluator, frames: int = 8) -> Pose:
    pose = build_demo_pose()
    for _ in range(frames):
        evaluator.evaluate_and_filter(pose)
    return pose


def test_plausibility_calibrates_stable_pose():
    evaluator = PosePlausibilityEvaluator(
        detection_threshold=0.5,
        min_baseline_observations=4,
    )
    pose = calibrate(evaluator, frames=5)

    _, report = evaluator.evaluate_and_filter(pose)

    assert report.calibrated_segments == 12
    assert report.percentage > 95.0
    assert not report.suspicious_indices


def test_plausibility_rejects_impossible_knee_jump():
    evaluator = PosePlausibilityEvaluator(
        detection_threshold=0.5,
        min_baseline_observations=4,
        reject_threshold=0.16,
    )
    pose = calibrate(evaluator, frames=6)

    knee_index = int(BodyKeypoint.LEFT_KNEE)
    corrupted = copy_pose(pose)
    corrupted.keypoints[knee_index] = Keypoint(0.95, 0.08, 1.0)

    filtered, report = evaluator.evaluate_and_filter(corrupted)

    assert knee_index in report.suspicious_indices
    assert filtered[knee_index].confidence == 0.0
    assert report.percentage < 95.0


def test_plausibility_accepts_bent_leg_with_consistent_lengths():
    evaluator = PosePlausibilityEvaluator(
        detection_threshold=0.5,
        min_baseline_observations=4,
        reject_threshold=0.16,
    )
    pose = calibrate(evaluator, frames=6)

    knee_index = int(BodyKeypoint.LEFT_KNEE)
    ankle_index = int(BodyKeypoint.LEFT_ANKLE)
    bent = copy_pose(pose)

    # Mantém comprimentos próximos aos observados, mas muda bastante a direção
    # da perna. Isso representa uma pose possível que não deve ser "endireitada".
    bent.keypoints[knee_index] = Keypoint(0.30, 0.70, 1.0)
    bent.keypoints[ankle_index] = Keypoint(0.20, 0.84, 1.0)

    filtered, report = evaluator.evaluate_and_filter(bent)

    assert knee_index not in report.suspicious_indices
    assert ankle_index not in report.suspicious_indices
    assert filtered[knee_index].confidence == 1.0
    assert filtered[ankle_index].confidence == 1.0


def test_plausibility_accepts_whole_body_translation():
    evaluator = PosePlausibilityEvaluator(
        detection_threshold=0.5,
        min_baseline_observations=4,
    )
    pose = calibrate(evaluator, frames=6)

    moved = Pose(
        [
            Keypoint(point.x + 0.02, point.y + 0.02, point.confidence)
            for point in pose.keypoints
        ]
    )
    _, report = evaluator.evaluate_and_filter(moved)

    assert not report.suspicious_indices
    assert report.percentage > 90.0
