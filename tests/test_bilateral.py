"""Testes da estabilização esquerda/direita do corpo."""

from src.core.types import Keypoint, Pose
from src.pose.demo_pose import build_demo_pose
from src.pose.keypoints import BodyKeypoint
from src.tracking.bilateral import BilateralLegIdentityTracker


def copy_pose(pose: Pose) -> Pose:
    return Pose([Keypoint(p.x, p.y, p.confidence) for p in pose.keypoints])


def test_bilateral_tracker_corrects_left_right_leg_pairs_without_flipping_hips():
    tracker = BilateralLegIdentityTracker(detection_threshold=0.5)
    reference = build_demo_pose()
    tracker.update(reference)

    swapped = copy_pose(reference)
    for left, right in (
        (BodyKeypoint.LEFT_KNEE, BodyKeypoint.RIGHT_KNEE),
        (BodyKeypoint.LEFT_ANKLE, BodyKeypoint.RIGHT_ANKLE),
    ):
        li, ri = int(left), int(right)
        swapped.keypoints[li], swapped.keypoints[ri] = swapped.keypoints[ri], swapped.keypoints[li]

    corrected, report = tracker.update(swapped)
    assert report.swapped_chain
    assert abs(corrected[int(BodyKeypoint.LEFT_KNEE)].x - reference[int(BodyKeypoint.LEFT_KNEE)].x) < 1e-6
    assert abs(corrected[int(BodyKeypoint.RIGHT_ANKLE)].x - reference[int(BodyKeypoint.RIGHT_ANKLE)].x) < 1e-6
    assert abs(corrected[int(BodyKeypoint.LEFT_HIP)].x - reference[int(BodyKeypoint.LEFT_HIP)].x) < 1e-6
    assert abs(corrected[int(BodyKeypoint.RIGHT_HIP)].x - reference[int(BodyKeypoint.RIGHT_HIP)].x) < 1e-6


def test_bilateral_tracker_rejects_duplicate_leg_after_clear_separation():
    tracker = BilateralLegIdentityTracker(detection_threshold=0.5, collapse_ratio=0.20)
    reference = build_demo_pose()
    tracker.update(reference)

    collapsed = copy_pose(reference)
    left_knee = int(BodyKeypoint.LEFT_KNEE)
    right_knee = int(BodyKeypoint.RIGHT_KNEE)
    left_ankle = int(BodyKeypoint.LEFT_ANKLE)
    right_ankle = int(BodyKeypoint.RIGHT_ANKLE)
    collapsed.keypoints[right_knee] = Keypoint(reference[left_knee].x, reference[left_knee].y, 0.90)
    collapsed.keypoints[right_ankle] = Keypoint(reference[left_ankle].x, reference[left_ankle].y, 0.90)

    corrected, report = tracker.update(collapsed)
    assert report.collapsed_pairs == 2
    assert right_knee in report.rejected_indices
    assert right_ankle in report.rejected_indices
    assert corrected[right_knee].confidence == 0.0
    assert corrected[right_ankle].confidence == 0.0
    assert corrected[left_knee].confidence > 0.0


def test_bilateral_tracker_rejects_duplicate_shoulder_after_history():
    tracker = BilateralLegIdentityTracker(detection_threshold=0.5, collapse_ratio=0.20)
    reference = build_demo_pose()
    tracker.update(reference)

    collapsed = copy_pose(reference)
    left = int(BodyKeypoint.LEFT_SHOULDER)
    right = int(BodyKeypoint.RIGHT_SHOULDER)
    collapsed.keypoints[right] = Keypoint(reference[left].x, reference[left].y, 0.90)

    corrected, report = tracker.update(collapsed)
    assert report.collapsed_pairs >= 1
    assert right in report.rejected_indices
    assert corrected[right].confidence == 0.0


def test_bilateral_tracker_does_not_reject_close_legs_without_previous_separation():
    tracker = BilateralLegIdentityTracker(detection_threshold=0.5, collapse_ratio=0.20)
    pose = build_demo_pose()
    close = copy_pose(pose)
    left_knee = int(BodyKeypoint.LEFT_KNEE)
    right_knee = int(BodyKeypoint.RIGHT_KNEE)
    close.keypoints[right_knee] = Keypoint(close[left_knee].x + 0.005, close[left_knee].y, 1.0)

    corrected, report = tracker.update(close)
    assert report.collapsed_pairs == 0
    assert not report.rejected_indices
    assert corrected[right_knee].confidence == 1.0
