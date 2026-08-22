"""Estabilização temporal da identidade esquerda/direita do corpo."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from statistics import mean, median

from src.core.types import Keypoint, Pose
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS


TRACKED_PAIRS: tuple[tuple[int, int], ...] = (
    (int(BodyKeypoint.LEFT_SHOULDER), int(BodyKeypoint.RIGHT_SHOULDER)),
    (int(BodyKeypoint.LEFT_ELBOW), int(BodyKeypoint.RIGHT_ELBOW)),
    (int(BodyKeypoint.LEFT_WRIST), int(BodyKeypoint.RIGHT_WRIST)),
    (int(BodyKeypoint.LEFT_HIP), int(BodyKeypoint.RIGHT_HIP)),
    (int(BodyKeypoint.LEFT_KNEE), int(BodyKeypoint.RIGHT_KNEE)),
    (int(BodyKeypoint.LEFT_ANKLE), int(BodyKeypoint.RIGHT_ANKLE)),
)

COLLAPSE_PAIRS: tuple[tuple[int, int], ...] = (
    (int(BodyKeypoint.LEFT_SHOULDER), int(BodyKeypoint.RIGHT_SHOULDER)),
    (int(BodyKeypoint.LEFT_HIP), int(BodyKeypoint.RIGHT_HIP)),
    (int(BodyKeypoint.LEFT_KNEE), int(BodyKeypoint.RIGHT_KNEE)),
    (int(BodyKeypoint.LEFT_ANKLE), int(BodyKeypoint.RIGHT_ANKLE)),
)


@dataclass(frozen=True, slots=True)
class BilateralLegReport:
    score: float
    swapped_chain: bool
    collapsed_pairs: int
    rejected_indices: frozenset[int]

    @property
    def percentage(self) -> float:
        return self.score * 100.0


class BilateralLegIdentityTracker:
    """Mantém identidade L/R por par e rejeita duplicatas evidentes."""

    def __init__(
        self,
        detection_threshold: float = 0.15,
        swap_ratio: float = 0.75,
        swap_margin_ratio: float = 0.10,
        collapse_ratio: float = 0.12,
        previous_separation_factor: float = 1.8,
    ) -> None:
        self.detection_threshold = detection_threshold
        self.swap_ratio = swap_ratio
        self.swap_margin_ratio = swap_margin_ratio
        self.collapse_ratio = collapse_ratio
        self.previous_separation_factor = previous_separation_factor
        self._previous_positions: list[tuple[float, float] | None] = [None] * NUM_KEYPOINTS
        self._previous_hip_width: float | None = None
        self._previous_shoulder_width: float | None = None

    def reset(self) -> None:
        self._previous_positions = [None] * NUM_KEYPOINTS
        self._previous_hip_width = None
        self._previous_shoulder_width = None

    def update(self, pose: Pose) -> tuple[Pose, BilateralLegReport]:
        if len(pose) != NUM_KEYPOINTS:
            raise ValueError(f"BilateralLegIdentityTracker espera {NUM_KEYPOINTS} keypoints.")

        working = Pose([Keypoint(p.x, p.y, p.confidence) for p in pose.keypoints])
        current_hip_width = self._pair_width(working, int(BodyKeypoint.LEFT_HIP), int(BodyKeypoint.RIGHT_HIP))
        current_shoulder_width = self._pair_width(working, int(BodyKeypoint.LEFT_SHOULDER), int(BodyKeypoint.RIGHT_SHOULDER))
        scales = [v for v in (self._previous_hip_width, self._previous_shoulder_width, current_hip_width, current_shoulder_width) if v is not None and v > 1e-4]
        reference_scale = median(scales) if scales else 0.12

        swapped_any = False
        for left_index, right_index in TRACKED_PAIRS:
            if self._maybe_swap_pair(working, left_index, right_index, reference_scale):
                swapped_any = True

        rejected: set[int] = set()
        pair_scores: list[float] = []
        collapsed_pairs = 0
        collapse_threshold = reference_scale * self.collapse_ratio

        for left_index, right_index in COLLAPSE_PAIRS:
            left = working[left_index]
            right = working[right_index]
            if not (left.is_valid(self.detection_threshold) and right.is_valid(self.detection_threshold)):
                continue

            separation = self._distance(left, right)
            previous_left = self._previous_positions[left_index]
            previous_right = self._previous_positions[right_index]
            if previous_left is None or previous_right is None:
                pair_scores.append(1.0)
                continue

            previous_separation = hypot(previous_left[0] - previous_right[0], previous_left[1] - previous_right[1])
            had_clear_separation = previous_separation > collapse_threshold * self.previous_separation_factor
            is_collapsed = separation < collapse_threshold and had_clear_separation

            if not had_clear_separation:
                pair_scores.append(1.0)
            else:
                target = max(collapse_threshold * self.previous_separation_factor, previous_separation * 0.35)
                pair_scores.append(min(1.0, separation / max(target, 1e-6)))

            if is_collapsed:
                collapsed_pairs += 1
                reject_index = self._choose_duplicate_to_reject(working, left_index, right_index)
                if reject_index is not None:
                    rejected.add(reject_index)
                    point = working[reject_index]
                    working.keypoints[reject_index] = Keypoint(point.x, point.y, 0.0)

        score = mean(pair_scores) if pair_scores else 1.0
        if swapped_any:
            score = min(score, 0.80)

        self._update_history(working)
        hip_width = self._pair_width(working, int(BodyKeypoint.LEFT_HIP), int(BodyKeypoint.RIGHT_HIP))
        shoulder_width = self._pair_width(working, int(BodyKeypoint.LEFT_SHOULDER), int(BodyKeypoint.RIGHT_SHOULDER))
        if hip_width is not None and hip_width > collapse_threshold:
            self._previous_hip_width = hip_width
        if shoulder_width is not None and shoulder_width > collapse_threshold:
            self._previous_shoulder_width = shoulder_width

        return working, BilateralLegReport(
            score=max(0.0, min(1.0, score)),
            swapped_chain=swapped_any,
            collapsed_pairs=collapsed_pairs,
            rejected_indices=frozenset(rejected),
        )

    def _maybe_swap_pair(self, pose: Pose, left_index: int, right_index: int, reference_scale: float) -> bool:
        left = pose[left_index]
        right = pose[right_index]
        previous_left = self._previous_positions[left_index]
        previous_right = self._previous_positions[right_index]
        if not (
            left.is_valid(self.detection_threshold)
            and right.is_valid(self.detection_threshold)
            and previous_left is not None
            and previous_right is not None
        ):
            return False

        direct_cost = hypot(left.x - previous_left[0], left.y - previous_left[1]) + hypot(right.x - previous_right[0], right.y - previous_right[1])
        swapped_cost = hypot(left.x - previous_right[0], left.y - previous_right[1]) + hypot(right.x - previous_left[0], right.y - previous_left[1])
        margin = reference_scale * self.swap_margin_ratio
        if not (swapped_cost + margin < direct_cost and swapped_cost < direct_cost * self.swap_ratio):
            return False

        pose.keypoints[left_index], pose.keypoints[right_index] = pose.keypoints[right_index], pose.keypoints[left_index]
        return True

    def _choose_duplicate_to_reject(self, pose: Pose, left_index: int, right_index: int) -> int | None:
        previous_left = self._previous_positions[left_index]
        previous_right = self._previous_positions[right_index]
        if previous_left is None or previous_right is None:
            return None

        left = pose[left_index]
        right = pose[right_index]
        left_motion = hypot(left.x - previous_left[0], left.y - previous_left[1])
        right_motion = hypot(right.x - previous_right[0], right.y - previous_right[1])
        if left_motion + 1e-4 < right_motion * 0.80:
            return right_index
        if right_motion + 1e-4 < left_motion * 0.80:
            return left_index
        if left.confidence < right.confidence * 0.85:
            return left_index
        if right.confidence < left.confidence * 0.85:
            return right_index
        if left_motion < right_motion:
            return right_index
        if right_motion < left_motion:
            return left_index
        return None

    def _pair_width(self, pose: Pose, left_index: int, right_index: int) -> float | None:
        left = pose[left_index]
        right = pose[right_index]
        if not (left.is_valid(self.detection_threshold) and right.is_valid(self.detection_threshold)):
            return None
        width = self._distance(left, right)
        return width if width > 1e-5 else None

    def _update_history(self, pose: Pose) -> None:
        for index, point in enumerate(pose.keypoints):
            if point.is_valid(self.detection_threshold):
                self._previous_positions[index] = (point.x, point.y)

    @staticmethod
    def _distance(first: Keypoint, second: Keypoint) -> float:
        return hypot(second.x - first.x, second.y - first.y)
