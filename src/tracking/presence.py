"""Gate simples de presença corporal para reduzir poses fantasma."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from src.core.types import Pose
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS


TORSO_INDICES = (
    int(BodyKeypoint.LEFT_SHOULDER),
    int(BodyKeypoint.RIGHT_SHOULDER),
    int(BodyKeypoint.LEFT_HIP),
    int(BodyKeypoint.RIGHT_HIP),
)


@dataclass(frozen=True, slots=True)
class BodyPresenceReport:
    active: bool
    score: float
    valid_keypoints: int
    torso_keypoints: int
    acquired: bool
    lost: bool

    @property
    def percentage(self) -> float:
        return self.score * 100.0


class BodyPresenceGate:
    """Exige evidência consistente de corpo antes de liberar o tracking."""

    def __init__(
        self,
        detection_threshold: float = 0.10,
        minimum_keypoints: int = 5,
        minimum_torso_keypoints: int = 2,
        acquire_frames: int = 3,
        release_frames: int = 6,
        acquire_score: float = 0.68,
        retain_score: float = 0.34,
    ) -> None:
        self.detection_threshold = detection_threshold
        self.minimum_keypoints = minimum_keypoints
        self.minimum_torso_keypoints = minimum_torso_keypoints
        self.acquire_frames = acquire_frames
        self.release_frames = release_frames
        self.acquire_score = acquire_score
        self.retain_score = retain_score
        self._active = False
        self._strong_frames = 0
        self._weak_frames = 0

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False
        self._strong_frames = 0
        self._weak_frames = 0

    def update(self, pose: Pose) -> BodyPresenceReport:
        if len(pose) != NUM_KEYPOINTS:
            raise ValueError(f"BodyPresenceGate espera {NUM_KEYPOINTS} keypoints.")

        valid = [p.is_valid(self.detection_threshold) for p in pose.keypoints]
        valid_count = sum(valid)
        torso_count = sum(valid[index] for index in TORSO_INDICES)
        geometry_score = self._torso_geometry_score(pose, valid)

        count_score = min(1.0, valid_count / max(self.minimum_keypoints, 1))
        torso_score = min(1.0, torso_count / max(self.minimum_torso_keypoints, 1))
        score = 0.35 * count_score + 0.45 * torso_score + 0.20 * geometry_score

        strong = (
            valid_count >= self.minimum_keypoints
            and torso_count >= self.minimum_torso_keypoints
            and score >= self.acquire_score
        )
        retain = score >= self.retain_score and torso_count >= 1

        acquired = False
        lost = False

        if not self._active:
            self._strong_frames = self._strong_frames + 1 if strong else 0
            if self._strong_frames >= self.acquire_frames:
                self._active = True
                self._weak_frames = 0
                acquired = True
        else:
            self._weak_frames = 0 if retain else self._weak_frames + 1
            if self._weak_frames >= self.release_frames:
                self._active = False
                self._strong_frames = 0
                self._weak_frames = 0
                lost = True

        return BodyPresenceReport(
            active=self._active,
            score=max(0.0, min(1.0, score)),
            valid_keypoints=valid_count,
            torso_keypoints=torso_count,
            acquired=acquired,
            lost=lost,
        )

    def _torso_geometry_score(self, pose: Pose, valid: list[bool]) -> float:
        left_shoulder = int(BodyKeypoint.LEFT_SHOULDER)
        right_shoulder = int(BodyKeypoint.RIGHT_SHOULDER)
        left_hip = int(BodyKeypoint.LEFT_HIP)
        right_hip = int(BodyKeypoint.RIGHT_HIP)

        shoulder_points = [pose[i] for i in (left_shoulder, right_shoulder) if valid[i]]
        hip_points = [pose[i] for i in (left_hip, right_hip) if valid[i]]
        if not shoulder_points or not hip_points:
            return 0.0

        sx = sum(p.x for p in shoulder_points) / len(shoulder_points)
        sy = sum(p.y for p in shoulder_points) / len(shoulder_points)
        hx = sum(p.x for p in hip_points) / len(hip_points)
        hy = sum(p.y for p in hip_points) / len(hip_points)
        torso_span = hypot(sx - hx, sy - hy)
        if not 0.04 <= torso_span <= 0.70:
            return 0.0

        width_checks: list[float] = []
        if valid[left_shoulder] and valid[right_shoulder]:
            width_checks.append(
                hypot(
                    pose[left_shoulder].x - pose[right_shoulder].x,
                    pose[left_shoulder].y - pose[right_shoulder].y,
                )
            )
        if valid[left_hip] and valid[right_hip]:
            width_checks.append(
                hypot(
                    pose[left_hip].x - pose[right_hip].x,
                    pose[left_hip].y - pose[right_hip].y,
                )
            )

        if width_checks and any(width < 0.02 or width > 0.60 for width in width_checks):
            return 0.25
        return 1.0
