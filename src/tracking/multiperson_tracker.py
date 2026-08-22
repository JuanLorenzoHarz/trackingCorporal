"""Tracking temporal independente para múltiplas pessoas."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from src.core.types import Pose
from src.pose.keypoints import BodyKeypoint
from src.tracking.smoothing import ExponentialPoseSmoother
from src.tracking.temporal_tracker import TemporalPoseTracker


TORSO_INDICES = (
    int(BodyKeypoint.LEFT_SHOULDER),
    int(BodyKeypoint.RIGHT_SHOULDER),
    int(BodyKeypoint.LEFT_HIP),
    int(BodyKeypoint.RIGHT_HIP),
)


@dataclass(slots=True)
class _PersonTrack:
    track_id: int
    center_x: float
    center_y: float
    missing_frames: int
    tracker: TemporalPoseTracker
    smoother: ExponentialPoseSmoother


@dataclass(frozen=True, slots=True)
class TrackedPerson:
    track_id: int
    pose: Pose


class MultiPersonTemporalTracker:
    """Associa poses por centro e mantém histórico separado por pessoa."""

    def __init__(
        self,
        detection_threshold: float = 0.12,
        match_distance: float = 0.22,
        stale_frames: int = 5,
        prediction_frames: int = 6,
        prediction_decay: float = 0.80,
        anatomy_weight: float = 0.55,
        smoothing_alpha: float = 0.68,
    ) -> None:
        self.detection_threshold = detection_threshold
        self.match_distance = match_distance
        self.stale_frames = stale_frames
        self.prediction_frames = prediction_frames
        self.prediction_decay = prediction_decay
        self.anatomy_weight = anatomy_weight
        self.smoothing_alpha = smoothing_alpha
        self._tracks: dict[int, _PersonTrack] = {}
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def update(self, poses: list[Pose]) -> list[TrackedPerson]:
        detections: list[tuple[Pose, tuple[float, float]]] = []
        for pose in poses:
            center = self._pose_center(pose)
            if center is not None:
                detections.append((pose, center))

        # Todas as combinações possíveis, menor distância primeiro.
        candidates: list[tuple[float, int, int]] = []
        track_ids = list(self._tracks)
        for detection_index, (_, center) in enumerate(detections):
            for track_id in track_ids:
                track = self._tracks[track_id]
                distance = hypot(center[0] - track.center_x, center[1] - track.center_y)
                if distance <= self.match_distance:
                    candidates.append((distance, detection_index, track_id))
        candidates.sort(key=lambda item: item[0])

        matched_detections: set[int] = set()
        matched_tracks: set[int] = set()
        assignment: dict[int, int] = {}

        for _, detection_index, track_id in candidates:
            if detection_index in matched_detections or track_id in matched_tracks:
                continue
            matched_detections.add(detection_index)
            matched_tracks.add(track_id)
            assignment[detection_index] = track_id

        # Detecções sem par criam novo histórico.
        for detection_index in range(len(detections)):
            if detection_index in assignment:
                continue
            track = self._new_track(*detections[detection_index][1])
            self._tracks[track.track_id] = track
            assignment[detection_index] = track.track_id
            matched_tracks.add(track.track_id)

        output: list[TrackedPerson] = []
        for detection_index, (pose, center) in enumerate(detections):
            track = self._tracks[assignment[detection_index]]
            track.center_x, track.center_y = center
            track.missing_frames = 0
            temporally_refined = track.tracker.update(pose)
            smoothed = track.smoother.update(temporally_refined)
            output.append(TrackedPerson(track.track_id, smoothed))

        # Não desenhamos tracks sem detecção: isso evita pessoa fantasma. O estado
        # é mantido só por poucos frames para permitir reentrada/oclusão curta.
        for track_id in list(self._tracks):
            if track_id in matched_tracks:
                continue
            track = self._tracks[track_id]
            track.missing_frames += 1
            if track.missing_frames > self.stale_frames:
                del self._tracks[track_id]

        return output

    def _new_track(self, center_x: float, center_y: float) -> _PersonTrack:
        track_id = self._next_track_id
        self._next_track_id += 1
        return _PersonTrack(
            track_id=track_id,
            center_x=center_x,
            center_y=center_y,
            missing_frames=0,
            tracker=TemporalPoseTracker(
                detection_threshold=self.detection_threshold,
                max_missing_frames=self.prediction_frames,
                confidence_decay=self.prediction_decay,
                anatomy_weight=self.anatomy_weight,
            ),
            smoother=ExponentialPoseSmoother(alpha=self.smoothing_alpha),
        )

    def _pose_center(self, pose: Pose) -> tuple[float, float] | None:
        torso = [
            pose[index]
            for index in TORSO_INDICES
            if pose[index].is_valid(self.detection_threshold)
        ]
        points = torso
        if len(points) < 2:
            points = [
                point
                for point in pose.keypoints
                if point.is_valid(self.detection_threshold)
            ]
        if not points:
            return None
        return (
            sum(point.x for point in points) / len(points),
            sum(point.y for point in points) / len(points),
        )
