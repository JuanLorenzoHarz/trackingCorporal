"""Métrica de plausibilidade anatômica e temporal para poses 2D.

A CNN continua sendo a fonte primária dos keypoints. Este módulo não tenta
transformar pose 2D em anatomia 3D; ele apenas detecta resultados muito
inconsistentes com o próprio corpo observado nos frames anteriores.

Os sinais usados são:
- consistência do comprimento projetado dos segmentos corporais;
- simetria aproximada entre os lados esquerdo e direito;
- continuidade temporal dos keypoints.

Uma observação considerada muito improvável tem sua confiança zerada antes de
entrar no tracker temporal. Assim, o tracker pode preferir uma previsão baseada
no histórico em vez de aceitar um salto absurdo da CNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot
from statistics import mean, median

from src.core.types import Keypoint, Pose
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS


Segment = tuple[int, int]


def _segment(start: BodyKeypoint, end: BodyKeypoint) -> Segment:
    return int(start), int(end)


# Segmentos em que comprimento e continuidade são úteis como restrição suave.
STRUCTURAL_SEGMENTS: tuple[Segment, ...] = (
    _segment(BodyKeypoint.LEFT_SHOULDER, BodyKeypoint.RIGHT_SHOULDER),
    _segment(BodyKeypoint.LEFT_SHOULDER, BodyKeypoint.LEFT_ELBOW),
    _segment(BodyKeypoint.LEFT_ELBOW, BodyKeypoint.LEFT_WRIST),
    _segment(BodyKeypoint.RIGHT_SHOULDER, BodyKeypoint.RIGHT_ELBOW),
    _segment(BodyKeypoint.RIGHT_ELBOW, BodyKeypoint.RIGHT_WRIST),
    _segment(BodyKeypoint.LEFT_SHOULDER, BodyKeypoint.LEFT_HIP),
    _segment(BodyKeypoint.RIGHT_SHOULDER, BodyKeypoint.RIGHT_HIP),
    _segment(BodyKeypoint.LEFT_HIP, BodyKeypoint.RIGHT_HIP),
    _segment(BodyKeypoint.LEFT_HIP, BodyKeypoint.LEFT_KNEE),
    _segment(BodyKeypoint.LEFT_KNEE, BodyKeypoint.LEFT_ANKLE),
    _segment(BodyKeypoint.RIGHT_HIP, BodyKeypoint.RIGHT_KNEE),
    _segment(BodyKeypoint.RIGHT_KNEE, BodyKeypoint.RIGHT_ANKLE),
)


LEG_SEGMENTS: tuple[Segment, ...] = (
    _segment(BodyKeypoint.LEFT_HIP, BodyKeypoint.LEFT_KNEE),
    _segment(BodyKeypoint.LEFT_KNEE, BodyKeypoint.LEFT_ANKLE),
    _segment(BodyKeypoint.RIGHT_HIP, BodyKeypoint.RIGHT_KNEE),
    _segment(BodyKeypoint.RIGHT_KNEE, BodyKeypoint.RIGHT_ANKLE),
)


SYMMETRY_PAIRS: tuple[tuple[Segment, Segment], ...] = (
    (
        _segment(BodyKeypoint.LEFT_SHOULDER, BodyKeypoint.LEFT_ELBOW),
        _segment(BodyKeypoint.RIGHT_SHOULDER, BodyKeypoint.RIGHT_ELBOW),
    ),
    (
        _segment(BodyKeypoint.LEFT_ELBOW, BodyKeypoint.LEFT_WRIST),
        _segment(BodyKeypoint.RIGHT_ELBOW, BodyKeypoint.RIGHT_WRIST),
    ),
    (
        _segment(BodyKeypoint.LEFT_SHOULDER, BodyKeypoint.LEFT_HIP),
        _segment(BodyKeypoint.RIGHT_SHOULDER, BodyKeypoint.RIGHT_HIP),
    ),
    (
        _segment(BodyKeypoint.LEFT_HIP, BodyKeypoint.LEFT_KNEE),
        _segment(BodyKeypoint.RIGHT_HIP, BodyKeypoint.RIGHT_KNEE),
    ),
    (
        _segment(BodyKeypoint.LEFT_KNEE, BodyKeypoint.LEFT_ANKLE),
        _segment(BodyKeypoint.RIGHT_KNEE, BodyKeypoint.RIGHT_ANKLE),
    ),
)


@dataclass(slots=True)
class _LengthBaseline:
    value: float
    observations: int = 1


@dataclass(frozen=True, slots=True)
class PlausibilityReport:
    """Resumo numérico da plausibilidade da pose recebida."""

    score: float
    bone_score: float
    leg_score: float
    symmetry_score: float
    temporal_score: float
    suspicious_indices: frozenset[int]
    calibrated_segments: int

    @property
    def percentage(self) -> float:
        return self.score * 100.0

    @property
    def leg_percentage(self) -> float:
        return self.leg_score * 100.0


class PosePlausibilityEvaluator:
    """Aprende proporções da pessoa e rejeita observações muito inconsistentes."""

    def __init__(
        self,
        detection_threshold: float = 0.15,
        min_baseline_observations: int = 6,
        baseline_alpha: float = 0.08,
        bone_tolerance: float = 0.35,
        symmetry_tolerance: float = 0.50,
        temporal_free_motion: float = 1.25,
        temporal_tolerance: float = 1.00,
        reject_threshold: float = 0.16,
        baseline_update_max_error: float = 0.40,
    ) -> None:
        if not 0.0 <= detection_threshold <= 1.0:
            raise ValueError("detection_threshold deve estar entre 0 e 1.")
        if min_baseline_observations < 1:
            raise ValueError("min_baseline_observations deve ser pelo menos 1.")
        if not 0.0 < baseline_alpha <= 1.0:
            raise ValueError("baseline_alpha deve estar entre 0 e 1.")
        if bone_tolerance <= 0.0 or symmetry_tolerance <= 0.0:
            raise ValueError("As tolerâncias anatômicas devem ser positivas.")
        if temporal_free_motion < 0.0 or temporal_tolerance <= 0.0:
            raise ValueError("As tolerâncias temporais são inválidas.")
        if not 0.0 < reject_threshold < 1.0:
            raise ValueError("reject_threshold deve estar entre 0 e 1.")

        self.detection_threshold = detection_threshold
        self.min_baseline_observations = min_baseline_observations
        self.baseline_alpha = baseline_alpha
        self.bone_tolerance = bone_tolerance
        self.symmetry_tolerance = symmetry_tolerance
        self.temporal_free_motion = temporal_free_motion
        self.temporal_tolerance = temporal_tolerance
        self.reject_threshold = reject_threshold
        self.baseline_update_max_error = baseline_update_max_error

        self._baselines: dict[Segment, _LengthBaseline] = {}
        self._previous_positions: list[tuple[float, float] | None] = [
            None for _ in range(NUM_KEYPOINTS)
        ]

    def reset(self) -> None:
        """Apaga proporções aprendidas e histórico temporal."""
        self._baselines.clear()
        self._previous_positions = [None for _ in range(NUM_KEYPOINTS)]

    def evaluate_and_filter(self, pose: Pose) -> tuple[Pose, PlausibilityReport]:
        """Pontua a pose e zera a confiança de keypoints fortemente suspeitos."""
        if len(pose) != NUM_KEYPOINTS:
            raise ValueError(
                f"PosePlausibilityEvaluator espera {NUM_KEYPOINTS} keypoints; "
                f"recebeu {len(pose)}."
            )

        valid = [point.is_valid(self.detection_threshold) for point in pose.keypoints]
        scale = self._estimate_body_scale(pose, valid)
        temporal_scores = self._temporal_scores(pose, valid, scale)

        segment_scores: dict[Segment, float] = {}
        bad_segments_by_point: list[list[tuple[Segment, float]]] = [
            [] for _ in range(NUM_KEYPOINTS)
        ]

        for segment in STRUCTURAL_SEGMENTS:
            start, end = segment
            if not valid[start] or not valid[end]:
                continue

            length = self._distance(pose[start], pose[end])
            baseline = self._baselines.get(segment)
            score = 1.0

            if (
                baseline is not None
                and baseline.observations >= self.min_baseline_observations
                and baseline.value > 1e-6
            ):
                relative_error = abs(length - baseline.value) / baseline.value
                score = self._soft_score(relative_error, self.bone_tolerance)

            segment_scores[segment] = score
            if score < self.reject_threshold:
                bad_segments_by_point[start].append((segment, score))
                bad_segments_by_point[end].append((segment, score))

        symmetry_scores = self._symmetry_scores(pose, valid)
        suspicious: set[int] = set()

        # Saltos temporais extremos são suspeitos por si só.
        for index, temporal_score in enumerate(temporal_scores):
            if valid[index] and temporal_score < self.reject_threshold:
                suspicious.add(index)

        # Um ponto central (ex.: joelho) costuma quebrar dois ossos ao mesmo tempo.
        for index, bad_segments in enumerate(bad_segments_by_point):
            if len(bad_segments) >= 2:
                suspicious.add(index)

        # Para extremidades com apenas um segmento (ex.: tornozelo), usamos o
        # histórico temporal e a confiança relativa para decidir qual ponta é
        # mais provável de estar errada.
        for segment, segment_score in segment_scores.items():
            if segment_score >= self.reject_threshold:
                continue

            start, end = segment
            start_temporal = temporal_scores[start]
            end_temporal = temporal_scores[end]

            if start_temporal + 0.20 < end_temporal:
                suspicious.add(start)
            elif end_temporal + 0.20 < start_temporal:
                suspicious.add(end)
            else:
                start_confidence = pose[start].confidence
                end_confidence = pose[end].confidence
                if start_confidence < end_confidence * 0.60:
                    suspicious.add(start)
                elif end_confidence < start_confidence * 0.60:
                    suspicious.add(end)

        bone_score = mean(segment_scores.values()) if segment_scores else 1.0
        leg_values = [
            segment_scores[segment]
            for segment in LEG_SEGMENTS
            if segment in segment_scores
        ]
        leg_score = mean(leg_values) if leg_values else 1.0
        symmetry_score = mean(symmetry_scores) if symmetry_scores else 1.0
        valid_temporal_scores = [
            temporal_scores[index]
            for index in range(NUM_KEYPOINTS)
            if valid[index] and self._previous_positions[index] is not None
        ]
        temporal_score = (
            mean(valid_temporal_scores) if valid_temporal_scores else 1.0
        )

        overall = (
            0.55 * bone_score
            + 0.15 * symmetry_score
            + 0.30 * temporal_score
        )
        overall = min(1.0, max(0.0, overall))

        filtered_points: list[Keypoint] = []
        for index, point in enumerate(pose.keypoints):
            if index in suspicious:
                filtered_points.append(Keypoint(point.x, point.y, 0.0))
            else:
                filtered_points.append(Keypoint(point.x, point.y, point.confidence))

        filtered_pose = Pose(filtered_points)
        self._update_baselines(filtered_pose)
        self._update_previous_positions(filtered_pose)

        calibrated_segments = sum(
            baseline.observations >= self.min_baseline_observations
            for baseline in self._baselines.values()
        )

        report = PlausibilityReport(
            score=overall,
            bone_score=bone_score,
            leg_score=leg_score,
            symmetry_score=symmetry_score,
            temporal_score=temporal_score,
            suspicious_indices=frozenset(suspicious),
            calibrated_segments=calibrated_segments,
        )
        return filtered_pose, report

    def _estimate_body_scale(self, pose: Pose, valid: list[bool]) -> float:
        """Obtém uma escala relativa para tornar movimento independente da distância."""
        candidates: list[float] = []

        for segment in STRUCTURAL_SEGMENTS:
            baseline = self._baselines.get(segment)
            if (
                baseline is not None
                and baseline.observations >= self.min_baseline_observations
            ):
                candidates.append(baseline.value)
                continue

            start, end = segment
            if valid[start] and valid[end]:
                length = self._distance(pose[start], pose[end])
                if length > 1e-6:
                    candidates.append(length)

        return median(candidates) if candidates else 0.20

    def _temporal_scores(
        self,
        pose: Pose,
        valid: list[bool],
        body_scale: float,
    ) -> list[float]:
        scores = [1.0 for _ in range(NUM_KEYPOINTS)]
        safe_scale = max(body_scale, 1e-4)

        for index, point in enumerate(pose.keypoints):
            previous = self._previous_positions[index]
            if not valid[index] or previous is None:
                continue

            displacement = hypot(point.x - previous[0], point.y - previous[1])
            relative_motion = displacement / safe_scale

            if relative_motion <= self.temporal_free_motion:
                scores[index] = 1.0
                continue

            excess = relative_motion - self.temporal_free_motion
            scores[index] = self._soft_score(excess, self.temporal_tolerance)

        return scores

    def _symmetry_scores(self, pose: Pose, valid: list[bool]) -> list[float]:
        scores: list[float] = []

        for left_segment, right_segment in SYMMETRY_PAIRS:
            ls, le = left_segment
            rs, re = right_segment
            if not (valid[ls] and valid[le] and valid[rs] and valid[re]):
                continue

            left_length = self._distance(pose[ls], pose[le])
            right_length = self._distance(pose[rs], pose[re])
            average_length = max((left_length + right_length) / 2.0, 1e-6)
            relative_difference = abs(left_length - right_length) / average_length
            scores.append(
                self._soft_score(relative_difference, self.symmetry_tolerance)
            )

        return scores

    def _update_baselines(self, pose: Pose) -> None:
        valid = [point.is_valid(self.detection_threshold) for point in pose.keypoints]

        for segment in STRUCTURAL_SEGMENTS:
            start, end = segment
            if not valid[start] or not valid[end]:
                continue

            length = self._distance(pose[start], pose[end])
            if length <= 1e-6:
                continue

            baseline = self._baselines.get(segment)
            if baseline is None:
                self._baselines[segment] = _LengthBaseline(length)
                continue

            relative_error = abs(length - baseline.value) / max(baseline.value, 1e-6)
            if (
                baseline.observations >= self.min_baseline_observations
                and relative_error > self.baseline_update_max_error
            ):
                continue

            if baseline.observations < self.min_baseline_observations:
                alpha = 1.0 / (baseline.observations + 1.0)
            else:
                alpha = self.baseline_alpha

            baseline.value = (1.0 - alpha) * baseline.value + alpha * length
            baseline.observations += 1

    def _update_previous_positions(self, pose: Pose) -> None:
        for index, point in enumerate(pose.keypoints):
            if point.is_valid(self.detection_threshold):
                self._previous_positions[index] = (point.x, point.y)

    @staticmethod
    def _distance(first: Keypoint, second: Keypoint) -> float:
        return hypot(second.x - first.x, second.y - first.y)

    @staticmethod
    def _soft_score(error: float, tolerance: float) -> float:
        """Converte erro relativo em score suave 1 -> 0 sem corte rígido."""
        normalized = error / max(tolerance, 1e-6)
        return exp(-0.5 * normalized * normalized)
