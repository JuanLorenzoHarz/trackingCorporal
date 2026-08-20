"""Tracking temporal e previsão curta de keypoints temporariamente ocultos.

O tracker combina duas fontes simples de informação:
- movimento recente de cada articulação;
- relação geométrica aprendida entre articulações conectadas do esqueleto.

A intenção não é "inventar" uma pose indefinidamente. Pontos previstos perdem
confiança a cada frame e expiram após alguns frames sem observação da CNN.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.types import Keypoint, Pose
from src.pose.keypoints import NUM_KEYPOINTS, SKELETON_CONNECTIONS


@dataclass(slots=True)
class _PointState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.0
    missing_frames: int = 0
    initialized: bool = False


class TemporalPoseTracker:
    """Mantém continuidade da pose quando keypoints somem por poucos frames."""

    def __init__(
        self,
        detection_threshold: float = 0.15,
        max_missing_frames: int = 8,
        confidence_decay: float = 0.82,
        velocity_alpha: float = 0.55,
        velocity_decay: float = 0.85,
        anatomy_weight: float = 0.60,
        anatomy_alpha: float = 0.20,
    ) -> None:
        if not 0.0 <= detection_threshold <= 1.0:
            raise ValueError("detection_threshold deve estar entre 0 e 1.")
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames não pode ser negativo.")
        if not 0.0 < confidence_decay <= 1.0:
            raise ValueError("confidence_decay deve estar entre 0 e 1.")
        if not 0.0 <= velocity_alpha <= 1.0:
            raise ValueError("velocity_alpha deve estar entre 0 e 1.")
        if not 0.0 <= velocity_decay <= 1.0:
            raise ValueError("velocity_decay deve estar entre 0 e 1.")
        if not 0.0 <= anatomy_weight <= 1.0:
            raise ValueError("anatomy_weight deve estar entre 0 e 1.")
        if not 0.0 < anatomy_alpha <= 1.0:
            raise ValueError("anatomy_alpha deve estar entre 0 e 1.")

        self.detection_threshold = detection_threshold
        self.max_missing_frames = max_missing_frames
        self.confidence_decay = confidence_decay
        self.velocity_alpha = velocity_alpha
        self.velocity_decay = velocity_decay
        self.anatomy_weight = anatomy_weight
        self.anatomy_alpha = anatomy_alpha

        self._states = [_PointState() for _ in range(NUM_KEYPOINTS)]
        self._offsets: dict[tuple[int, int], tuple[float, float]] = {}
        self._neighbors: list[list[int]] = [[] for _ in range(NUM_KEYPOINTS)]
        self.predicted_indices: set[int] = set()

        for start, end in SKELETON_CONNECTIONS:
            start_index = int(start)
            end_index = int(end)
            self._neighbors[start_index].append(end_index)
            self._neighbors[end_index].append(start_index)

    @property
    def predicted_count(self) -> int:
        """Quantidade de articulações atualmente mantidas por previsão."""
        return len(self.predicted_indices)

    def reset(self) -> None:
        """Descarta todo o histórico temporal e geométrico."""
        self._states = [_PointState() for _ in range(NUM_KEYPOINTS)]
        self._offsets.clear()
        self.predicted_indices.clear()

    def update(self, pose: Pose) -> Pose:
        """Combina uma pose nova com o histórico e retorna a pose estabilizada.

        Um ponto com confiança suficiente é tratado como observação real da CNN.
        Quando ele some, a posição prevista combina movimento recente e relações
        geométricas previamente observadas com articulações vizinhas ainda visíveis.
        """
        if len(pose) != NUM_KEYPOINTS:
            raise ValueError(
                f"TemporalPoseTracker espera {NUM_KEYPOINTS} keypoints; recebeu {len(pose)}."
            )

        detected = [
            point.is_valid(self.detection_threshold)
            for point in pose.keypoints
        ]

        self._update_anatomical_offsets(pose, detected)

        output: list[Keypoint] = []
        predicted_indices: set[int] = set()

        for index, point in enumerate(pose.keypoints):
            state = self._states[index]

            if detected[index]:
                was_missing = state.missing_frames > 0

                if state.initialized and not was_missing:
                    measured_vx = point.x - state.x
                    measured_vy = point.y - state.y
                    state.vx = (
                        self.velocity_alpha * measured_vx
                        + (1.0 - self.velocity_alpha) * state.vx
                    )
                    state.vy = (
                        self.velocity_alpha * measured_vy
                        + (1.0 - self.velocity_alpha) * state.vy
                    )
                else:
                    # Ao reencontrar um ponto oculto, evitamos transformar o salto
                    # entre previsão e observação em uma velocidade artificial enorme.
                    state.vx = 0.0
                    state.vy = 0.0

                state.x = point.x
                state.y = point.y
                state.confidence = point.confidence
                state.missing_frames = 0
                state.initialized = True
                output.append(Keypoint(point.x, point.y, point.confidence))
                continue

            if not state.initialized:
                output.append(Keypoint(point.x, point.y, 0.0))
                continue

            state.missing_frames += 1
            if state.missing_frames > self.max_missing_frames:
                state.confidence = 0.0
                state.vx = 0.0
                state.vy = 0.0
                output.append(Keypoint(state.x, state.y, 0.0))
                continue

            temporal_x = state.x + state.vx
            temporal_y = state.y + state.vy
            anatomy_candidates = self._anatomical_candidates(
                point_index=index,
                pose=pose,
                detected=detected,
            )

            predicted_x = temporal_x
            predicted_y = temporal_y

            if anatomy_candidates:
                anatomy_x = sum(candidate[0] for candidate in anatomy_candidates) / len(
                    anatomy_candidates
                )
                anatomy_y = sum(candidate[1] for candidate in anatomy_candidates) / len(
                    anatomy_candidates
                )
                predicted_x = (
                    (1.0 - self.anatomy_weight) * temporal_x
                    + self.anatomy_weight * anatomy_x
                )
                predicted_y = (
                    (1.0 - self.anatomy_weight) * temporal_y
                    + self.anatomy_weight * anatomy_y
                )

            predicted_x = min(1.0, max(0.0, predicted_x))
            predicted_y = min(1.0, max(0.0, predicted_y))
            predicted_confidence = state.confidence * self.confidence_decay

            state.x = predicted_x
            state.y = predicted_y
            state.vx *= self.velocity_decay
            state.vy *= self.velocity_decay
            state.confidence = predicted_confidence

            predicted_indices.add(index)
            output.append(
                Keypoint(
                    x=predicted_x,
                    y=predicted_y,
                    confidence=predicted_confidence,
                )
            )

        self.predicted_indices = predicted_indices
        return Pose(output)

    def _update_anatomical_offsets(self, pose: Pose, detected: list[bool]) -> None:
        """Aprende vetores relativos entre articulações conectadas visíveis."""
        for start, end in SKELETON_CONNECTIONS:
            start_index = int(start)
            end_index = int(end)

            if not detected[start_index] or not detected[end_index]:
                continue

            start_point = pose[start_index]
            end_point = pose[end_index]
            measured_dx = end_point.x - start_point.x
            measured_dy = end_point.y - start_point.y

            self._store_offset(
                start_index,
                end_index,
                measured_dx,
                measured_dy,
            )
            self._store_offset(
                end_index,
                start_index,
                -measured_dx,
                -measured_dy,
            )

    def _store_offset(
        self,
        start_index: int,
        end_index: int,
        measured_dx: float,
        measured_dy: float,
    ) -> None:
        key = (start_index, end_index)
        previous = self._offsets.get(key)

        if previous is None:
            self._offsets[key] = (measured_dx, measured_dy)
            return

        self._offsets[key] = (
            (1.0 - self.anatomy_alpha) * previous[0]
            + self.anatomy_alpha * measured_dx,
            (1.0 - self.anatomy_alpha) * previous[1]
            + self.anatomy_alpha * measured_dy,
        )

    def _anatomical_candidates(
        self,
        point_index: int,
        pose: Pose,
        detected: list[bool],
    ) -> list[tuple[float, float]]:
        """Infere um ponto oculto a partir de vizinhos corporais ainda visíveis."""
        candidates: list[tuple[float, float]] = []

        for neighbor_index in self._neighbors[point_index]:
            if not detected[neighbor_index]:
                continue

            offset = self._offsets.get((point_index, neighbor_index))
            if offset is None:
                continue

            neighbor = pose[neighbor_index]
            # offset representa neighbor - point; portanto point = neighbor - offset.
            candidates.append(
                (
                    neighbor.x - offset[0],
                    neighbor.y - offset[1],
                )
            )

        return candidates
