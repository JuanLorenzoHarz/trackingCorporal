"""Suavização temporal simples para reduzir tremores nos keypoints."""

from __future__ import annotations

from src.core.types import Keypoint, Pose
from src.pose.keypoints import NUM_KEYPOINTS


class ExponentialPoseSmoother:
    """Aplica média móvel exponencial às coordenadas de cada articulação."""

    def __init__(self, alpha: float = 0.65) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha deve estar entre 0 e 1.")

        self.alpha = alpha
        self._previous: list[Keypoint | None] = [None] * NUM_KEYPOINTS

    def reset(self) -> None:
        self._previous = [None] * NUM_KEYPOINTS

    def update(self, pose: Pose) -> Pose:
        if len(pose) != NUM_KEYPOINTS:
            raise ValueError(
                f"ExponentialPoseSmoother espera {NUM_KEYPOINTS} keypoints; recebeu {len(pose)}."
            )

        output: list[Keypoint] = []

        for index, point in enumerate(pose.keypoints):
            previous = self._previous[index]

            if point.confidence <= 0.0:
                self._previous[index] = None
                output.append(Keypoint(point.x, point.y, point.confidence))
                continue

            if previous is None:
                smoothed = Keypoint(point.x, point.y, point.confidence)
            else:
                smoothed = Keypoint(
                    x=self.alpha * point.x + (1.0 - self.alpha) * previous.x,
                    y=self.alpha * point.y + (1.0 - self.alpha) * previous.y,
                    confidence=point.confidence,
                )

            self._previous[index] = smoothed
            output.append(smoothed)

        return Pose(output)
