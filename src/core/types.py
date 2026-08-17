"""Tipos de dados compartilhados pelo pipeline."""

from dataclasses import dataclass
from enum import Enum


@dataclass(slots=True)
class Keypoint:
    """Um ponto corporal em coordenadas normalizadas (0.0 a 1.0)."""

    x: float
    y: float
    confidence: float

    def is_valid(self, threshold: float = 0.5) -> bool:
        """Retorna True quando o ponto está dentro da imagem e é confiável."""
        return (
            0.0 <= self.x <= 1.0
            and 0.0 <= self.y <= 1.0
            and self.confidence >= threshold
        )


@dataclass(slots=True)
class Pose:
    """Conjunto ordenado de keypoints de uma pessoa."""

    keypoints: list[Keypoint]

    def __len__(self) -> int:
        return len(self.keypoints)

    def __getitem__(self, index: int) -> Keypoint:
        return self.keypoints[index]


class HandState(Enum):
    """Estados simples que serão usados futuramente para cada mão."""

    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"
