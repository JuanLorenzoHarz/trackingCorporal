"""Definição dos keypoints e das conexões do esqueleto corporal."""

from enum import IntEnum


class BodyKeypoint(IntEnum):
    """Índices dos 17 keypoints corporais usados pelo projeto."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


NUM_KEYPOINTS = len(BodyKeypoint)


# Cada par representa uma linha que compõe o esqueleto visual.
SKELETON_CONNECTIONS: tuple[tuple[BodyKeypoint, BodyKeypoint], ...] = (
    # Cabeça
    (BodyKeypoint.LEFT_EAR, BodyKeypoint.LEFT_EYE),
    (BodyKeypoint.LEFT_EYE, BodyKeypoint.NOSE),
    (BodyKeypoint.NOSE, BodyKeypoint.RIGHT_EYE),
    (BodyKeypoint.RIGHT_EYE, BodyKeypoint.RIGHT_EAR),

    # Ombros
    (BodyKeypoint.LEFT_SHOULDER, BodyKeypoint.RIGHT_SHOULDER),

    # Braço esquerdo
    (BodyKeypoint.LEFT_SHOULDER, BodyKeypoint.LEFT_ELBOW),
    (BodyKeypoint.LEFT_ELBOW, BodyKeypoint.LEFT_WRIST),

    # Braço direito
    (BodyKeypoint.RIGHT_SHOULDER, BodyKeypoint.RIGHT_ELBOW),
    (BodyKeypoint.RIGHT_ELBOW, BodyKeypoint.RIGHT_WRIST),

    # Tronco
    (BodyKeypoint.LEFT_SHOULDER, BodyKeypoint.LEFT_HIP),
    (BodyKeypoint.RIGHT_SHOULDER, BodyKeypoint.RIGHT_HIP),
    (BodyKeypoint.LEFT_HIP, BodyKeypoint.RIGHT_HIP),

    # Perna esquerda
    (BodyKeypoint.LEFT_HIP, BodyKeypoint.LEFT_KNEE),
    (BodyKeypoint.LEFT_KNEE, BodyKeypoint.LEFT_ANKLE),

    # Perna direita
    (BodyKeypoint.RIGHT_HIP, BodyKeypoint.RIGHT_KNEE),
    (BodyKeypoint.RIGHT_KNEE, BodyKeypoint.RIGHT_ANKLE),
)
