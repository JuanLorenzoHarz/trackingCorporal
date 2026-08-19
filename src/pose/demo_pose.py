"""Pose artificial usada para validar o pipeline antes da CNN existir."""

from src.core.types import Keypoint, Pose


def build_demo_pose() -> Pose:
    """Cria uma pose fixa simples com os 17 keypoints do projeto."""
    coordinates = (
        (0.50, 0.16),  # NOSE
        (0.47, 0.14),  # LEFT_EYE
        (0.53, 0.14),  # RIGHT_EYE
        (0.43, 0.16),  # LEFT_EAR
        (0.57, 0.16),  # RIGHT_EAR
        (0.39, 0.30),  # LEFT_SHOULDER
        (0.61, 0.30),  # RIGHT_SHOULDER
        (0.32, 0.45),  # LEFT_ELBOW
        (0.68, 0.45),  # RIGHT_ELBOW
        (0.27, 0.60),  # LEFT_WRIST
        (0.73, 0.60),  # RIGHT_WRIST
        (0.43, 0.57),  # LEFT_HIP
        (0.57, 0.57),  # RIGHT_HIP
        (0.41, 0.75),  # LEFT_KNEE
        (0.59, 0.75),  # RIGHT_KNEE
        (0.39, 0.94),  # LEFT_ANKLE
        (0.61, 0.94),  # RIGHT_ANKLE
    )

    return Pose(
        [Keypoint(x=x, y=y, confidence=1.0) for x, y in coordinates]
    )
