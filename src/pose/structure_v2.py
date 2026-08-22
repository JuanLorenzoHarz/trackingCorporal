"""Hierarquia corporal usada pela PoseNet V2."""

from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS

PARENT_BY_KEYPOINT = (
    -1,
    int(BodyKeypoint.NOSE), int(BodyKeypoint.NOSE),
    int(BodyKeypoint.LEFT_EYE), int(BodyKeypoint.RIGHT_EYE),
    -1, -1,
    int(BodyKeypoint.LEFT_SHOULDER), int(BodyKeypoint.RIGHT_SHOULDER),
    int(BodyKeypoint.LEFT_ELBOW), int(BodyKeypoint.RIGHT_ELBOW),
    -1, -1,
    int(BodyKeypoint.LEFT_HIP), int(BodyKeypoint.RIGHT_HIP),
    int(BodyKeypoint.LEFT_KNEE), int(BodyKeypoint.RIGHT_KNEE),
)

assert len(PARENT_BY_KEYPOINT) == NUM_KEYPOINTS

DECODE_ORDER = (
    int(BodyKeypoint.NOSE),
    int(BodyKeypoint.LEFT_SHOULDER), int(BodyKeypoint.RIGHT_SHOULDER),
    int(BodyKeypoint.LEFT_HIP), int(BodyKeypoint.RIGHT_HIP),
    int(BodyKeypoint.LEFT_EYE), int(BodyKeypoint.RIGHT_EYE),
    int(BodyKeypoint.LEFT_EAR), int(BodyKeypoint.RIGHT_EAR),
    int(BodyKeypoint.LEFT_ELBOW), int(BodyKeypoint.RIGHT_ELBOW),
    int(BodyKeypoint.LEFT_KNEE), int(BodyKeypoint.RIGHT_KNEE),
    int(BodyKeypoint.LEFT_WRIST), int(BodyKeypoint.RIGHT_WRIST),
    int(BodyKeypoint.LEFT_ANKLE), int(BodyKeypoint.RIGHT_ANKLE),
)

BILATERAL_PAIRS = (
    (int(BodyKeypoint.LEFT_SHOULDER), int(BodyKeypoint.RIGHT_SHOULDER)),
    (int(BodyKeypoint.LEFT_ELBOW), int(BodyKeypoint.RIGHT_ELBOW)),
    (int(BodyKeypoint.LEFT_WRIST), int(BodyKeypoint.RIGHT_WRIST)),
    (int(BodyKeypoint.LEFT_HIP), int(BodyKeypoint.RIGHT_HIP)),
    (int(BodyKeypoint.LEFT_KNEE), int(BodyKeypoint.RIGHT_KNEE)),
    (int(BodyKeypoint.LEFT_ANKLE), int(BodyKeypoint.RIGHT_ANKLE)),
)

EXTREMITY_INDICES = frozenset((
    int(BodyKeypoint.LEFT_WRIST), int(BodyKeypoint.RIGHT_WRIST),
    int(BodyKeypoint.LEFT_ANKLE), int(BodyKeypoint.RIGHT_ANKLE),
))
