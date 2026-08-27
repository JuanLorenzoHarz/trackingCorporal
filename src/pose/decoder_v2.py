"""Decoder multi-pessoa da PoseNet V2.

A associação ocorre em duas etapas:
1. cada candidato de keypoint vota no centro da pessoa através de center_offsets;
2. quando existe pai anatômico, parent_offsets mede se o candidato realmente se
   conecta ao ombro/cotovelo/quadril/joelho correto.

A proteção anti-X é compatível com poses de perfil. Além disso, centros múltiplos
que produzam praticamente a mesma pose são fundidos antes do tracking temporal,
evantando o melhor keypoint de cada hipótese em vez de criar dois IDs para a
mesma pessoa.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot

import torch
from torch.nn import functional as F

from src.core.types import Keypoint, Pose
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS
from src.pose.structure_v2 import (
    BILATERAL_PAIRS,
    DECODE_ORDER,
    EXTREMITY_INDICES,
    PARENT_BY_KEYPOINT,
)


LIMB_INDICES = frozenset(
    (
        int(BodyKeypoint.LEFT_ELBOW),
        int(BodyKeypoint.RIGHT_ELBOW),
        int(BodyKeypoint.LEFT_KNEE),
        int(BodyKeypoint.RIGHT_KNEE),
    )
)

LOWER_BODY_SWAP_PAIRS = (
    (int(BodyKeypoint.LEFT_HIP), int(BodyKeypoint.RIGHT_HIP)),
    (int(BodyKeypoint.LEFT_KNEE), int(BodyKeypoint.RIGHT_KNEE)),
    (int(BodyKeypoint.LEFT_ANKLE), int(BodyKeypoint.RIGHT_ANKLE)),
)


@dataclass(frozen=True, slots=True)
class MultiPersonDecodeReport:
    person_count: int
    assigned_candidates: int
    rejected_bilateral_points: int
    suppressed_duplicate_people: int = 0
    corrected_torso_swaps: int = 0


@dataclass(frozen=True, slots=True)
class _Center:
    x: float
    y: float
    score: float


@dataclass(frozen=True, slots=True)
class _Candidate:
    keypoint_index: int
    x: float
    y: float
    score: float
    center_vote_x: float
    center_vote_y: float
    parent_vote_x: float | None
    parent_vote_y: float | None


@dataclass(frozen=True, slots=True)
class _DecodedPerson:
    center: _Center
    pose: Pose
    quality: float


def decode_pose_v2(
    output: dict[str, torch.Tensor],
    center_threshold: float = 0.30,
    keypoint_threshold: float = 0.12,
    max_people: int = 6,
    candidates_per_keypoint: int = 12,
    association_radius: float = 7.0,
    limb_association_factor: float = 1.30,
    extremity_association_factor: float = 1.65,
    parent_sigma: float = 4.0,
    extremity_parent_sigma: float = 4.0,
    bilateral_min_separation: float = 1.5,
    duplicate_center_radius: float = 6.0,
    duplicate_joint_distance: float = 0.05,
    duplicate_overlap_ratio: float = 0.60,
) -> tuple[list[Pose], MultiPersonDecodeReport]:
    """Converte heads V2 em poses independentes e funde duplicatas evidentes."""
    center_logits = _single_batch(output["center"])
    keypoint_logits = _single_batch(output["keypoints"])
    center_offsets = _single_batch(output["center_offsets"])
    parent_offsets = _single_batch(output["parent_offsets"])

    if center_logits.shape[0] != 1:
        raise ValueError("center deve possuir um único canal.")
    if keypoint_logits.shape[0] != NUM_KEYPOINTS:
        raise ValueError(f"keypoints deve possuir {NUM_KEYPOINTS} canais.")
    if center_offsets.shape[0] != NUM_KEYPOINTS * 2:
        raise ValueError("center_offsets deve possuir 34 canais.")
    if parent_offsets.shape[0] != NUM_KEYPOINTS * 2:
        raise ValueError("parent_offsets deve possuir 34 canais.")
    if association_radius <= 0.0:
        raise ValueError("association_radius deve ser positivo.")
    if limb_association_factor < 1.0 or extremity_association_factor < 1.0:
        raise ValueError("Fatores de associação de membros devem ser >= 1.")

    _, height, width = keypoint_logits.shape
    centers = _extract_centers(
        center_logits[0],
        threshold=center_threshold,
        max_people=max_people,
    )
    if not centers:
        return [], MultiPersonDecodeReport(0, 0, 0, 0)

    candidates_by_person: list[list[list[_Candidate]]] = [
        [[] for _ in range(NUM_KEYPOINTS)] for _ in centers
    ]
    assigned_candidates = 0

    for keypoint_index in range(NUM_KEYPOINTS):
        candidates = _extract_keypoint_candidates(
            keypoint_logits[keypoint_index],
            keypoint_index=keypoint_index,
            center_offsets=center_offsets,
            parent_offsets=parent_offsets,
            threshold=keypoint_threshold,
            top_k=max(candidates_per_keypoint, max_people * 2),
        )
        allowed_radius = _association_radius_for_keypoint(
            keypoint_index,
            base_radius=association_radius,
            limb_factor=limb_association_factor,
            extremity_factor=extremity_association_factor,
        )

        for candidate in candidates:
            person_index, distance = _nearest_center(
                candidate.center_vote_x,
                candidate.center_vote_y,
                centers,
            )
            if person_index is None or distance > allowed_radius:
                continue
            candidates_by_person[person_index][keypoint_index].append(candidate)
            assigned_candidates += 1

    decoded_people: list[_DecodedPerson] = []
    rejected_bilateral_points = 0
    corrected_torso_swaps = 0

    for person_index, center in enumerate(centers):
        selected: list[tuple[_Candidate, float] | None] = [None] * NUM_KEYPOINTS

        for keypoint_index in DECODE_ORDER:
            parent_index = PARENT_BY_KEYPOINT[keypoint_index]
            parent = selected[parent_index] if parent_index >= 0 else None
            best: tuple[_Candidate, float] | None = None
            allowed_radius = _association_radius_for_keypoint(
                keypoint_index,
                base_radius=association_radius,
                limb_factor=limb_association_factor,
                extremity_factor=extremity_association_factor,
            )

            for candidate in candidates_by_person[person_index][keypoint_index]:
                center_error = hypot(
                    candidate.center_vote_x - center.x,
                    candidate.center_vote_y - center.y,
                )
                center_factor = exp(
                    -0.5 * (center_error / max(allowed_radius * 0.60, 1e-6)) ** 2
                )
                adjusted = candidate.score * center_factor

                if parent is not None and candidate.parent_vote_x is not None:
                    parent_candidate = parent[0]
                    parent_error = hypot(
                        candidate.parent_vote_x - parent_candidate.x,
                        candidate.parent_vote_y - parent_candidate.y,
                    )
                    sigma = (
                        extremity_parent_sigma
                        if keypoint_index in EXTREMITY_INDICES
                        else parent_sigma
                    )
                    parent_factor = exp(
                        -0.5 * (parent_error / max(sigma, 1e-6)) ** 2
                    )
                    adjusted *= parent_factor

                if best is None or adjusted > best[1]:
                    best = (candidate, adjusted)

            selected[keypoint_index] = best

        # Ombros e quadris não possuem pai em PARENT_BY_KEYPOINT. Validamos
        # explicitamente a coerência entre as duas metades do tronco. Se a
        # atribuição cruzada for claramente melhor, trocamos toda a cadeia
        # inferior para manter quadril->joelho->tornozelo consistente.
        if _align_lower_body_to_shoulders(selected):
            corrected_torso_swaps += 1

        # Perfil-safe: proximidade L/R por si só não é erro. Só removemos um
        # lado quando a cadeia cruzada é claramente mais coerente que a direta.
        for left_index, right_index in BILATERAL_PAIRS:
            left = selected[left_index]
            right = selected[right_index]
            if left is None or right is None:
                continue

            separation = hypot(left[0].x - right[0].x, left[0].y - right[0].y)
            if separation >= bilateral_min_separation:
                continue
            if not _pair_geometry_is_crossed(
                selected,
                left_index=left_index,
                right_index=right_index,
            ):
                continue

            if left[1] >= right[1]:
                selected[right_index] = None
            else:
                selected[left_index] = None
            rejected_bilateral_points += 1

        points: list[Keypoint] = []
        selected_scores: list[float] = []
        for keypoint_index in range(NUM_KEYPOINTS):
            item = selected[keypoint_index]
            if item is None:
                points.append(Keypoint(0.0, 0.0, 0.0))
                continue
            candidate, adjusted_score = item
            confidence = max(0.0, min(1.0, adjusted_score))
            points.append(
                Keypoint(
                    x=candidate.x / max(width - 1, 1),
                    y=candidate.y / max(height - 1, 1),
                    confidence=confidence,
                )
            )
            if confidence >= keypoint_threshold:
                selected_scores.append(confidence)

        valid_count = len(selected_scores)
        if valid_count >= 3:
            mean_confidence = sum(selected_scores) / valid_count
            quality = center.score + mean_confidence + 0.04 * valid_count
            decoded_people.append(
                _DecodedPerson(center=center, pose=Pose(points), quality=quality)
            )

    deduplicated, suppressed_duplicates = _deduplicate_people(
        decoded_people,
        keypoint_threshold=keypoint_threshold,
        center_radius=duplicate_center_radius,
        joint_distance=duplicate_joint_distance,
        overlap_ratio=duplicate_overlap_ratio,
    )
    poses = [person.pose for person in deduplicated]

    return poses, MultiPersonDecodeReport(
        person_count=len(poses),
        assigned_candidates=assigned_candidates,
        rejected_bilateral_points=rejected_bilateral_points,
        suppressed_duplicate_people=suppressed_duplicates,
        corrected_torso_swaps=corrected_torso_swaps,
    )


def _align_lower_body_to_shoulders(
    selected: list[tuple[_Candidate, float] | None],
    *,
    swap_ratio: float = 0.80,
    margin: float = 0.75,
) -> bool:
    """Corrige X estrutural entre ombros e quadris sem usar lado da tela.

    Em perfil, os custos direto e cruzado ficam parecidos; por isso nenhuma
    troca ocorre só por proximidade. A correção exige vantagem clara da
    associação cruzada e troca a cadeia inferior inteira.
    """
    left_shoulder = selected[int(BodyKeypoint.LEFT_SHOULDER)]
    right_shoulder = selected[int(BodyKeypoint.RIGHT_SHOULDER)]
    left_hip = selected[int(BodyKeypoint.LEFT_HIP)]
    right_hip = selected[int(BodyKeypoint.RIGHT_HIP)]

    if any(
        item is None
        for item in (left_shoulder, right_shoulder, left_hip, right_hip)
    ):
        return False

    ls = left_shoulder[0]
    rs = right_shoulder[0]
    lh = left_hip[0]
    rh = right_hip[0]

    direct_cost = hypot(ls.x - lh.x, ls.y - lh.y) + hypot(
        rs.x - rh.x, rs.y - rh.y
    )
    crossed_cost = hypot(ls.x - rh.x, ls.y - rh.y) + hypot(
        rs.x - lh.x, rs.y - lh.y
    )

    if not (
        crossed_cost + margin < direct_cost
        and crossed_cost < direct_cost * swap_ratio
    ):
        return False

    for left_index, right_index in LOWER_BODY_SWAP_PAIRS:
        selected[left_index], selected[right_index] = (
            selected[right_index],
            selected[left_index],
        )
    return True


def _association_radius_for_keypoint(
    keypoint_index: int,
    base_radius: float,
    limb_factor: float,
    extremity_factor: float,
) -> float:
    if keypoint_index in EXTREMITY_INDICES:
        return base_radius * extremity_factor
    if keypoint_index in LIMB_INDICES:
        return base_radius * limb_factor
    return base_radius


def _deduplicate_people(
    people: list[_DecodedPerson],
    *,
    keypoint_threshold: float,
    center_radius: float,
    joint_distance: float,
    overlap_ratio: float,
) -> tuple[list[_DecodedPerson], int]:
    """Funde hipóteses que representam semanticamente a mesma pessoa.

    Distância de centro sozinha nunca é suficiente: duas pessoas reais podem
    estar próximas. Também exigimos que vários keypoints de mesmo significado
    ocupem praticamente a mesma posição.
    """
    if not people:
        return [], 0

    kept: list[_DecodedPerson] = []
    suppressed = 0

    for person in sorted(people, key=lambda item: item.quality, reverse=True):
        duplicate_index: int | None = None
        for index, existing in enumerate(kept):
            center_distance = hypot(
                person.center.x - existing.center.x,
                person.center.y - existing.center.y,
            )
            if center_distance > center_radius:
                continue
            if _poses_semantically_overlap(
                person.pose,
                existing.pose,
                keypoint_threshold=keypoint_threshold,
                joint_distance=joint_distance,
                overlap_ratio=overlap_ratio,
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(person)
            continue

        kept[duplicate_index] = _merge_decoded_people(
            kept[duplicate_index],
            person,
            keypoint_threshold=keypoint_threshold,
        )
        suppressed += 1

    return kept, suppressed


def _poses_semantically_overlap(
    first: Pose,
    second: Pose,
    *,
    keypoint_threshold: float,
    joint_distance: float,
    overlap_ratio: float,
) -> bool:
    comparable = 0
    close = 0
    first_valid = 0
    second_valid = 0

    for first_point, second_point in zip(first.keypoints, second.keypoints):
        first_ok = first_point.is_valid(keypoint_threshold)
        second_ok = second_point.is_valid(keypoint_threshold)
        first_valid += int(first_ok)
        second_valid += int(second_ok)
        if not (first_ok and second_ok):
            continue
        comparable += 1
        distance = hypot(
            first_point.x - second_point.x,
            first_point.y - second_point.y,
        )
        if distance <= joint_distance:
            close += 1

    if comparable < 3 or close < 3:
        return False

    comparable_ratio = close / comparable
    smaller_pose_coverage = close / max(1, min(first_valid, second_valid))
    return (
        comparable_ratio >= overlap_ratio
        and smaller_pose_coverage >= 0.45
    )


def _merge_decoded_people(
    first: _DecodedPerson,
    second: _DecodedPerson,
    *,
    keypoint_threshold: float,
) -> _DecodedPerson:
    points: list[Keypoint] = []
    for first_point, second_point in zip(first.pose.keypoints, second.pose.keypoints):
        if second_point.confidence > first_point.confidence:
            points.append(second_point)
        else:
            points.append(first_point)

    valid = [
        point.confidence
        for point in points
        if point.is_valid(keypoint_threshold)
    ]
    mean_confidence = sum(valid) / len(valid) if valid else 0.0
    better_center = first.center if first.center.score >= second.center.score else second.center
    quality = better_center.score + mean_confidence + 0.04 * len(valid)
    return _DecodedPerson(center=better_center, pose=Pose(points), quality=quality)


def _pair_geometry_is_crossed(
    selected: list[tuple[_Candidate, float] | None],
    left_index: int,
    right_index: int,
) -> bool:
    """Distingue um X real de sobreposição L/R causada por pose de perfil."""
    left_parent_index = PARENT_BY_KEYPOINT[left_index]
    right_parent_index = PARENT_BY_KEYPOINT[right_index]
    if left_parent_index < 0 or right_parent_index < 0:
        return False

    left = selected[left_index]
    right = selected[right_index]
    left_parent = selected[left_parent_index]
    right_parent = selected[right_parent_index]
    if left is None or right is None or left_parent is None or right_parent is None:
        return False

    left_candidate = left[0]
    right_candidate = right[0]
    left_parent_candidate = left_parent[0]
    right_parent_candidate = right_parent[0]

    direct_cost = hypot(
        left_candidate.x - left_parent_candidate.x,
        left_candidate.y - left_parent_candidate.y,
    ) + hypot(
        right_candidate.x - right_parent_candidate.x,
        right_candidate.y - right_parent_candidate.y,
    )
    crossed_cost = hypot(
        left_candidate.x - right_parent_candidate.x,
        left_candidate.y - right_parent_candidate.y,
    ) + hypot(
        right_candidate.x - left_parent_candidate.x,
        right_candidate.y - left_parent_candidate.y,
    )

    parent_separation = hypot(
        left_parent_candidate.x - right_parent_candidate.x,
        left_parent_candidate.y - right_parent_candidate.y,
    )
    margin = max(0.50, parent_separation * 0.10)
    return crossed_cost + margin < direct_cost and crossed_cost < direct_cost * 0.80


def _single_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("Decoder V2 aceita tensores [1,C,H,W].")
    return tensor[0].detach().float().cpu()


def _nms_probabilities(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    pooled = F.max_pool2d(
        probability.unsqueeze(0).unsqueeze(0),
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0]
    return probability * (probability >= pooled).to(probability.dtype)


def _extract_centers(
    logits: torch.Tensor,
    threshold: float,
    max_people: int,
) -> list[_Center]:
    heatmap = _nms_probabilities(logits)
    flat = heatmap.flatten()
    count = min(max_people, flat.numel())
    values, indices = torch.topk(flat, k=count)
    width = heatmap.shape[1]
    centers: list[_Center] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        if value < threshold:
            continue
        y = index // width
        x = index % width
        centers.append(_Center(float(x), float(y), float(value)))
    return centers


def _extract_keypoint_candidates(
    logits: torch.Tensor,
    keypoint_index: int,
    center_offsets: torch.Tensor,
    parent_offsets: torch.Tensor,
    threshold: float,
    top_k: int,
) -> list[_Candidate]:
    heatmap = _nms_probabilities(logits)
    flat = heatmap.flatten()
    count = min(top_k, flat.numel())
    values, indices = torch.topk(flat, k=count)
    width = heatmap.shape[1]
    candidates: list[_Candidate] = []
    parent_index = PARENT_BY_KEYPOINT[keypoint_index]

    for value, index in zip(values.tolist(), indices.tolist()):
        if value < threshold:
            continue
        y = int(index // width)
        x = int(index % width)
        dx = float(center_offsets[2 * keypoint_index, y, x].item())
        dy = float(center_offsets[2 * keypoint_index + 1, y, x].item())

        parent_vote_x: float | None = None
        parent_vote_y: float | None = None
        if parent_index >= 0:
            parent_dx = float(parent_offsets[2 * keypoint_index, y, x].item())
            parent_dy = float(parent_offsets[2 * keypoint_index + 1, y, x].item())
            parent_vote_x = float(x) + parent_dx
            parent_vote_y = float(y) + parent_dy

        candidates.append(
            _Candidate(
                keypoint_index=keypoint_index,
                x=float(x),
                y=float(y),
                score=float(value),
                center_vote_x=float(x) + dx,
                center_vote_y=float(y) + dy,
                parent_vote_x=parent_vote_x,
                parent_vote_y=parent_vote_y,
            )
        )
    return candidates


def _nearest_center(
    x: float,
    y: float,
    centers: list[_Center],
) -> tuple[int | None, float]:
    best_index: int | None = None
    best_distance = float("inf")
    for index, center in enumerate(centers):
        distance = hypot(x - center.x, y - center.y)
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index, best_distance
