"""Decoder multi-pessoa da PoseNet V2.

A associação ocorre em duas etapas:
1. cada candidato de keypoint vota no centro da pessoa através de center_offsets;
2. quando existe pai anatômico, parent_offsets mede se o candidato realmente se
   conecta ao ombro/cotovelo/quadril/joelho correto.

A proteção anti-X não assume mais que pares L/R próximos são necessariamente
errados. Em perfil, ombros, braços e pernas podem se sobrepor naturalmente na
projeção 2D. Um ponto só é removido quando, além da proximidade, a geometria da
cadeia indica que a associação cruzada é claramente mais coerente que a direta.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot

import torch
from torch.nn import functional as F

from src.core.types import Keypoint, Pose
from src.pose.keypoints import NUM_KEYPOINTS
from src.pose.structure_v2 import (
    BILATERAL_PAIRS,
    DECODE_ORDER,
    EXTREMITY_INDICES,
    PARENT_BY_KEYPOINT,
)


@dataclass(frozen=True, slots=True)
class MultiPersonDecodeReport:
    person_count: int
    assigned_candidates: int
    rejected_bilateral_points: int


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


def decode_pose_v2(
    output: dict[str, torch.Tensor],
    center_threshold: float = 0.30,
    keypoint_threshold: float = 0.12,
    max_people: int = 6,
    candidates_per_keypoint: int = 12,
    association_radius: float = 7.0,
    parent_sigma: float = 4.0,
    extremity_parent_sigma: float = 2.5,
    bilateral_min_separation: float = 1.5,
) -> tuple[list[Pose], MultiPersonDecodeReport]:
    """Converte heads V2 em uma lista de poses, uma por centro detectado."""
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

    _, height, width = keypoint_logits.shape
    centers = _extract_centers(
        center_logits[0],
        threshold=center_threshold,
        max_people=max_people,
    )
    if not centers:
        return [], MultiPersonDecodeReport(0, 0, 0)

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

        for candidate in candidates:
            person_index, distance = _nearest_center(
                candidate.center_vote_x,
                candidate.center_vote_y,
                centers,
            )
            if person_index is None or distance > association_radius:
                continue
            candidates_by_person[person_index][keypoint_index].append(candidate)
            assigned_candidates += 1

    poses: list[Pose] = []
    rejected_bilateral_points = 0

    for person_index, center in enumerate(centers):
        selected: list[tuple[_Candidate, float] | None] = [None] * NUM_KEYPOINTS

        for keypoint_index in DECODE_ORDER:
            parent_index = PARENT_BY_KEYPOINT[keypoint_index]
            parent = selected[parent_index] if parent_index >= 0 else None
            best: tuple[_Candidate, float] | None = None

            for candidate in candidates_by_person[person_index][keypoint_index]:
                center_error = hypot(
                    candidate.center_vote_x - center.x,
                    candidate.center_vote_y - center.y,
                )
                center_factor = exp(
                    -0.5 * (center_error / max(association_radius * 0.55, 1e-6)) ** 2
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
                    parent_factor = exp(-0.5 * (parent_error / max(sigma, 1e-6)) ** 2)
                    adjusted *= parent_factor

                if best is None or adjusted > best[1]:
                    best = (candidate, adjusted)

            selected[keypoint_index] = best

        # Perfil-safe: proximidade L/R por si só NÃO é erro. Ombros, quadris,
        # joelhos etc. podem se sobrepor quando a pessoa gira de lado. Só
        # rejeitamos um lado se a cadeia parecer realmente cruzada em relação
        # aos pais semânticos (ex.: punho esquerdo geometricamente ligado ao
        # cotovelo direito e vice-versa).
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
        for keypoint_index in range(NUM_KEYPOINTS):
            item = selected[keypoint_index]
            if item is None:
                points.append(Keypoint(0.0, 0.0, 0.0))
                continue
            candidate, adjusted_score = item
            points.append(
                Keypoint(
                    x=candidate.x / max(width - 1, 1),
                    y=candidate.y / max(height - 1, 1),
                    confidence=max(0.0, min(1.0, adjusted_score)),
                )
            )

        # Um centro sozinho não basta: exigimos pelo menos alguns pontos
        # associados para materializar uma pessoa na saída.
        valid_count = sum(point.confidence >= keypoint_threshold for point in points)
        if valid_count >= 3:
            poses.append(Pose(points))

    return poses, MultiPersonDecodeReport(
        person_count=len(poses),
        assigned_candidates=assigned_candidates,
        rejected_bilateral_points=rejected_bilateral_points,
    )


def _pair_geometry_is_crossed(
    selected: list[tuple[_Candidate, float] | None],
    left_index: int,
    right_index: int,
) -> bool:
    """Distingue um X real de sobreposição L/R causada por pose de perfil.

    Para pares sem pai (ombros/quadris) a proximidade é sempre permitida. Para
    cotovelos, punhos, joelhos e tornozelos comparamos o custo geométrico da
    cadeia direta contra a cadeia trocada. Em perfil os custos tendem a ficar
    próximos; em um X verdadeiro a ligação cruzada fica claramente menor.
    """
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
