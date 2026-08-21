"""Decodificação de heatmaps em coordenadas normalizadas de keypoints."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import torch

from src.core.types import Keypoint, Pose
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS


@dataclass(frozen=True, slots=True)
class BilateralDecodeReport:
    """Diagnóstico de correções esquerda/direita feitas no decoder."""

    corrected_pairs: int = 0


@dataclass(frozen=True, slots=True)
class _PeakCandidate:
    x: int
    y: int
    confidence: float


LOWER_BODY_PAIRS: tuple[tuple[int, int], ...] = (
    (int(BodyKeypoint.LEFT_KNEE), int(BodyKeypoint.RIGHT_KNEE)),
    (int(BodyKeypoint.LEFT_ANKLE), int(BodyKeypoint.RIGHT_ANKLE)),
)


def decode_heatmaps(heatmaps: torch.Tensor) -> Pose:
    """Converte heatmaps [17,H,W] ou [1,17,H,W] em uma Pose.

    Esta função mantém o comportamento simples original: cada keypoint usa o
    maior valor do respectivo heatmap.
    """
    heatmaps = _prepare_heatmaps(heatmaps)
    _, height, width = heatmaps.shape
    points: list[Keypoint] = []

    for heatmap in heatmaps:
        candidate = _top_candidates(heatmap, top_k=1, suppression_radius=0)[0]
        points.append(_candidate_to_keypoint(candidate, width, height))

    return Pose(points)


def decode_heatmaps_bilateral(
    heatmaps: torch.Tensor,
    top_k: int = 3,
    suppression_radius: int = 4,
    minimum_separation_pixels: float = 3.0,
    minimum_alternative_ratio: float = 0.65,
    minimum_pair_score_ratio: float = 0.82,
) -> tuple[Pose, BilateralDecodeReport]:
    """Decodifica heatmaps tentando evitar colapso das duas pernas no mesmo pico.

    Cada canal continua usando seu maior pico normalmente. Quando joelho ou
    tornozelo esquerdo/direito ficam praticamente no mesmo ponto, procuramos
    outras máximas locais. Uma alternativa só é aceita quando:

    - os dois canais possuem evidência positiva para o par;
    - fica suficientemente distante do outro lado;
    - mantém confiança razoavelmente próxima ao melhor pico do próprio canal;
    - a soma das duas confianças não cai demais em relação à solução original.

    Dessa forma uma segunda hipótese real pode ser recuperada, mas um pico fraco
    de ruído ou um heatmap zerado não é promovido apenas para forçar duas pernas
    separadas.
    """
    if top_k < 1:
        raise ValueError("top_k deve ser pelo menos 1.")
    if suppression_radius < 0:
        raise ValueError("suppression_radius não pode ser negativo.")
    if minimum_separation_pixels < 0.0:
        raise ValueError("minimum_separation_pixels não pode ser negativo.")
    if not 0.0 <= minimum_alternative_ratio <= 1.0:
        raise ValueError("minimum_alternative_ratio deve estar entre 0 e 1.")
    if not 0.0 <= minimum_pair_score_ratio <= 1.0:
        raise ValueError("minimum_pair_score_ratio deve estar entre 0 e 1.")

    heatmaps = _prepare_heatmaps(heatmaps)
    _, height, width = heatmaps.shape

    candidates_by_keypoint = [
        _top_candidates(
            heatmap,
            top_k=top_k,
            suppression_radius=suppression_radius,
        )
        for heatmap in heatmaps
    ]
    selected = [candidates[0] for candidates in candidates_by_keypoint]
    corrected_pairs = 0

    for left_index, right_index in LOWER_BODY_PAIRS:
        left_top = selected[left_index]
        right_top = selected[right_index]

        # Heatmaps sem evidência positiva não possuem uma segunda hipótese útil.
        # Sem este corte, um mapa totalmente zerado poderia gerar "picos"
        # arbitrários de confiança 0 após a supressão e ser contado como correção.
        if left_top.confidence <= 0.0 or right_top.confidence <= 0.0:
            continue

        top_distance = hypot(left_top.x - right_top.x, left_top.y - right_top.y)
        if top_distance >= minimum_separation_pixels:
            continue

        original_score = left_top.confidence + right_top.confidence
        minimum_pair_score = original_score * minimum_pair_score_ratio
        best_pair: tuple[_PeakCandidate, _PeakCandidate] | None = None
        best_score = float("-inf")

        left_min_confidence = left_top.confidence * minimum_alternative_ratio
        right_min_confidence = right_top.confidence * minimum_alternative_ratio

        for left_candidate in candidates_by_keypoint[left_index]:
            if left_candidate.confidence <= 0.0:
                continue
            if left_candidate.confidence < left_min_confidence:
                continue

            for right_candidate in candidates_by_keypoint[right_index]:
                if right_candidate.confidence <= 0.0:
                    continue
                if right_candidate.confidence < right_min_confidence:
                    continue

                distance = hypot(
                    left_candidate.x - right_candidate.x,
                    left_candidate.y - right_candidate.y,
                )
                if distance < minimum_separation_pixels:
                    continue

                pair_score = left_candidate.confidence + right_candidate.confidence
                if pair_score < minimum_pair_score:
                    continue

                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = (left_candidate, right_candidate)

        if best_pair is not None:
            selected[left_index], selected[right_index] = best_pair
            corrected_pairs += 1

    points = [
        _candidate_to_keypoint(candidate, width, height)
        for candidate in selected
    ]
    return Pose(points), BilateralDecodeReport(corrected_pairs=corrected_pairs)


def _prepare_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    if heatmaps.ndim == 4:
        if heatmaps.shape[0] != 1:
            raise ValueError("decode_heatmaps aceita apenas batch de tamanho 1.")
        heatmaps = heatmaps[0]

    if heatmaps.ndim != 3:
        raise ValueError("heatmaps deve possuir formato [K,H,W] ou [1,K,H,W].")
    if heatmaps.shape[0] != NUM_KEYPOINTS:
        raise ValueError(
            f"Esperados {NUM_KEYPOINTS} heatmaps, recebidos {heatmaps.shape[0]}."
        )

    return heatmaps.detach().float().cpu()


def _top_candidates(
    heatmap: torch.Tensor,
    top_k: int,
    suppression_radius: int,
) -> list[_PeakCandidate]:
    """Retorna máximas locais separadas por NMS quadrado simples.

    O primeiro máximo é sempre retornado para manter compatibilidade com o
    decoder simples. Depois dele, candidatos com confiança <= 0 não são úteis
    como hipóteses alternativas e encerram a busca.
    """
    height, width = heatmap.shape
    working = heatmap.clone()
    candidates: list[_PeakCandidate] = []

    for candidate_index in range(top_k):
        flat_index = int(torch.argmax(working).item())
        y = flat_index // width
        x = flat_index % width
        raw_confidence = float(heatmap[y, x].item())
        confidence = min(1.0, max(0.0, raw_confidence))

        if candidate_index > 0 and confidence <= 0.0:
            break

        candidates.append(_PeakCandidate(x=x, y=y, confidence=confidence))

        if suppression_radius == 0 or confidence <= 0.0:
            break

        left = max(0, x - suppression_radius)
        right = min(width, x + suppression_radius + 1)
        top = max(0, y - suppression_radius)
        bottom = min(height, y + suppression_radius + 1)
        working[top:bottom, left:right] = float("-inf")

    return candidates


def _candidate_to_keypoint(
    candidate: _PeakCandidate,
    width: int,
    height: int,
) -> Keypoint:
    normalized_x = candidate.x / (width - 1) if width > 1 else 0.0
    normalized_y = candidate.y / (height - 1) if height > 1 else 0.0
    return Keypoint(
        x=normalized_x,
        y=normalized_y,
        confidence=candidate.confidence,
    )