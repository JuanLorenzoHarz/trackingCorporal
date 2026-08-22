"""Decodificação de heatmaps em coordenadas normalizadas de keypoints."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt
from statistics import mean

import torch

from src.core.types import Keypoint, Pose
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS


@dataclass(frozen=True, slots=True)
class BilateralDecodeReport:
    """Diagnóstico de correções esquerda/direita feitas no decoder."""

    corrected_pairs: int = 0


@dataclass(frozen=True, slots=True)
class PeakQualityDecodeReport:
    """Diagnóstico da nitidez/unicidade dos picos escolhidos."""

    mean_quality: float
    mean_raw_confidence: float
    ambiguous_keypoints: int

    @property
    def percentage(self) -> float:
        return self.mean_quality * 100.0


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
    """Decoder simples original: usa apenas o maior valor de cada heatmap."""
    heatmaps = _prepare_heatmaps(heatmaps)
    _, height, width = heatmaps.shape
    points: list[Keypoint] = []

    for heatmap in heatmaps:
        candidate = _top_candidates(heatmap, top_k=1, suppression_radius=0)[0]
        points.append(_candidate_to_keypoint(candidate, width, height))

    return Pose(points)


def decode_heatmaps_reliable(
    heatmaps: torch.Tensor,
    suppression_radius: int = 4,
    full_confidence_dominance: float = 0.35,
    ambiguous_quality_threshold: float = 0.50,
) -> tuple[Pose, PeakQualityDecodeReport]:
    """Decodifica o melhor pico, mas reduz confiança quando há ambiguidade.

    A rede foi treinada com MSE em heatmaps e o valor máximo bruto não é uma
    probabilidade calibrada. Um objeto de fundo pode produzir um máximo positivo
    mesmo sem existir uma articulação real. Para evitar confiar cegamente nesse
    número, comparamos o maior pico com a melhor hipótese espacialmente distinta.

    Se o maior e o segundo pico são muito parecidos, a confiança final cai. A
    posição escolhida continua sendo o argmax; este decoder nunca promove o 2º
    ou 3º pico, portanto segue a política conservadora de "na dúvida, não inventar".
    """
    if suppression_radius < 0:
        raise ValueError("suppression_radius não pode ser negativo.")
    if full_confidence_dominance <= 0.0:
        raise ValueError("full_confidence_dominance deve ser positivo.")
    if not 0.0 <= ambiguous_quality_threshold <= 1.0:
        raise ValueError("ambiguous_quality_threshold deve estar entre 0 e 1.")

    heatmaps = _prepare_heatmaps(heatmaps)
    _, height, width = heatmaps.shape
    points: list[Keypoint] = []
    qualities: list[float] = []
    raw_confidences: list[float] = []
    ambiguous = 0

    for heatmap in heatmaps:
        candidates = _top_candidates(
            heatmap,
            top_k=2,
            suppression_radius=suppression_radius,
        )
        primary = candidates[0]
        secondary_confidence = candidates[1].confidence if len(candidates) > 1 else 0.0
        raw_confidences.append(primary.confidence)

        if primary.confidence <= 0.0:
            quality = 0.0
            calibrated_confidence = 0.0
        elif secondary_confidence <= 0.0:
            quality = 1.0
            calibrated_confidence = primary.confidence
        else:
            dominance = max(
                0.0,
                (primary.confidence - secondary_confidence)
                / max(primary.confidence, 1e-6),
            )
            quality = min(1.0, dominance / full_confidence_dominance)
            # sqrt deixa a redução progressiva: ambiguidade baixa não mata um
            # keypoint bom, mas dois picos quase iguais perdem bastante confiança.
            calibrated_confidence = primary.confidence * sqrt(quality)

        if quality < ambiguous_quality_threshold:
            ambiguous += 1
        qualities.append(quality)
        points.append(
            _candidate_to_keypoint(
                _PeakCandidate(
                    x=primary.x,
                    y=primary.y,
                    confidence=calibrated_confidence,
                ),
                width,
                height,
            )
        )

    report = PeakQualityDecodeReport(
        mean_quality=mean(qualities) if qualities else 0.0,
        mean_raw_confidence=mean(raw_confidences) if raw_confidences else 0.0,
        ambiguous_keypoints=ambiguous,
    )
    return Pose(points), report


def decode_heatmaps_bilateral(
    heatmaps: torch.Tensor,
    top_k: int = 3,
    suppression_radius: int = 4,
    minimum_separation_pixels: float = 3.0,
    minimum_alternative_ratio: float = 0.65,
    minimum_pair_score_ratio: float = 0.82,
) -> tuple[Pose, BilateralDecodeReport]:
    """Decoder experimental que pode promover picos alternativos das pernas.

    Ele permanece disponível para comparação, mas não deve ser a opção padrão:
    promover hipóteses secundárias pode aumentar alucinações quando os heatmaps
    ainda não estão suficientemente bem separados.
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
    """Retorna máximas locais separadas por NMS quadrado simples."""
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
