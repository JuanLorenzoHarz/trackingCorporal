"""Ponto de entrada do trackingCorporal.

Modos atuais:
- demo: webcam real + pose artificial fixa;
- model: webcam real + CNN + resolução bilateral + plausibilidade + tracking.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.capture.camera import Camera
from src.pose.decoder import decode_heatmaps_bilateral
from src.pose.demo_pose import build_demo_pose
from src.pose.keypoints import NUM_KEYPOINTS
from src.pose.model import PoseNet
from src.preprocessing.frame_preprocessor import preprocess_frame_with_transform
from src.tracking.bilateral import BilateralLegIdentityTracker
from src.tracking.plausibility import PosePlausibilityEvaluator
from src.tracking.smoothing import ExponentialPoseSmoother
from src.tracking.temporal_tracker import TemporalPoseTracker
from src.visualization.renderer import draw_pose


WINDOW_NAME = "trackingCorporal"
CAMERA_INDEX = 0
REQUESTED_WIDTH = 640
REQUESTED_HEIGHT = 480


def draw_status(
    frame: np.ndarray,
    fps: float,
    mode: str,
    predicted_count: int = 0,
    plausibility_percentage: float = 100.0,
    leg_plausibility_percentage: float = 100.0,
    bilateral_percentage: float = 100.0,
    bilateral_decode_corrections: int = 0,
    bilateral_collapses: int = 0,
    bilateral_swapped: bool = False,
    rejected_count: int = 0,
    calibrated_segments: int = 0,
) -> np.ndarray:
    """Exibe informações úteis da etapa atual sobre o frame."""
    if mode == "model":
        status = "POSE MODEL + BILATERAL + PLAUSIBILITY + TRACKING"
    else:
        status = "POSE DEMO - NOT TRACKING"

    cv2.putText(
        frame,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    line_y = 95
    if mode == "model":
        lines = (
            f"Plausibilidade: {plausibility_percentage:.0f}%",
            f"Pernas: {leg_plausibility_percentage:.0f}% | L/R: {bilateral_percentage:.0f}%",
            (
                f"Picos L/R corrigidos: {bilateral_decode_corrections} | "
                f"colapsos: {bilateral_collapses}"
            ),
            f"Troca de identidade corrigida: {'sim' if bilateral_swapped else 'nao'}",
            f"Keypoints rejeitados: {rejected_count} | previstos: {predicted_count}",
            f"Calibracao anatomica: {calibrated_segments}/12",
        )
        for text in lines:
            cv2.putText(
                frame,
                text,
                (20, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            line_y += 27

    cv2.putText(
        frame,
        "Q ou ESC para sair",
        (20, line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def load_pose_model(
    weights_path: str | Path,
    device: torch.device,
) -> tuple[PoseNet, int]:
    """Carrega os pesos produzidos por scripts/train_pose.py."""
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Pesos não encontrados: {path}. Treine o modelo antes de usar --mode model."
        )

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError("Checkpoint inválido: model_state não encontrado.")

    keypoint_count = int(checkpoint.get("keypoint_count", NUM_KEYPOINTS))
    if keypoint_count != NUM_KEYPOINTS:
        raise RuntimeError(
            f"Checkpoint possui {keypoint_count} keypoints; esperados {NUM_KEYPOINTS}."
        )

    model = PoseNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_size = int(checkpoint.get("input_size", 256))
    return model, input_size


def main(args: argparse.Namespace | None = None) -> None:
    """Executa o loop de vídeo em tempo real."""
    if args is None:
        args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo_pose = build_demo_pose() if args.mode == "demo" else None
    model: PoseNet | None = None
    input_size = 256
    bilateral: BilateralLegIdentityTracker | None = None
    plausibility: PosePlausibilityEvaluator | None = None
    tracker: TemporalPoseTracker | None = None
    smoother: ExponentialPoseSmoother | None = None

    if args.mode == "model":
        model, input_size = load_pose_model(args.weights, device)
        bilateral = BilateralLegIdentityTracker(
            detection_threshold=args.confidence,
            collapse_ratio=args.bilateral_collapse_ratio,
        )
        plausibility = PosePlausibilityEvaluator(
            detection_threshold=args.confidence,
            reject_threshold=args.plausibility_reject_threshold,
        )
        tracker = TemporalPoseTracker(
            detection_threshold=args.confidence,
            max_missing_frames=args.prediction_frames,
            confidence_decay=args.prediction_decay,
            anatomy_weight=args.anatomy_weight,
        )
        smoother = ExponentialPoseSmoother(alpha=args.smoothing_alpha)

        print(f"Modelo de pose carregado em {device}.")
        print(
            "Pré-processamento sem distorção ativo: o maior quadrado central da "
            "webcam é usado pela CNN e os keypoints são remapeados para o frame original."
        )
        print(
            "Resolução bilateral ativa: joelhos/tornozelos usam múltiplos picos e "
            "identidade temporal para reduzir fusão esquerda/direita."
        )
        print(
            "Filtro de plausibilidade ativo: proporções corporais e continuidade "
            "temporal serão aprendidas durante os primeiros frames."
        )
        if args.disable_plausibility_filter:
            print(
                "AVISO: filtro corretivo de plausibilidade desativado; o score ainda "
                "será calculado. A correção bilateral continua ativa."
            )
        print(
            f"Detecção mínima {args.confidence:.2f} | "
            f"render mínimo {args.render_confidence:.2f} | "
            f"rejeição plausibilidade {args.plausibility_reject_threshold:.2f} | "
            f"colapso L/R {args.bilateral_collapse_ratio:.2f}x largura do quadril | "
            f"previsão até {args.prediction_frames} frames | "
            f"decay {args.prediction_decay:.2f} | "
            f"peso anatômico {args.anatomy_weight:.2f} | "
            f"smoothing {args.smoothing_alpha:.2f}."
        )

    previous_time = time.perf_counter()

    try:
        with Camera(
            index=args.camera,
            width=REQUESTED_WIDTH,
            height=REQUESTED_HEIGHT,
        ) as camera:
            width, height = camera.resolution
            print(f"Câmera {args.camera} aberta em {width}x{height}.")

            if args.mode == "demo":
                print("A pose exibida é artificial e ainda NÃO acompanha o corpo.")
            else:
                print(
                    "A pose passa por resolução bilateral, plausibilidade e depois "
                    "é refinada pelo histórico temporal."
                )

            print("Pressione Q ou ESC na janela para encerrar.")

            while True:
                frame = camera.read()
                predicted_count = 0
                plausibility_percentage = 100.0
                leg_plausibility_percentage = 100.0
                bilateral_percentage = 100.0
                bilateral_decode_corrections = 0
                bilateral_collapses = 0
                bilateral_swapped = False
                rejected_count = 0
                calibrated_segments = 0

                if args.mode == "model":
                    assert model is not None
                    assert bilateral is not None
                    assert plausibility is not None
                    assert tracker is not None
                    assert smoother is not None

                    model_input, transform = preprocess_frame_with_transform(
                        frame,
                        input_size=input_size,
                        device=device,
                    )
                    with torch.inference_mode():
                        heatmaps = model(model_input)

                    crop_pose, decode_report = decode_heatmaps_bilateral(
                        heatmaps,
                        top_k=args.bilateral_top_k,
                        suppression_radius=args.bilateral_suppression_radius,
                        minimum_separation_pixels=args.bilateral_peak_separation,
                        minimum_alternative_ratio=args.bilateral_alternative_ratio,
                    )
                    raw_pose = transform.pose_to_original(crop_pose)
                    identity_pose, bilateral_report = bilateral.update(raw_pose)
                    filtered_pose, report = plausibility.evaluate_and_filter(identity_pose)
                    pose_for_tracker = (
                        identity_pose if args.disable_plausibility_filter else filtered_pose
                    )
                    tracked_pose = tracker.update(pose_for_tracker)
                    pose = smoother.update(tracked_pose)

                    predicted_count = tracker.predicted_count
                    plausibility_percentage = report.percentage
                    leg_plausibility_percentage = report.leg_percentage
                    bilateral_percentage = bilateral_report.percentage
                    bilateral_decode_corrections = decode_report.corrected_pairs
                    bilateral_collapses = bilateral_report.collapsed_pairs
                    bilateral_swapped = bilateral_report.swapped_chain
                    rejected_count = len(
                        report.suspicious_indices | bilateral_report.rejected_indices
                    )
                    calibrated_segments = report.calibrated_segments

                    draw_pose(
                        frame,
                        pose,
                        confidence_threshold=args.render_confidence,
                    )
                else:
                    assert demo_pose is not None
                    draw_pose(frame, demo_pose)

                current_time = time.perf_counter()
                elapsed = current_time - previous_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                previous_time = current_time

                draw_status(
                    frame,
                    fps,
                    args.mode,
                    predicted_count=predicted_count,
                    plausibility_percentage=plausibility_percentage,
                    leg_plausibility_percentage=leg_plausibility_percentage,
                    bilateral_percentage=bilateral_percentage,
                    bilateral_decode_corrections=bilateral_decode_corrections,
                    bilateral_collapses=bilateral_collapses,
                    bilateral_swapped=bilateral_swapped,
                    rejected_count=rejected_count,
                    calibrated_segments=calibrated_segments,
                )
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

    except (RuntimeError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Erro: {error}") from error
    finally:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="trackingCorporal em tempo real.")
    parser.add_argument(
        "--mode",
        choices=("demo", "model"),
        default="demo",
        help="demo usa pose fixa; model usa a CNN treinada.",
    )
    parser.add_argument("--weights", default="models/pose_model.pt")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.15,
        help="Confiança mínima para uma observação da CNN ser considerada real.",
    )
    parser.add_argument(
        "--render-confidence",
        type=float,
        default=0.05,
        help="Confiança mínima para desenhar a pose final, incluindo previsões em decay.",
    )
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument(
        "--prediction-frames",
        type=int,
        default=8,
        help="Máximo de frames em que um keypoint oculto continua sendo previsto.",
    )
    parser.add_argument(
        "--prediction-decay",
        type=float,
        default=0.82,
        help="Multiplicador de confiança aplicado a cada frame previsto.",
    )
    parser.add_argument(
        "--anatomy-weight",
        type=float,
        default=0.60,
        help="Peso da geometria dos membros contra a extrapolação por velocidade.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.65,
        help="Peso da posição atual na suavização exponencial; maior = resposta mais rápida.",
    )
    parser.add_argument(
        "--plausibility-reject-threshold",
        type=float,
        default=0.16,
        help="Abaixo deste score local, uma observação muito improvável pode ser rejeitada.",
    )
    parser.add_argument(
        "--disable-plausibility-filter",
        action="store_true",
        help="Calcula a métrica, mas não rejeita keypoints; útil para comparação A/B.",
    )
    parser.add_argument(
        "--bilateral-top-k",
        type=int,
        default=3,
        help="Quantidade de picos candidatos por joelho/tornozelo.",
    )
    parser.add_argument(
        "--bilateral-suppression-radius",
        type=int,
        default=4,
        help="Raio em pixels do heatmap usado para procurar máximas locais distintas.",
    )
    parser.add_argument(
        "--bilateral-peak-separation",
        type=float,
        default=3.0,
        help="Separação mínima em pixels do heatmap para considerar dois picos distintos.",
    )
    parser.add_argument(
        "--bilateral-alternative-ratio",
        type=float,
        default=0.65,
        help="Confiança mínima da hipótese alternativa em relação ao melhor pico.",
    )
    parser.add_argument(
        "--bilateral-collapse-ratio",
        type=float,
        default=0.12,
        help="Distância L/R abaixo desta fração da largura do quadril indica possível colapso.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
