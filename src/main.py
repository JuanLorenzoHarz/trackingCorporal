"""Ponto de entrada do trackingCorporal.

Pipeline padrão conservador:
webcam -> CNN -> confiança por qualidade de pico -> gate de presença ->
identidade bilateral -> plausibilidade -> tracking temporal -> smoothing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.capture.camera import Camera
from src.pose.decoder import decode_heatmaps_bilateral, decode_heatmaps_reliable
from src.pose.demo_pose import build_demo_pose
from src.pose.keypoints import NUM_KEYPOINTS
from src.pose.model import PoseNet
from src.preprocessing.frame_preprocessor import preprocess_frame_with_transform
from src.tracking.bilateral import BilateralLegIdentityTracker
from src.tracking.plausibility import PosePlausibilityEvaluator
from src.tracking.presence import BodyPresenceGate
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
    presence_active: bool = False,
    presence_percentage: float = 0.0,
    peak_quality_percentage: float = 100.0,
    ambiguous_keypoints: int = 0,
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
    status = (
        "POSE MODEL - CONSERVATIVE TRACKING"
        if mode == "model"
        else "POSE DEMO - NOT TRACKING"
    )
    cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    line_y = 95
    if mode == "model":
        lines = (
            f"Presenca: {'ATIVA' if presence_active else 'aguardando'} | evidencia {presence_percentage:.0f}%",
            f"Qualidade dos picos: {peak_quality_percentage:.0f}% | ambiguos: {ambiguous_keypoints}",
            f"Plausibilidade: {plausibility_percentage:.0f}%",
            f"Pernas: {leg_plausibility_percentage:.0f}% | L/R: {bilateral_percentage:.0f}%",
            f"Picos secundarios promovidos: {bilateral_decode_corrections} | colapsos: {bilateral_collapses}",
            f"Troca L/R corrigida: {'sim' if bilateral_swapped else 'nao'}",
            f"Keypoints rejeitados: {rejected_count} | previstos: {predicted_count}",
            f"Calibracao anatomica: {calibrated_segments}/12",
        )
        for text in lines:
            cv2.putText(frame, text, (20, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            line_y += 25

    cv2.putText(frame, "Q ou ESC para sair", (20, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def load_pose_model(weights_path: str | Path, device: torch.device) -> tuple[PoseNet, int]:
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
    return model, int(checkpoint.get("input_size", 256))


def _reset_tracking_state(
    bilateral: BilateralLegIdentityTracker,
    plausibility: PosePlausibilityEvaluator,
    tracker: TemporalPoseTracker,
    smoother: ExponentialPoseSmoother,
) -> None:
    bilateral.reset()
    plausibility.reset()
    tracker.reset()
    smoother.reset()


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo_pose = build_demo_pose() if args.mode == "demo" else None
    model: PoseNet | None = None
    input_size = 256
    presence: BodyPresenceGate | None = None
    bilateral: BilateralLegIdentityTracker | None = None
    plausibility: PosePlausibilityEvaluator | None = None
    tracker: TemporalPoseTracker | None = None
    smoother: ExponentialPoseSmoother | None = None

    if args.mode == "model":
        model, input_size = load_pose_model(args.weights, device)
        presence = BodyPresenceGate(
            detection_threshold=args.presence_confidence,
            minimum_keypoints=args.presence_min_keypoints,
            minimum_torso_keypoints=args.presence_min_torso,
            acquire_frames=args.presence_acquire_frames,
            release_frames=args.presence_release_frames,
        )
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
        print("Decoder conservador ativo: picos ambíguos perdem confiança em vez de promover hipóteses secundárias.")
        print("Gate de presença ativo: o tracking só é liberado após evidência corporal consistente.")
        if args.enable_bilateral_multipeak:
            print("AVISO: promoção experimental de 2º/3º pico bilateral ATIVA.")
        else:
            print("Promoção de 2º/3º pico bilateral DESATIVADA (recomendado).")

    previous_time = time.perf_counter()

    try:
        with Camera(index=args.camera, width=REQUESTED_WIDTH, height=REQUESTED_HEIGHT) as camera:
            width, height = camera.resolution
            print(f"Câmera {args.camera} aberta em {width}x{height}.")
            print("Pressione Q ou ESC na janela para encerrar.")

            while True:
                frame = camera.read()
                presence_active = args.mode != "model"
                presence_percentage = 100.0 if presence_active else 0.0
                peak_quality_percentage = 100.0
                ambiguous_keypoints = 0
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
                    assert presence is not None
                    assert bilateral is not None
                    assert plausibility is not None
                    assert tracker is not None
                    assert smoother is not None

                    model_input, transform = preprocess_frame_with_transform(frame, input_size=input_size, device=device)
                    with torch.inference_mode():
                        heatmaps = model(model_input)

                    if args.enable_bilateral_multipeak:
                        crop_pose, decode_report = decode_heatmaps_bilateral(
                            heatmaps,
                            top_k=args.bilateral_top_k,
                            suppression_radius=args.bilateral_suppression_radius,
                            minimum_separation_pixels=args.bilateral_peak_separation,
                            minimum_alternative_ratio=args.bilateral_alternative_ratio,
                        )
                        bilateral_decode_corrections = decode_report.corrected_pairs
                    else:
                        crop_pose, quality_report = decode_heatmaps_reliable(
                            heatmaps,
                            suppression_radius=args.peak_quality_suppression_radius,
                            full_confidence_dominance=args.peak_full_confidence_dominance,
                        )
                        peak_quality_percentage = quality_report.percentage
                        ambiguous_keypoints = quality_report.ambiguous_keypoints

                    raw_pose = transform.pose_to_original(crop_pose)
                    presence_report = presence.update(raw_pose)
                    presence_active = presence_report.active or args.disable_presence_gate
                    presence_percentage = presence_report.percentage

                    if presence_report.lost and not args.disable_presence_gate:
                        _reset_tracking_state(bilateral, plausibility, tracker, smoother)

                    if presence_active:
                        identity_pose, bilateral_report = bilateral.update(raw_pose)
                        filtered_pose, report = plausibility.evaluate_and_filter(identity_pose)
                        pose_for_tracker = identity_pose if args.disable_plausibility_filter else filtered_pose
                        tracked_pose = tracker.update(pose_for_tracker)
                        pose = smoother.update(tracked_pose)

                        predicted_count = tracker.predicted_count
                        plausibility_percentage = report.percentage
                        leg_plausibility_percentage = report.leg_percentage
                        bilateral_percentage = bilateral_report.percentage
                        bilateral_collapses = bilateral_report.collapsed_pairs
                        bilateral_swapped = bilateral_report.swapped_chain
                        rejected_count = len(report.suspicious_indices | bilateral_report.rejected_indices)
                        calibrated_segments = report.calibrated_segments

                        draw_pose(frame, pose, confidence_threshold=args.render_confidence)
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
                    presence_active=presence_active,
                    presence_percentage=presence_percentage,
                    peak_quality_percentage=peak_quality_percentage,
                    ambiguous_keypoints=ambiguous_keypoints,
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
    parser.add_argument("--mode", choices=("demo", "model"), default="demo")
    parser.add_argument("--weights", default="models/pose_model.pt")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--render-confidence", type=float, default=0.04)

    parser.add_argument("--presence-confidence", type=float, default=0.06)
    parser.add_argument("--presence-min-keypoints", type=int, default=5)
    parser.add_argument("--presence-min-torso", type=int, default=2)
    parser.add_argument("--presence-acquire-frames", type=int, default=3)
    parser.add_argument("--presence-release-frames", type=int, default=6)
    parser.add_argument("--disable-presence-gate", action="store_true")

    parser.add_argument("--peak-quality-suppression-radius", type=int, default=4)
    parser.add_argument("--peak-full-confidence-dominance", type=float, default=0.35)

    parser.add_argument("--prediction-frames", type=int, default=8)
    parser.add_argument("--prediction-decay", type=float, default=0.82)
    parser.add_argument("--anatomy-weight", type=float, default=0.60)
    parser.add_argument("--smoothing-alpha", type=float, default=0.65)
    parser.add_argument("--plausibility-reject-threshold", type=float, default=0.16)
    parser.add_argument("--disable-plausibility-filter", action="store_true")

    parser.add_argument(
        "--enable-bilateral-multipeak",
        action="store_true",
        help="Experimental: permite promover 2º/3º pico quando pernas colapsam.",
    )
    parser.add_argument("--bilateral-top-k", type=int, default=3)
    parser.add_argument("--bilateral-suppression-radius", type=int, default=4)
    parser.add_argument("--bilateral-peak-separation", type=float, default=3.0)
    parser.add_argument("--bilateral-alternative-ratio", type=float, default=0.65)
    parser.add_argument("--bilateral-collapse-ratio", type=float, default=0.12)
    return parser.parse_args()


if __name__ == "__main__":
    main()
