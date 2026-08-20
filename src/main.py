"""Ponto de entrada do trackingCorporal.

Modos atuais:
- demo: webcam real + pose artificial fixa;
- model: webcam real + CNN própria + tracking temporal + suavização.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.capture.camera import Camera
from src.pose.decoder import decode_heatmaps
from src.pose.demo_pose import build_demo_pose
from src.pose.keypoints import NUM_KEYPOINTS
from src.pose.model import PoseNet
from src.preprocessing.frame_preprocessor import preprocess_frame
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
) -> np.ndarray:
    """Exibe informações úteis da etapa atual sobre o frame."""
    if mode == "model":
        status = "POSE MODEL + TEMPORAL TRACKING"
    else:
        status = "POSE DEMO - NOT TRACKING"

    cv2.putText(
        frame,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    line_y = 105
    if mode == "model":
        cv2.putText(
            frame,
            f"Keypoints previstos: {predicted_count}",
            (20, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        line_y += 35

    cv2.putText(
        frame,
        "Q ou ESC para sair",
        (20, line_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
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
    tracker: TemporalPoseTracker | None = None
    smoother: ExponentialPoseSmoother | None = None

    if args.mode == "model":
        model, input_size = load_pose_model(args.weights, device)
        tracker = TemporalPoseTracker(
            detection_threshold=args.confidence,
            max_missing_frames=args.prediction_frames,
            confidence_decay=args.prediction_decay,
            anatomy_weight=args.anatomy_weight,
        )
        smoother = ExponentialPoseSmoother(alpha=args.smoothing_alpha)

        print(f"Modelo de pose carregado em {device}.")
        print(
            "Tracking temporal ativo: keypoints ocultos podem ser previstos por poucos frames."
        )
        print(
            f"Previsão: até {args.prediction_frames} frames | "
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
                print("A pose vem da CNN e é refinada pelo histórico temporal.")

            print("Pressione Q ou ESC na janela para encerrar.")

            while True:
                frame = camera.read()
                predicted_count = 0

                if args.mode == "model":
                    assert model is not None
                    assert tracker is not None
                    assert smoother is not None

                    model_input = preprocess_frame(
                        frame,
                        input_size=input_size,
                        device=device,
                    )
                    with torch.inference_mode():
                        heatmaps = model(model_input)

                    raw_pose = decode_heatmaps(heatmaps)
                    tracked_pose = tracker.update(raw_pose)
                    pose = smoother.update(tracked_pose)
                    predicted_count = tracker.predicted_count

                    draw_pose(
                        frame,
                        pose,
                        confidence_threshold=args.confidence,
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
