"""Webcam em tempo real usando PoseNet V2 multi-pessoa."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.capture.camera import Camera
from src.pose.decoder_v2 import decode_pose_v2
from src.pose.model_v2 import PoseNetV2
from src.preprocessing.frame_preprocessor_v2 import preprocess_frame_v2
from src.tracking.multiperson_tracker import MultiPersonTemporalTracker, TrackedPerson
from src.visualization.renderer import draw_pose, normalized_to_pixel


WINDOW_NAME = "trackingCorporal V2"


def load_model_v2(path: str | Path, device: torch.device) -> tuple[PoseNetV2, int]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        smoke_path = Path("models/pose_model_v2_smoke.pt")
        hint = ""
        if smoke_path.is_file():
            hint = (
                "\nO checkpoint smoke existe. Para testá-lo use: "
                "python -m src.main_v2 --weights models/pose_model_v2_smoke.pt"
            )
        raise FileNotFoundError(
            f"Checkpoint V2 não encontrado: {checkpoint_path.resolve()}.{hint}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != "PoseNetV2":
        raise RuntimeError(
            f"Checkpoint informado não é PoseNetV2: {checkpoint_path.resolve()}"
        )
    if "model_state" not in checkpoint:
        raise RuntimeError("Checkpoint V2 inválido: model_state não encontrado.")

    model = PoseNetV2().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, int(checkpoint.get("input_size", 256))


def _person_label_position(person: TrackedPerson, frame: np.ndarray) -> tuple[int, int]:
    valid = [point for point in person.pose.keypoints if point.confidence > 0.12]
    if not valid:
        return 20, 20
    center_x = sum(point.x for point in valid) / len(valid)
    center_y = min(point.y for point in valid)
    height, width = frame.shape[:2]
    return normalized_to_pixel(center_x, max(0.0, center_y - 0.04), width, height)


def draw_status(
    frame: np.ndarray,
    fps: float,
    person_count: int,
    assigned_candidates: int,
    rejected_bilateral: int,
    suppressed_duplicates: int,
    corrected_torso_swaps: int,
) -> None:
    lines = (
        "POSENET V2 - MULTI-PESSOA",
        f"FPS: {fps:.1f}",
        f"Pessoas: {person_count}",
        f"Candidatos associados: {assigned_candidates}",
        f"Duplicatas de pessoa fundidas: {suppressed_duplicates}",
        f"Trocas de tronco corrigidas: {corrected_torso_swaps}",
        f"L/R rejeitados para evitar X: {rejected_bilateral}",
        "Q ou ESC para sair",
    )
    y = 30
    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 if index else 0.62,
            (255, 255, 255),
            2 if index <= 1 else 1,
            cv2.LINE_AA,
        )
        y += 27


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, input_size = load_model_v2(args.weights, device)
    tracker = MultiPersonTemporalTracker(
        detection_threshold=args.render_confidence,
        match_distance=args.track_match_distance,
        stale_frames=args.track_stale_frames,
        prediction_frames=args.prediction_frames,
        prediction_decay=args.prediction_decay,
        anatomy_weight=args.anatomy_weight,
        smoothing_alpha=args.smoothing_alpha,
    )

    print(f"PoseNet V2 carregada em {device}.")
    print(
        "A V2 usa full-frame letterbox, centros de pessoa, associação keypoint->centro, "
        "keypoint->pai anatômico e fusão de duplicatas por sobreposição semântica."
    )

    previous_time = time.perf_counter()

    try:
        with Camera(index=args.camera, width=args.width, height=args.height) as camera:
            print(f"Câmera aberta em {camera.resolution[0]}x{camera.resolution[1]}.")

            while True:
                frame = camera.read()
                model_input, transform = preprocess_frame_v2(
                    frame,
                    input_size=input_size,
                    device=device,
                )

                with torch.inference_mode():
                    output = model(model_input)

                crop_poses, report = decode_pose_v2(
                    output,
                    center_threshold=args.center_threshold,
                    keypoint_threshold=args.keypoint_threshold,
                    max_people=args.max_people,
                    candidates_per_keypoint=args.candidates_per_keypoint,
                    association_radius=args.association_radius,
                    limb_association_factor=args.limb_association_factor,
                    extremity_association_factor=args.extremity_association_factor,
                    parent_sigma=args.parent_sigma,
                    extremity_parent_sigma=args.extremity_parent_sigma,
                    bilateral_min_separation=args.bilateral_min_separation,
                    duplicate_center_radius=args.duplicate_center_radius,
                    duplicate_joint_distance=args.duplicate_joint_distance,
                    duplicate_overlap_ratio=args.duplicate_overlap_ratio,
                )
                original_poses = [transform.pose_to_original(pose) for pose in crop_poses]
                people = tracker.update(original_poses)

                for person in people:
                    draw_pose(
                        frame,
                        person.pose,
                        confidence_threshold=args.render_confidence,
                    )
                    label_position = _person_label_position(person, frame)
                    cv2.putText(
                        frame,
                        f"ID {person.track_id}",
                        label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                current_time = time.perf_counter()
                elapsed = current_time - previous_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                previous_time = current_time

                draw_status(
                    frame,
                    fps=fps,
                    person_count=len(people),
                    assigned_candidates=report.assigned_candidates,
                    rejected_bilateral=report.rejected_bilateral_points,
                    suppressed_duplicates=report.suppressed_duplicate_people,
                    corrected_torso_swaps=report.corrected_torso_swaps,
                )
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
    finally:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoseNet V2 multi-pessoa em tempo real.")
    parser.add_argument("--weights", default="models/pose_model_v2.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--center-threshold",
        type=float,
        default=0.20,
        help="Limiar do centro de pessoa. Perfil pode produzir centros menos intensos.",
    )
    parser.add_argument("--keypoint-threshold", type=float, default=0.12)
    parser.add_argument("--render-confidence", type=float, default=0.10)
    parser.add_argument("--max-people", type=int, default=6)
    parser.add_argument("--candidates-per-keypoint", type=int, default=12)
    parser.add_argument("--association-radius", type=float, default=7.0)
    parser.add_argument(
        "--limb-association-factor",
        type=float,
        default=1.30,
        help="Multiplicador do raio para cotovelos/joelhos.",
    )
    parser.add_argument(
        "--extremity-association-factor",
        type=float,
        default=1.65,
        help="Multiplicador do raio para punhos/tornozelos.",
    )
    parser.add_argument("--parent-sigma", type=float, default=4.0)
    parser.add_argument(
        "--extremity-parent-sigma",
        type=float,
        default=4.0,
        help="Tolerância do vetor punho->cotovelo / tornozelo->joelho.",
    )
    parser.add_argument("--bilateral-min-separation", type=float, default=1.5)
    parser.add_argument(
        "--duplicate-center-radius",
        type=float,
        default=6.0,
        help="Distância máxima entre centros candidatos à fusão; não funde sem sobreposição de keypoints.",
    )
    parser.add_argument(
        "--duplicate-joint-distance",
        type=float,
        default=0.05,
        help="Distância normalizada para considerar dois keypoints semânticos sobrepostos.",
    )
    parser.add_argument("--duplicate-overlap-ratio", type=float, default=0.60)
    parser.add_argument("--track-match-distance", type=float, default=0.22)
    parser.add_argument("--track-stale-frames", type=int, default=5)
    parser.add_argument("--prediction-frames", type=int, default=6)
    parser.add_argument("--prediction-decay", type=float, default=0.80)
    parser.add_argument("--anatomy-weight", type=float, default=0.55)
    parser.add_argument("--smoothing-alpha", type=float, default=0.68)
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
