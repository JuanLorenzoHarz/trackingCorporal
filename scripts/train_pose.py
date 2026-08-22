"""Treinamento da CNN de pose do trackingCorporal."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.coco_pose_dataset import CocoPoseDataset
from src.pose.keypoints import BodyKeypoint, NUM_KEYPOINTS
from src.pose.model import PoseNet


LEG_KEYPOINT_INDICES: tuple[int, ...] = (
    int(BodyKeypoint.LEFT_HIP),
    int(BodyKeypoint.RIGHT_HIP),
    int(BodyKeypoint.LEFT_KNEE),
    int(BodyKeypoint.RIGHT_KNEE),
    int(BodyKeypoint.LEFT_ANKLE),
    int(BodyKeypoint.RIGHT_ANKLE),
)

BILATERAL_LEG_PAIRS: tuple[tuple[int, int], ...] = (
    (int(BodyKeypoint.LEFT_HIP), int(BodyKeypoint.RIGHT_HIP)),
    (int(BodyKeypoint.LEFT_KNEE), int(BodyKeypoint.RIGHT_KNEE)),
    (int(BodyKeypoint.LEFT_ANKLE), int(BodyKeypoint.RIGHT_ANKLE)),
)


def bilateral_cross_peak_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility_mask: torch.Tensor,
    minimum_target_distance: float = 3.0,
) -> torch.Tensor:
    """Penaliza resposta L sobre alvo R e vice-versa quando alvos estão separados."""
    if minimum_target_distance < 0.0:
        raise ValueError("minimum_target_distance não pode ser negativo.")
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("prediction e target devem ter formato [B,K,H,W].")

    _, channel_count, _, width = target.shape
    total = prediction.new_tensor(0.0)
    term_count = prediction.new_tensor(0.0)

    for left_index, right_index in BILATERAL_LEG_PAIRS:
        if left_index >= channel_count or right_index >= channel_count:
            continue

        left_visible = visibility_mask[:, left_index, 0, 0] > 0
        right_visible = visibility_mask[:, right_index, 0, 0] > 0
        # Negativos usam máscara 1 mas target totalmente zero; precisam ser
        # excluídos deste termo bilateral específico.
        left_has_target = target[:, left_index].flatten(1).amax(dim=1) > 0
        right_has_target = target[:, right_index].flatten(1).amax(dim=1) > 0
        both_visible = left_visible & right_visible & left_has_target & right_has_target
        if not bool(both_visible.any()):
            continue

        left_flat = target[:, left_index].flatten(1).argmax(dim=1)
        right_flat = target[:, right_index].flatten(1).argmax(dim=1)
        left_y = torch.div(left_flat, width, rounding_mode="floor").float()
        left_x = (left_flat % width).float()
        right_y = torch.div(right_flat, width, rounding_mode="floor").float()
        right_x = (right_flat % width).float()
        separation = torch.sqrt((left_x - right_x).square() + (left_y - right_y).square())
        eligible = both_visible & (separation >= minimum_target_distance)
        if not bool(eligible.any()):
            continue

        left_wrong = (
            prediction[:, left_index].square() * target[:, right_index]
        ).sum(dim=(1, 2)) / target[:, right_index].sum(dim=(1, 2)).clamp_min(1e-6)
        right_wrong = (
            prediction[:, right_index].square() * target[:, left_index]
        ).sum(dim=(1, 2)) / target[:, left_index].sum(dim=(1, 2)).clamp_min(1e-6)

        total = total + left_wrong[eligible].sum() + right_wrong[eligible].sum()
        term_count = term_count + eligible.sum().to(prediction.dtype) * 2.0

    if float(term_count.item()) == 0.0:
        return prediction.sum() * 0.0
    return total / term_count


def masked_heatmap_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility_mask: torch.Tensor,
    positive_weight: float = 0.0,
    leg_keypoint_weight: float = 1.0,
    bilateral_loss_weight: float = 0.0,
    bilateral_min_target_distance: float = 3.0,
) -> torch.Tensor:
    """MSE ponderado, incluindo negativos supervisionados para heatmap zero."""
    squared_error = F.mse_loss(prediction, target, reduction="none")
    pixel_weights = 1.0 + target * positive_weight

    channel_weights = torch.ones(
        (1, prediction.shape[1], 1, 1),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    for index in LEG_KEYPOINT_INDICES:
        if index < prediction.shape[1]:
            channel_weights[:, index] = leg_keypoint_weight

    effective_weights = pixel_weights * visibility_mask * channel_weights
    weighted = squared_error * effective_weights
    normalizer = effective_weights.sum().clamp_min(1.0)
    base_loss = weighted.sum() / normalizer

    if bilateral_loss_weight <= 0.0:
        return base_loss

    bilateral = bilateral_cross_peak_loss(
        prediction,
        target,
        visibility_mask,
        minimum_target_distance=bilateral_min_target_distance,
    )
    return base_loss + bilateral_loss_weight * bilateral


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_checkpoint_for_training(
    checkpoint_path: str | Path,
    model: PoseNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    expected_input_size: int,
    expected_heatmap_size: int,
    learning_rate: float,
) -> tuple[int, int]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint para continuar não encontrado: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError("Checkpoint inválido: model_state não encontrado.")

    if int(checkpoint.get("keypoint_count", NUM_KEYPOINTS)) != NUM_KEYPOINTS:
        raise RuntimeError("Quantidade de keypoints do checkpoint é incompatível.")
    if int(checkpoint.get("input_size", expected_input_size)) != expected_input_size:
        raise RuntimeError("input-size do checkpoint é incompatível.")
    if int(checkpoint.get("heatmap_size", expected_heatmap_size)) != expected_heatmap_size:
        raise RuntimeError("heatmap-size do checkpoint é incompatível.")

    model.load_state_dict(checkpoint["model_state"])
    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        print("Estado do otimizador restaurado do checkpoint.")
    else:
        print("Checkpoint antigo sem estado do otimizador: pesos restaurados e Adam reiniciado.")

    saved_epoch = int(checkpoint.get("epoch", 0))
    saved_global_step = int(checkpoint.get("global_step", 0))
    saved_batch = int(checkpoint.get("batch_index", 0))
    print(f"Continuando de: {path.resolve()}")
    print(f"Checkpoint anterior: epoch {saved_epoch}, batch {saved_batch}, global_step {saved_global_step}.")
    return saved_epoch, saved_global_step


def save_checkpoint(
    model: PoseNet,
    optimizer: torch.optim.Optimizer,
    output_path: str | Path,
    input_size: int,
    heatmap_size: int,
    epoch: int,
    batch_index: int = 0,
    global_step: int = 0,
    heatmap_positive_weight: float = 0.0,
    leg_keypoint_weight: float = 1.0,
    bilateral_loss_weight: float = 0.0,
    negative_sample_probability: float = 0.0,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "input_size": input_size,
            "heatmap_size": heatmap_size,
            "keypoint_count": NUM_KEYPOINTS,
            "epoch": epoch,
            "batch_index": batch_index,
            "global_step": global_step,
            "heatmap_positive_weight": heatmap_positive_weight,
            "leg_keypoint_weight": leg_keypoint_weight,
            "bilateral_loss_weight": bilateral_loss_weight,
            "negative_sample_probability": negative_sample_probability,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    if args.epochs <= 0 and args.max_hours is None:
        raise ValueError("Use --epochs maior que 0 ou informe --max-hours.")
    if args.max_hours is not None and args.max_hours <= 0:
        raise ValueError("--max-hours deve ser maior que zero.")
    if args.heatmap_positive_weight < 0.0:
        raise ValueError("--heatmap-positive-weight não pode ser negativo.")
    if args.leg_keypoint_weight <= 0.0:
        raise ValueError("--leg-keypoint-weight deve ser maior que zero.")
    if args.bilateral_loss_weight < 0.0:
        raise ValueError("--bilateral-loss-weight não pode ser negativo.")
    if not 0.0 <= args.negative_sample_probability <= 1.0:
        raise ValueError("--negative-sample-probability deve estar entre 0 e 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de treino: {device}")

    dataset = CocoPoseDataset(
        images_dir=args.images,
        annotations_file=args.annotations,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        min_keypoints=args.min_keypoints,
        max_samples=args.max_samples,
        occlusion_probability=args.occlusion_probability,
        occlusion_min_size=args.occlusion_min_size,
        occlusion_max_size=args.occlusion_max_size,
        negative_sample_probability=args.negative_sample_probability,
        negative_max_person_overlap=args.negative_max_person_overlap,
        augmentation_seed=args.augmentation_seed,
    )
    print(f"Exemplos-base carregados: {len(dataset)}")
    print(
        f"Loss: pico {args.heatmap_positive_weight:.1f} | "
        f"pernas {args.leg_keypoint_weight:.2f} | bilateral {args.bilateral_loss_weight:.3f}."
    )
    if args.negative_sample_probability > 0.0:
        print(
            f"Negativos sem pessoa ativos: probabilidade {args.negative_sample_probability:.2f} | "
            f"sobreposição máxima {args.negative_max_person_overlap:.3f}."
        )
    if args.occlusion_probability > 0.0:
        print(f"Oclusão artificial ativa: probabilidade {args.occlusion_probability:.2f}.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = PoseNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    global_step = 0
    completed_epoch = 0
    if args.resume is not None:
        completed_epoch, global_step = load_checkpoint_for_training(
            args.resume,
            model,
            optimizer,
            device,
            args.input_size,
            args.heatmap_size,
            args.learning_rate,
        )

    training_started = time.perf_counter()
    max_seconds = args.max_hours * 3600.0 if args.max_hours is not None else None
    session_epoch = 0
    current_epoch = completed_epoch

    if max_seconds is not None:
        print(f"Limite de tempo desta sessão: {format_duration(max_seconds)}. O modelo será salvo antes de encerrar.")

    def save(batch_index: int) -> None:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            output_path=args.output,
            input_size=args.input_size,
            heatmap_size=args.heatmap_size,
            epoch=current_epoch,
            batch_index=batch_index,
            global_step=global_step,
            heatmap_positive_weight=args.heatmap_positive_weight,
            leg_keypoint_weight=args.leg_keypoint_weight,
            bilateral_loss_weight=args.bilateral_loss_weight,
            negative_sample_probability=args.negative_sample_probability,
        )

    try:
        while True:
            session_epoch += 1
            if args.epochs > 0 and session_epoch > args.epochs:
                break
            current_epoch += 1
            model.train()
            running_loss = 0.0

            for batch_index, (images, targets, visibility) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                visibility = visibility.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                prediction = model(images)
                loss = masked_heatmap_loss(
                    prediction,
                    targets,
                    visibility,
                    positive_weight=args.heatmap_positive_weight,
                    leg_keypoint_weight=args.leg_keypoint_weight,
                    bilateral_loss_weight=args.bilateral_loss_weight,
                    bilateral_min_target_distance=args.bilateral_min_target_distance,
                )
                loss.backward()
                optimizer.step()

                global_step += 1
                running_loss += float(loss.item())
                elapsed = time.perf_counter() - training_started

                if batch_index % args.log_every == 0 or batch_index == len(loader):
                    average = running_loss / batch_index
                    limit = str(args.epochs) if args.epochs > 0 else "sem-limite"
                    message = (
                        f"epoch total {current_epoch:02d} | sessao {session_epoch:02d}/{limit} | "
                        f"batch {batch_index:04d}/{len(loader):04d} | loss {average:.6f} | "
                        f"tempo {format_duration(elapsed)}"
                    )
                    if max_seconds is not None:
                        message += f" | restante {format_duration(max(0.0, max_seconds - elapsed))}"
                    print(message)

                if max_seconds is not None and elapsed >= max_seconds:
                    save(batch_index)
                    print(f"Limite de tempo atingido. Checkpoint salvo em: {Path(args.output).resolve()}")
                    return

            save(len(loader))

    except KeyboardInterrupt:
        save(0)
        print(f"Treinamento interrompido. Checkpoint salvo em: {Path(args.output).resolve()}")
        return

    print(f"Modelo salvo em: {Path(args.output).resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina a CNN de pose.")
    parser.add_argument("--images", default="data/coco/train2017")
    parser.add_argument("--annotations", default="data/coco/annotations/person_keypoints_train2017.json")
    parser.add_argument("--output", default="models/pose_model.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.8)
    parser.add_argument("--min-keypoints", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--heatmap-positive-weight", type=float, default=8.0)
    parser.add_argument("--leg-keypoint-weight", type=float, default=1.0)
    parser.add_argument("--bilateral-loss-weight", type=float, default=0.0)
    parser.add_argument("--bilateral-min-target-distance", type=float, default=3.0)
    parser.add_argument("--negative-sample-probability", type=float, default=0.0)
    parser.add_argument("--negative-max-person-overlap", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--occlusion-probability", type=float, default=0.0)
    parser.add_argument("--occlusion-min-size", type=float, default=0.12)
    parser.add_argument("--occlusion-max-size", type=float, default=0.35)
    parser.add_argument("--augmentation-seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
