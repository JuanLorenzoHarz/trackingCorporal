"""Treinamento da PoseNet V2 multi-pessoa."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.coco_multiperson_dataset import CocoMultiPersonPoseDataset
from src.pose.model_v2 import PoseNetV2


def focal_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """Focal loss estilo CenterNet para heatmaps gaussianos."""
    probability = torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)
    positive = (target >= 0.999).to(logits.dtype)
    negative = (target < 0.999).to(logits.dtype)
    negative_weight = (1.0 - target).pow(beta)

    positive_loss = -torch.log(probability) * (1.0 - probability).pow(alpha) * positive
    negative_loss = (
        -torch.log(1.0 - probability)
        * probability.pow(alpha)
        * negative_weight
        * negative
    )

    positive_count = positive.sum()
    if positive_count > 0:
        return (positive_loss.sum() + negative_loss.sum()) / positive_count
    # Frames negativos são importantes: ensinam que nenhum centro/keypoint é válido.
    return negative_loss.mean()


def masked_offset_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 nos vetores 2D apenas onde existe supervisão."""
    if prediction.ndim != 4 or target.shape != prediction.shape:
        raise ValueError("prediction/target de offset devem ter o mesmo formato [B,2K,H,W].")
    batch, channels, height, width = prediction.shape
    if channels % 2 != 0:
        raise ValueError("Quantidade de canais de offset deve ser par.")
    keypoints = channels // 2
    if mask.shape != (batch, keypoints, height, width):
        raise ValueError("mask de offset possui formato incompatível.")

    pred = prediction.view(batch, keypoints, 2, height, width)
    tgt = target.view(batch, keypoints, 2, height, width)
    expanded_mask = mask.unsqueeze(2)
    error = F.smooth_l1_loss(pred, tgt, reduction="none") * expanded_mask
    denominator = expanded_mask.sum().clamp_min(1.0) * 2.0
    return error.sum() / denominator


def total_v2_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    center_weight: float = 1.0,
    keypoint_weight: float = 1.0,
    center_offset_weight: float = 0.25,
    parent_offset_weight: float = 0.35,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    center = focal_heatmap_loss(output["center"], target["center"])
    keypoints = focal_heatmap_loss(output["keypoints"], target["keypoints"])
    center_offsets = masked_offset_loss(
        output["center_offsets"],
        target["center_offsets"],
        target["center_offset_mask"],
    )
    parent_offsets = masked_offset_loss(
        output["parent_offsets"],
        target["parent_offsets"],
        target["parent_offset_mask"],
    )

    total = (
        center_weight * center
        + keypoint_weight * keypoints
        + center_offset_weight * center_offsets
        + parent_offset_weight * parent_offsets
    )
    return total, {
        "center": center.detach(),
        "keypoints": keypoints.detach(),
        "center_offsets": center_offsets.detach(),
        "parent_offsets": parent_offsets.detach(),
    }


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def save_checkpoint(
    model: PoseNetV2,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    input_size: int,
    heatmap_size: int,
    epoch: int,
    batch_index: int,
    global_step: int,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "PoseNetV2",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "input_size": input_size,
            "heatmap_size": heatmap_size,
            "epoch": epoch,
            "batch_index": batch_index,
            "global_step": global_step,
        },
        output_path,
    )


def load_v2_checkpoint(
    path: str | Path,
    model: PoseNetV2,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    learning_rate: float,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "PoseNetV2":
        raise RuntimeError("Checkpoint não é PoseNetV2.")
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def train(args: argparse.Namespace) -> None:
    if args.epochs <= 0 and args.max_hours is None:
        raise ValueError("Use --epochs > 0 ou --max-hours.")
    if args.max_hours is not None and args.max_hours <= 0:
        raise ValueError("--max-hours deve ser positivo.")
    if args.resume and args.init_v1:
        raise ValueError("Use --resume OU --init-v1, não os dois.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    dataset = CocoMultiPersonPoseDataset(
        images_dir=args.images,
        annotations_file=args.annotations,
        input_size=args.input_size,
        heatmap_size=args.heatmap_size,
        min_keypoints=args.min_keypoints,
        max_people=args.max_people,
        max_samples=args.max_samples,
        center_sigma=args.center_sigma,
        keypoint_sigma=args.keypoint_sigma,
    )
    print(f"Imagens full-frame carregadas: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = PoseNetV2().to(device)
    if args.init_v1:
        copied = model.initialize_from_v1(args.init_v1)
        print(f"Transferência V1 -> V2: {copied} tensores compatíveis copiados.")
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    completed_epoch = 0
    global_step = 0
    if args.resume:
        completed_epoch, global_step = load_v2_checkpoint(
            args.resume,
            model,
            optimizer,
            device,
            args.learning_rate,
        )
        print(f"Continuando V2 de {Path(args.resume).resolve()}.")

    started = time.perf_counter()
    max_seconds = args.max_hours * 3600.0 if args.max_hours is not None else None
    session_epoch = 0
    current_epoch = completed_epoch

    try:
        while True:
            session_epoch += 1
            if args.epochs > 0 and session_epoch > args.epochs:
                break
            current_epoch += 1
            model.train()
            running_total = 0.0
            running_parts = {name: 0.0 for name in ("center", "keypoints", "center_offsets", "parent_offsets")}

            for batch_index, (images, targets) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                targets = {
                    name: tensor.to(device, non_blocking=True)
                    for name, tensor in targets.items()
                }

                optimizer.zero_grad(set_to_none=True)
                output = model(images)
                loss, parts = total_v2_loss(
                    output,
                    targets,
                    center_weight=args.center_loss_weight,
                    keypoint_weight=args.keypoint_loss_weight,
                    center_offset_weight=args.center_offset_loss_weight,
                    parent_offset_weight=args.parent_offset_loss_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                global_step += 1
                running_total += float(loss.item())
                for name in running_parts:
                    running_parts[name] += float(parts[name].item())

                elapsed = time.perf_counter() - started
                if batch_index % args.log_every == 0 or batch_index == len(loader):
                    divisor = batch_index
                    remaining_text = ""
                    if max_seconds is not None:
                        remaining_text = f" | restante {format_duration(max(0.0, max_seconds - elapsed))}"
                    print(
                        f"epoch {current_epoch:02d} | batch {batch_index:05d}/{len(loader):05d} | "
                        f"loss {running_total/divisor:.5f} | "
                        f"ctr {running_parts['center']/divisor:.4f} | "
                        f"kp {running_parts['keypoints']/divisor:.4f} | "
                        f"c-off {running_parts['center_offsets']/divisor:.4f} | "
                        f"p-off {running_parts['parent_offsets']/divisor:.4f} | "
                        f"tempo {format_duration(elapsed)}{remaining_text}"
                    )

                if max_seconds is not None and elapsed >= max_seconds:
                    save_checkpoint(
                        model,
                        optimizer,
                        args.output,
                        args.input_size,
                        args.heatmap_size,
                        current_epoch,
                        batch_index,
                        global_step,
                    )
                    print(f"Limite atingido. V2 salva em: {Path(args.output).resolve()}")
                    return

            save_checkpoint(
                model,
                optimizer,
                args.output,
                args.input_size,
                args.heatmap_size,
                current_epoch,
                len(loader),
                global_step,
            )
    except KeyboardInterrupt:
        save_checkpoint(
            model,
            optimizer,
            args.output,
            args.input_size,
            args.heatmap_size,
            current_epoch,
            0,
            global_step,
        )
        print(f"Interrompido; checkpoint salvo em: {Path(args.output).resolve()}")
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina PoseNet V2 multi-pessoa.")
    parser.add_argument("--images", default="data/coco/train2017")
    parser.add_argument("--annotations", default="data/coco/annotations/person_keypoints_train2017.json")
    parser.add_argument("--output", default="models/pose_model_v2.pt")
    parser.add_argument("--resume", default=None, help="Checkpoint PoseNetV2 para continuar.")
    parser.add_argument("--init-v1", default=None, help="Checkpoint V1 para transferir backbone/decoder compatível.")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--min-keypoints", type=int, default=4)
    parser.add_argument("--max-people", type=int, default=12)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--center-sigma", type=float, default=2.0)
    parser.add_argument("--keypoint-sigma", type=float, default=1.8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--center-loss-weight", type=float, default=1.0)
    parser.add_argument("--keypoint-loss-weight", type=float, default=1.0)
    parser.add_argument("--center-offset-loss-weight", type=float, default=0.25)
    parser.add_argument("--parent-offset-loss-weight", type=float, default=0.35)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
