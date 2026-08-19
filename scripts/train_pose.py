"""Treinamento da primeira CNN de pose do trackingCorporal.

Exemplo:
    python -m scripts.train_pose --images data/coco/train2017 \
        --annotations data/coco/annotations/person_keypoints_train2017.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.coco_pose_dataset import CocoPoseDataset
from src.pose.keypoints import NUM_KEYPOINTS
from src.pose.model import PoseNet


def masked_heatmap_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE apenas para articulações anotadas no exemplo."""
    squared_error = F.mse_loss(prediction, target, reduction="none")
    weighted = squared_error * visibility_mask

    visible_points = visibility_mask.sum().clamp_min(1.0)
    pixels_per_heatmap = target.shape[-1] * target.shape[-2]
    return weighted.sum() / (visible_points * pixels_per_heatmap)


def train(args: argparse.Namespace) -> None:
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
    )
    print(f"Exemplos de pessoas carregados: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = PoseNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_index, (images, targets, visibility) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            visibility = visibility.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(images)
            loss = masked_heatmap_loss(prediction, targets, visibility)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

            if batch_index % args.log_every == 0 or batch_index == len(loader):
                average = running_loss / batch_index
                print(
                    f"epoch {epoch:02d}/{args.epochs:02d} | "
                    f"batch {batch_index:04d}/{len(loader):04d} | "
                    f"loss {average:.6f}"
                )

        save_checkpoint(
            model,
            args.output,
            input_size=args.input_size,
            heatmap_size=args.heatmap_size,
            epoch=epoch,
        )

    print(f"Modelo salvo em: {Path(args.output).resolve()}")


def save_checkpoint(
    model: PoseNet,
    output_path: str | Path,
    input_size: int,
    heatmap_size: int,
    epoch: int,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_size": input_size,
            "heatmap_size": heatmap_size,
            "keypoint_count": NUM_KEYPOINTS,
            "epoch": epoch,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina a primeira CNN de pose.")
    parser.add_argument("--images", default="data/coco/train2017")
    parser.add_argument(
        "--annotations",
        default="data/coco/annotations/person_keypoints_train2017.json",
    )
    parser.add_argument("--output", default="models/pose_model.pt")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.8)
    parser.add_argument("--min-keypoints", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 é o valor mais simples/seguro no Windows para começar.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limita exemplos para testes rápidos do pipeline.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
