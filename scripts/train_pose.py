"""Treinamento da primeira CNN de pose do trackingCorporal.

Exemplos:
    python -m scripts.train_pose --images data/coco/train2017 \
        --annotations data/coco/annotations/person_keypoints_train2017.json

    # Continua a partir de um checkpoint existente.
    python -m scripts.train_pose --resume models/pose_model.pt \
        --max-hours 6 --epochs 0 --batch-size 8

    # Refinamento focado em pernas e oclusões.
    python -m scripts.train_pose --resume models/pose_model.pt \
        --max-hours 6 --epochs 0 --batch-size 8 \
        --heatmap-positive-weight 8 --leg-keypoint-weight 2.0 \
        --occlusion-probability 0.35
"""

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


def masked_heatmap_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility_mask: torch.Tensor,
    positive_weight: float = 0.0,
    leg_keypoint_weight: float = 1.0,
) -> torch.Tensor:
    """MSE ponderado para articulações anotadas.

    ``positive_weight`` aumenta a importância dos pixels próximos ao pico do
    heatmap, evitando que o fundo quase todo zero domine a loss.

    ``leg_keypoint_weight`` dá peso adicional aos canais de quadril, joelho e
    tornozelo. O normalizador usa os mesmos pesos, mantendo a escala da loss
    aproximadamente estável mesmo quando as pernas recebem maior prioridade.
    """
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
    return weighted.sum() / normalizer


def format_duration(seconds: float) -> str:
    """Formata segundos como HH:MM:SS para os logs de treinamento."""
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
    """Carrega pesos e, quando disponível, estado do otimizador."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint para continuar não encontrado: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError("Checkpoint inválido: model_state não encontrado.")

    keypoint_count = int(checkpoint.get("keypoint_count", NUM_KEYPOINTS))
    if keypoint_count != NUM_KEYPOINTS:
        raise RuntimeError(
            f"Checkpoint possui {keypoint_count} keypoints; esperados {NUM_KEYPOINTS}."
        )

    checkpoint_input_size = int(checkpoint.get("input_size", expected_input_size))
    checkpoint_heatmap_size = int(
        checkpoint.get("heatmap_size", expected_heatmap_size)
    )
    if checkpoint_input_size != expected_input_size:
        raise RuntimeError(
            "O input-size do checkpoint não corresponde ao treino atual: "
            f"{checkpoint_input_size} != {expected_input_size}."
        )
    if checkpoint_heatmap_size != expected_heatmap_size:
        raise RuntimeError(
            "O heatmap-size do checkpoint não corresponde ao treino atual: "
            f"{checkpoint_heatmap_size} != {expected_heatmap_size}."
        )

    model.load_state_dict(checkpoint["model_state"])

    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        print("Estado do otimizador restaurado do checkpoint.")
    else:
        print(
            "Checkpoint antigo sem estado do otimizador: "
            "pesos restaurados e Adam reiniciado."
        )

    saved_epoch = int(checkpoint.get("epoch", 0))
    saved_global_step = int(checkpoint.get("global_step", 0))
    saved_batch = int(checkpoint.get("batch_index", 0))

    print(f"Continuando de: {path.resolve()}")
    print(
        f"Checkpoint anterior: epoch {saved_epoch}, "
        f"batch {saved_batch}, global_step {saved_global_step}."
    )

    return saved_epoch, saved_global_step


def train(args: argparse.Namespace) -> None:
    if args.epochs <= 0 and args.max_hours is None:
        raise ValueError("Use --epochs maior que 0 ou informe --max-hours.")
    if args.max_hours is not None and args.max_hours <= 0:
        raise ValueError("--max-hours deve ser maior que zero.")
    if args.heatmap_positive_weight < 0.0:
        raise ValueError("--heatmap-positive-weight não pode ser negativo.")
    if args.leg_keypoint_weight <= 0.0:
        raise ValueError("--leg-keypoint-weight deve ser maior que zero.")

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
        augmentation_seed=args.augmentation_seed,
    )
    print(f"Exemplos de pessoas carregados: {len(dataset)}")
    print(
        "Loss: peso de pico "
        f"{args.heatmap_positive_weight:.1f} | "
        f"peso de quadril/joelho/tornozelo {args.leg_keypoint_weight:.2f}."
    )

    if args.occlusion_probability > 0.0:
        print(
            "Oclusão artificial ativa: "
            f"probabilidade {args.occlusion_probability:.2f}, "
            f"tamanho {args.occlusion_min_size:.2f}-{args.occlusion_max_size:.2f} do recorte."
        )

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
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            device=device,
            expected_input_size=args.input_size,
            expected_heatmap_size=args.heatmap_size,
            learning_rate=args.learning_rate,
        )

    training_started = time.perf_counter()
    max_seconds = args.max_hours * 3600.0 if args.max_hours is not None else None
    session_epoch = 0
    current_epoch = completed_epoch

    if max_seconds is not None:
        print(
            "Limite de tempo desta sessão: "
            f"{format_duration(max_seconds)}. O modelo será salvo antes de encerrar."
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
                )
                loss.backward()
                optimizer.step()

                global_step += 1
                running_loss += float(loss.item())
                elapsed = time.perf_counter() - training_started

                if batch_index % args.log_every == 0 or batch_index == len(loader):
                    average = running_loss / batch_index
                    session_limit = str(args.epochs) if args.epochs > 0 else "sem-limite"
                    message = (
                        f"epoch total {current_epoch:02d} | "
                        f"sessao {session_epoch:02d}/{session_limit} | "
                        f"batch {batch_index:04d}/{len(loader):04d} | "
                        f"loss {average:.6f} | "
                        f"tempo {format_duration(elapsed)}"
                    )

                    if max_seconds is not None:
                        remaining = max(0.0, max_seconds - elapsed)
                        message += f" | restante {format_duration(remaining)}"

                    print(message)

                if max_seconds is not None and elapsed >= max_seconds:
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
                    )
                    print(
                        "Limite de tempo atingido. "
                        f"Checkpoint salvo em: {Path(args.output).resolve()}"
                    )
                    return

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                output_path=args.output,
                input_size=args.input_size,
                heatmap_size=args.heatmap_size,
                epoch=current_epoch,
                batch_index=len(loader),
                global_step=global_step,
                heatmap_positive_weight=args.heatmap_positive_weight,
                leg_keypoint_weight=args.leg_keypoint_weight,
            )

    except KeyboardInterrupt:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            output_path=args.output,
            input_size=args.input_size,
            heatmap_size=args.heatmap_size,
            epoch=current_epoch,
            batch_index=0,
            global_step=global_step,
            heatmap_positive_weight=args.heatmap_positive_weight,
            leg_keypoint_weight=args.leg_keypoint_weight,
        )
        print(
            "Treinamento interrompido pelo usuário. "
            f"Checkpoint salvo em: {Path(args.output).resolve()}"
        )
        return

    print(f"Modelo salvo em: {Path(args.output).resolve()}")


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
) -> None:
    """Salva pesos e estado de treino para permitir continuação posterior."""
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
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint existente cujos pesos devem ser usados para continuar o treino.",
    )
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.8)
    parser.add_argument("--min-keypoints", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Quantidade máxima de épocas desta sessão. Use 0 para deixar --max-hours controlar.",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Encerra esta sessão após aproximadamente esta quantidade de horas e salva o checkpoint.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--heatmap-positive-weight",
        type=float,
        default=8.0,
        help="Peso adicional para pixels próximos ao pico do keypoint. 0 reproduz a loss antiga.",
    )
    parser.add_argument(
        "--leg-keypoint-weight",
        type=float,
        default=1.0,
        help="Peso relativo dos canais de quadril, joelho e tornozelo. Use >1 para priorizar pernas.",
    )
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
        help="Limita exemplos para testes rápidos. Omitido = usa todos os exemplos disponíveis.",
    )
    parser.add_argument(
        "--occlusion-probability",
        type=float,
        default=0.0,
        help="Chance de esconder artificialmente uma região de cada pessoa durante o treino.",
    )
    parser.add_argument(
        "--occlusion-min-size",
        type=float,
        default=0.12,
        help="Menor lado relativo do retângulo de oclusão.",
    )
    parser.add_argument(
        "--occlusion-max-size",
        type=float,
        default=0.35,
        help="Maior lado relativo do retângulo de oclusão.",
    )
    parser.add_argument(
        "--augmentation-seed",
        type=int,
        default=None,
        help="Seed opcional para reproduzir as oclusões artificiais.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
