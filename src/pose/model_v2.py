"""PoseNet V2: pose multi-pessoa com associação explícita de articulações.

A V2 mantém a filosofia do projeto: CNN própria, sem backbone pré-treinado e
sem detector de pose pronto. A rede recebe a imagem inteira e produz:

- ``center``: heatmap de centros de pessoas;
- ``keypoints``: 17 heatmaps semânticos de articulações;
- ``center_offsets``: para cada keypoint, vetor keypoint -> centro da pessoa;
- ``parent_offsets``: para cada keypoint, vetor keypoint -> articulação pai.

Os center offsets impedem que mãos/pés de pessoas diferentes sejam agrupados no
mesmo esqueleto. Os parent offsets dão ao decoder informação explícita sobre a
cadeia anatômica e ajudam a evitar os "X" causados por punho/tornozelo trocado.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.pose.keypoints import NUM_KEYPOINTS
from src.pose.model import ConvBlock


class DownsampleConvBlock(nn.Sequential):
    """Mesmo formato de pesos da V1, mas reduz resolução na primeira conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class HeatmapHead(nn.Sequential):
    """Cabeça pequena para logits de heatmap."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        # Prior pequeno de ativação evita começar vendo pessoas em todo lugar.
        nn.init.constant_(self[-1].bias, -2.19)


class OffsetHead(nn.Sequential):
    """Cabeça de regressão de vetores 2D."""

    def __init__(self, in_channels: int, vector_count: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, vector_count * 2, kernel_size=1),
        )
        nn.init.zeros_(self[-1].bias)


class PoseNetV2(nn.Module):
    """Encoder-decoder eficiente com heads bottom-up para múltiplas pessoas."""

    def __init__(self, keypoint_count: int = NUM_KEYPOINTS) -> None:
        super().__init__()
        self.keypoint_count = keypoint_count

        # A V1 fazia duas convs em 256x256 antes de cada pooling. Na V2 a
        # primeira conv de cada bloco já reduz resolução, economizando bastante CPU.
        self.encoder1 = DownsampleConvBlock(3, 32)   # 256 -> 128
        self.encoder2 = DownsampleConvBlock(32, 64)  # 128 -> 64
        self.encoder3 = DownsampleConvBlock(64, 128) # 64 -> 32

        # Mantém formato de pesos da V1 para permitir transferência.
        self.bottleneck = ConvBlock(128, 256)
        self.upsample = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # Skip de 64x64: devolve ao decoder detalhe espacial que a V1 perdia.
        self.skip64 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.refine = ConvBlock(128, 64)

        self.center_head = HeatmapHead(64, 1)
        self.keypoint_head = HeatmapHead(64, keypoint_count)
        self.center_offset_head = OffsetHead(64, keypoint_count)
        self.parent_offset_head = OffsetHead(64, keypoint_count)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.encoder1(image)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        x = self.bottleneck(e3)
        x = self.upsample(x)
        x = x + self.skip64(e2)
        features = self.refine(x)

        return {
            "center": self.center_head(features),
            "keypoints": self.keypoint_head(features),
            "center_offsets": self.center_offset_head(features),
            "parent_offsets": self.parent_offset_head(features),
        }

    def initialize_from_v1(self, checkpoint_path: str | Path) -> int:
        """Transfere todos os tensores V1 compatíveis para backbone/decoder.

        Os heads V2 são propositalmente novos: a semântica da saída mudou de
        uma única pose para detecção multi-pessoa. Retorna o número de tensores
        copiados para facilitar diagnóstico no script de treino.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise RuntimeError("Checkpoint V1 inválido: model_state não encontrado.")

        source = checkpoint["model_state"]
        target = self.state_dict()
        transferable_prefixes = (
            "encoder1.",
            "encoder2.",
            "encoder3.",
            "bottleneck.",
            "upsample.",
            "refine.",
        )

        copied = 0
        for name, tensor in source.items():
            if not name.startswith(transferable_prefixes):
                continue
            if name not in target or target[name].shape != tensor.shape:
                continue
            target[name] = tensor.detach().clone()
            copied += 1

        self.load_state_dict(target)
        return copied
