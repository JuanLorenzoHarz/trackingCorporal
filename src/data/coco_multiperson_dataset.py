"""COCO full-frame para treinamento multi-pessoa da PoseNet V2.

Cada exemplo é a imagem inteira. Todas as pessoas anotadas contribuem para os
mesmos mapas; imagens sem pessoa utilizável viram negativos naturais. O flip
horizontal troca também a semântica left/right e o sinal X dos vetores.

Opcionalmente, uma compressão horizontal moderada da cena cria exemplos de
silhueta estreita para aumentar a robustez a pessoas meio ou completamente de
perfil. A transformação é aplicada à imagem e a todos os targets de forma
consistente.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.pose.keypoints import NUM_KEYPOINTS
from src.pose.structure_v2 import BILATERAL_PAIRS, PARENT_BY_KEYPOINT


FLIP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (3, 4),
    *BILATERAL_PAIRS,
)


class CocoMultiPersonPoseDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        annotations_file: str | Path,
        input_size: int = 256,
        heatmap_size: int = 64,
        min_keypoints: int = 4,
        max_people: int = 12,
        max_samples: int | None = None,
        center_sigma: float = 2.0,
        keypoint_sigma: float = 1.8,
        horizontal_flip_probability: float = 0.5,
        profile_squeeze_probability: float = 0.30,
        profile_squeeze_min: float = 0.60,
        profile_squeeze_max: float = 0.88,
        augmentation_seed: int | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.min_keypoints = min_keypoints
        self.max_people = max_people
        self.center_sigma = center_sigma
        self.keypoint_sigma = keypoint_sigma
        self.horizontal_flip_probability = horizontal_flip_probability
        self.profile_squeeze_probability = profile_squeeze_probability
        self.profile_squeeze_min = profile_squeeze_min
        self.profile_squeeze_max = profile_squeeze_max
        self._rng = np.random.default_rng(augmentation_seed)

        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability deve estar entre 0 e 1.")
        if not 0.0 <= profile_squeeze_probability <= 1.0:
            raise ValueError("profile_squeeze_probability deve estar entre 0 e 1.")
        if not 0.0 < profile_squeeze_min <= profile_squeeze_max <= 1.0:
            raise ValueError(
                "profile_squeeze_min/max devem satisfazer 0 < min <= max <= 1."
            )
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Pasta COCO não encontrada: {self.images_dir}")
        if not self.annotations_file.is_file():
            raise FileNotFoundError(f"Anotações COCO não encontradas: {self.annotations_file}")
        if input_size <= 0 or heatmap_size <= 1:
            raise ValueError("input_size/heatmap_size inválidos.")
        if max_people < 1:
            raise ValueError("max_people deve ser pelo menos 1.")

        with self.annotations_file.open("r", encoding="utf-8") as file:
            coco = json.load(file)

        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            if annotation.get("category_id") != 1 or annotation.get("iscrowd", 0):
                continue
            raw_keypoints = annotation.get("keypoints")
            if raw_keypoints is None or len(raw_keypoints) != NUM_KEYPOINTS * 3:
                continue
            if int(annotation.get("num_keypoints", 0)) < min_keypoints:
                continue
            bbox = annotation.get("bbox")
            if bbox is None or bbox[2] <= 2 or bbox[3] <= 2:
                continue
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        images = list(coco.get("images", []))
        images.sort(key=lambda item: int(item["id"]))
        if max_samples is not None:
            images = images[:max_samples]
        if not images:
            raise RuntimeError("Nenhuma imagem encontrada nas anotações COCO.")

        self.images = images
        self.annotations_by_image = annotations_by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        info = self.images[index]
        image_path = self.images_dir / info["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Não foi possível ler: {image_path}")

        canvas, scale, pad_x, pad_y = self._letterbox(image)

        profile_scale_x = 1.0
        profile_pad_x = 0.0
        if self._rng.random() < self.profile_squeeze_probability:
            canvas, profile_scale_x, profile_pad_x = self._apply_profile_squeeze(canvas)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )

        targets = self._empty_targets()
        annotations = self.annotations_by_image.get(int(info["id"]), [])
        annotations = sorted(
            annotations,
            key=lambda ann: float(ann["bbox"][2]) * float(ann["bbox"][3]),
            reverse=True,
        )[: self.max_people]

        for annotation in annotations:
            self._add_person_targets(
                targets,
                annotation,
                scale,
                pad_x,
                pad_y,
                profile_scale_x=profile_scale_x,
                profile_pad_x=profile_pad_x,
            )

        if self._rng.random() < self.horizontal_flip_probability:
            image_tensor = torch.flip(image_tensor, dims=(2,))
            targets = self._flip_targets(targets)

        return image_tensor, targets

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        left = (self.input_size - resized_width) // 2
        top = (self.input_size - resized_height) // 2
        canvas[top : top + resized_height, left : left + resized_width] = resized
        return canvas, scale, float(left), float(top)

    def _apply_profile_squeeze(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Comprime a cena horizontalmente para simular silhuetas mais de perfil."""
        factor = float(
            self._rng.uniform(self.profile_squeeze_min, self.profile_squeeze_max)
        )
        new_width = max(2, int(round(self.input_size * factor)))
        resized = cv2.resize(
            image,
            (new_width, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        result = np.zeros_like(image)
        left = (self.input_size - new_width) // 2
        result[:, left : left + new_width] = resized

        # Mapeamento das coordenadas de pixels do canvas original para o canvas
        # comprimido. Usamos extremos 0..W-1 para alinhar com os heatmaps.
        scale_x = (new_width - 1) / max(self.input_size - 1, 1)
        return result, float(scale_x), float(left)

    def _empty_targets(self) -> dict[str, torch.Tensor]:
        h = self.heatmap_size
        k = NUM_KEYPOINTS
        return {
            "center": torch.zeros((1, h, h), dtype=torch.float32),
            "keypoints": torch.zeros((k, h, h), dtype=torch.float32),
            "center_offsets": torch.zeros((k * 2, h, h), dtype=torch.float32),
            "center_offset_mask": torch.zeros((k, h, h), dtype=torch.float32),
            "parent_offsets": torch.zeros((k * 2, h, h), dtype=torch.float32),
            "parent_offset_mask": torch.zeros((k, h, h), dtype=torch.float32),
        }

    def _add_person_targets(
        self,
        targets: dict[str, torch.Tensor],
        annotation: dict,
        scale: float,
        pad_x: float,
        pad_y: float,
        profile_scale_x: float = 1.0,
        profile_pad_x: float = 0.0,
    ) -> None:
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        center_input_x = (
            ((x + width * 0.5) * scale + pad_x) * profile_scale_x
            + profile_pad_x
        )
        center_input_y = (y + height * 0.5) * scale + pad_y
        center_x, center_y = self._input_to_heatmap(center_input_x, center_input_y)
        if not self._inside_heatmap(center_x, center_y):
            return
        self._draw_gaussian(targets["center"][0], center_x, center_y, self.center_sigma)

        keypoints = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
        transformed: list[tuple[float, float] | None] = [None] * NUM_KEYPOINTS
        for index, (px, py, visibility) in enumerate(keypoints):
            if visibility <= 0:
                continue
            input_x = (float(px) * scale + pad_x) * profile_scale_x + profile_pad_x
            input_y = float(py) * scale + pad_y
            hx, hy = self._input_to_heatmap(input_x, input_y)
            if self._inside_heatmap(hx, hy):
                transformed[index] = (hx, hy)

        for keypoint_index, point in enumerate(transformed):
            if point is None:
                continue
            hx, hy = point
            self._draw_gaussian(targets["keypoints"][keypoint_index], hx, hy, self.keypoint_sigma)
            ix, iy = int(round(hx)), int(round(hy))
            if not (0 <= ix < self.heatmap_size and 0 <= iy < self.heatmap_size):
                continue

            if targets["center_offset_mask"][keypoint_index, iy, ix] <= 0:
                targets["center_offsets"][2 * keypoint_index, iy, ix] = center_x - hx
                targets["center_offsets"][2 * keypoint_index + 1, iy, ix] = center_y - hy
                targets["center_offset_mask"][keypoint_index, iy, ix] = 1.0

            parent_index = PARENT_BY_KEYPOINT[keypoint_index]
            if parent_index < 0 or transformed[parent_index] is None:
                continue
            parent_x, parent_y = transformed[parent_index]
            if targets["parent_offset_mask"][keypoint_index, iy, ix] <= 0:
                targets["parent_offsets"][2 * keypoint_index, iy, ix] = parent_x - hx
                targets["parent_offsets"][2 * keypoint_index + 1, iy, ix] = parent_y - hy
                targets["parent_offset_mask"][keypoint_index, iy, ix] = 1.0

    def _flip_targets(self, targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = {name: torch.flip(tensor.clone(), dims=(-1,)) for name, tensor in targets.items()}
        for name in ("keypoints", "center_offset_mask", "parent_offset_mask"):
            tensor = result[name]
            for left, right in FLIP_PAIRS:
                left_copy = tensor[left].clone()
                tensor[left] = tensor[right]
                tensor[right] = left_copy

        for name in ("center_offsets", "parent_offsets"):
            tensor = result[name].view(NUM_KEYPOINTS, 2, self.heatmap_size, self.heatmap_size)
            for left, right in FLIP_PAIRS:
                left_copy = tensor[left].clone()
                tensor[left] = tensor[right]
                tensor[right] = left_copy
            tensor[:, 0].mul_(-1.0)
            result[name] = tensor.view(NUM_KEYPOINTS * 2, self.heatmap_size, self.heatmap_size)
        return result

    def _input_to_heatmap(self, x: float, y: float) -> tuple[float, float]:
        factor = (self.heatmap_size - 1) / max(self.input_size - 1, 1)
        return x * factor, y * factor

    def _inside_heatmap(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.heatmap_size - 1 and 0.0 <= y <= self.heatmap_size - 1

    @staticmethod
    def _draw_gaussian(heatmap: torch.Tensor, center_x: float, center_y: float, sigma: float) -> None:
        height, width = heatmap.shape
        radius = max(1, int(round(sigma * 3.0)))
        left = max(0, int(center_x) - radius)
        right = min(width, int(center_x) + radius + 1)
        top = max(0, int(center_y) - radius)
        bottom = min(height, int(center_y) + radius + 1)
        if left >= right or top >= bottom:
            return
        ys = torch.arange(top, bottom, dtype=heatmap.dtype)
        xs = torch.arange(left, right, dtype=heatmap.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        gaussian = torch.exp(
            -((grid_x - center_x).square() + (grid_y - center_y).square())
            / (2.0 * sigma * sigma)
        )
        heatmap[top:bottom, left:right] = torch.maximum(heatmap[top:bottom, left:right], gaussian)

        # Focal loss precisa de um positivo inequívoco. Mantemos a posição
        # fracionária para offsets, mas garantimos pico 1.0 no pixel mais próximo.
        peak_x = int(round(center_x))
        peak_y = int(round(center_y))
        if 0 <= peak_x < width and 0 <= peak_y < height:
            heatmap[peak_y, peak_x] = 1.0
