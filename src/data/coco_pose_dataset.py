"""Dataset COCO simplificado para treino de pose de uma pessoa por recorte.

O loader lê diretamente o JSON oficial de keypoints do COCO, sem depender de
um detector de pose pronto ou de pycocotools. Cada anotação de pessoa vira um
exemplo independente: a pessoa é recortada pela bounding box, centralizada em
um quadrado e redimensionada para a entrada da CNN.

Opcionalmente, retângulos de oclusão artificial podem ser aplicados à imagem
sem remover os targets dos keypoints. Isso ensina a CNN a inferir articulações
mesmo quando os pixels daquela região estão temporariamente escondidos.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.pose.heatmaps import generate_heatmaps
from src.pose.keypoints import NUM_KEYPOINTS


class CocoPoseDataset(Dataset):
    """Carrega pessoas anotadas no COCO como exemplos de pose individuais."""

    def __init__(
        self,
        images_dir: str | Path,
        annotations_file: str | Path,
        input_size: int = 256,
        heatmap_size: int = 64,
        sigma: float = 1.8,
        crop_margin: float = 0.15,
        min_keypoints: int = 5,
        max_samples: int | None = None,
        occlusion_probability: float = 0.0,
        occlusion_min_size: float = 0.12,
        occlusion_max_size: float = 0.35,
        augmentation_seed: int | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.crop_margin = crop_margin
        self.occlusion_probability = occlusion_probability
        self.occlusion_min_size = occlusion_min_size
        self.occlusion_max_size = occlusion_max_size
        self._rng = np.random.default_rng(augmentation_seed)

        if not 0.0 <= occlusion_probability <= 1.0:
            raise ValueError("occlusion_probability deve estar entre 0 e 1.")
        if not 0.0 < occlusion_min_size <= occlusion_max_size <= 1.0:
            raise ValueError(
                "occlusion_min_size e occlusion_max_size devem satisfazer "
                "0 < min <= max <= 1."
            )

        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"Pasta de imagens COCO não encontrada: {self.images_dir}"
            )
        if not self.annotations_file.is_file():
            raise FileNotFoundError(
                f"Arquivo de anotações COCO não encontrado: {self.annotations_file}"
            )

        with self.annotations_file.open("r", encoding="utf-8") as file:
            coco = json.load(file)

        images_by_id = {image["id"]: image for image in coco["images"]}
        samples: list[dict] = []

        for annotation in coco["annotations"]:
            if annotation.get("category_id") != 1:
                continue
            if annotation.get("iscrowd", 0):
                continue
            if annotation.get("num_keypoints", 0) < min_keypoints:
                continue

            bbox = annotation.get("bbox")
            raw_keypoints = annotation.get("keypoints")
            image_info = images_by_id.get(annotation.get("image_id"))

            if image_info is None or bbox is None or raw_keypoints is None:
                continue
            if len(raw_keypoints) != NUM_KEYPOINTS * 3:
                continue
            if bbox[2] <= 2 or bbox[3] <= 2:
                continue

            samples.append(
                {
                    "file_name": image_info["file_name"],
                    "bbox": bbox,
                    "keypoints": raw_keypoints,
                }
            )

            if max_samples is not None and len(samples) >= max_samples:
                break

        if not samples:
            raise RuntimeError("Nenhuma anotação de pessoa utilizável foi encontrada.")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image_path = self.images_dir / sample["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Não foi possível ler a imagem: {image_path}")

        keypoints = np.asarray(sample["keypoints"], dtype=np.float32).reshape(
            NUM_KEYPOINTS, 3
        )
        bbox = np.asarray(sample["bbox"], dtype=np.float32)

        crop, normalized_keypoints, visibility = self._crop_person(
            image,
            bbox,
            keypoints,
        )

        crop = self._apply_random_occlusion(crop)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        keypoint_tensor = torch.from_numpy(normalized_keypoints).float()
        visibility_tensor = torch.from_numpy(visibility).float()

        heatmaps = generate_heatmaps(
            keypoint_tensor,
            visibility_tensor,
            heatmap_size=self.heatmap_size,
            sigma=self.sigma,
        )
        visibility_mask = visibility_tensor.view(NUM_KEYPOINTS, 1, 1)

        return image_tensor, heatmaps, visibility_mask

    def _apply_random_occlusion(self, image: np.ndarray) -> np.ndarray:
        """Esconde uma região aleatória sem alterar os targets de pose."""
        if self.occlusion_probability <= 0.0:
            return image
        if self._rng.random() > self.occlusion_probability:
            return image

        height, width = image.shape[:2]
        occlusion_width = max(
            1,
            int(
                round(
                    width
                    * self._rng.uniform(
                        self.occlusion_min_size,
                        self.occlusion_max_size,
                    )
                )
            ),
        )
        occlusion_height = max(
            1,
            int(
                round(
                    height
                    * self._rng.uniform(
                        self.occlusion_min_size,
                        self.occlusion_max_size,
                    )
                )
            ),
        )

        max_x = max(0, width - occlusion_width)
        max_y = max(0, height - occlusion_height)
        left = int(self._rng.integers(0, max_x + 1)) if max_x > 0 else 0
        top = int(self._rng.integers(0, max_y + 1)) if max_y > 0 else 0

        fill = self._rng.integers(0, 256, size=3, dtype=np.uint8)
        result = image.copy()
        result[
            top : top + occlusion_height,
            left : left + occlusion_width,
        ] = fill
        return result

    def _crop_person(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        keypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recorta uma região quadrada ao redor da pessoa e remapeia keypoints."""
        x, y, width, height = bbox
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        side = max(width, height) * (1.0 + 2.0 * self.crop_margin)
        side = max(side, 2.0)

        left = center_x - side / 2.0
        top = center_y - side / 2.0
        scale = self.input_size / side

        transform = np.asarray(
            [
                [scale, 0.0, -left * scale],
                [0.0, scale, -top * scale],
            ],
            dtype=np.float32,
        )

        crop = cv2.warpAffine(
            image,
            transform,
            (self.input_size, self.input_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        normalized = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        visibility = np.zeros(NUM_KEYPOINTS, dtype=np.float32)

        for point_index, (point_x, point_y, coco_visibility) in enumerate(keypoints):
            if coco_visibility <= 0:
                continue

            normalized_x = (point_x - left) / side
            normalized_y = (point_y - top) / side

            if 0.0 <= normalized_x <= 1.0 and 0.0 <= normalized_y <= 1.0:
                normalized[point_index] = (normalized_x, normalized_y)
                visibility[point_index] = 1.0

        return crop, normalized, visibility
