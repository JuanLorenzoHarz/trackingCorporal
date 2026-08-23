"""Testes sintéticos do dataset full-frame da PoseNet V2."""

import json

import cv2
import numpy as np
import torch

from src.data.coco_multiperson_dataset import CocoMultiPersonPoseDataset
from src.pose.keypoints import NUM_KEYPOINTS


def _person_keypoints(base_x: float, base_y: float) -> list[float]:
    values: list[float] = []
    for index in range(NUM_KEYPOINTS):
        if index < 6:
            values.extend((base_x + index, base_y + index, 2))
        else:
            values.extend((0, 0, 0))
    return values


def _write_two_person_fixture(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "with_people.jpg"), image)
    cv2.imwrite(str(images_dir / "empty.jpg"), image)

    coco = {
        "images": [
            {"id": 1, "file_name": "with_people.jpg", "width": 200, "height": 100},
            {"id": 2, "file_name": "empty.jpg", "width": 200, "height": 100},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "iscrowd": 0,
                "num_keypoints": 6,
                "bbox": [20, 15, 45, 75],
                "keypoints": _person_keypoints(30, 25),
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "iscrowd": 0,
                "num_keypoints": 6,
                "bbox": [130, 15, 45, 75],
                "keypoints": _person_keypoints(140, 25),
            },
        ],
    }
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(coco), encoding="utf-8")
    return images_dir, annotations


def _center_peak_xs(center_heatmap: torch.Tensor) -> list[int]:
    positions = torch.nonzero(center_heatmap[0] >= 0.999, as_tuple=False)
    return sorted(int(position[1].item()) for position in positions)


def test_full_frame_dataset_contains_two_person_centers_and_negative_image(tmp_path):
    images_dir, annotations = _write_two_person_fixture(tmp_path)

    dataset = CocoMultiPersonPoseDataset(
        images_dir=images_dir,
        annotations_file=annotations,
        input_size=256,
        heatmap_size=64,
        min_keypoints=4,
        horizontal_flip_probability=0.0,
        profile_squeeze_probability=0.0,
    )

    image_tensor, targets = dataset[0]
    assert image_tensor.shape == (3, 256, 256)
    assert targets["center"].shape == (1, 64, 64)
    assert int((targets["center"] > 0.90).sum().item()) >= 2
    assert float(targets["keypoints"].max().item()) > 0.9
    assert float(targets["center_offset_mask"].sum().item()) > 0

    _, negative = dataset[1]
    assert float(negative["center"].sum().item()) == 0.0
    assert float(negative["keypoints"].sum().item()) == 0.0
    assert float(negative["center_offset_mask"].sum().item()) == 0.0


def test_profile_squeeze_moves_person_targets_toward_frame_center(tmp_path):
    images_dir, annotations = _write_two_person_fixture(tmp_path)

    normal = CocoMultiPersonPoseDataset(
        images_dir=images_dir,
        annotations_file=annotations,
        input_size=256,
        heatmap_size=64,
        min_keypoints=4,
        horizontal_flip_probability=0.0,
        profile_squeeze_probability=0.0,
    )
    squeezed = CocoMultiPersonPoseDataset(
        images_dir=images_dir,
        annotations_file=annotations,
        input_size=256,
        heatmap_size=64,
        min_keypoints=4,
        horizontal_flip_probability=0.0,
        profile_squeeze_probability=1.0,
        profile_squeeze_min=0.60,
        profile_squeeze_max=0.60,
        augmentation_seed=123,
    )

    _, normal_targets = normal[0]
    _, squeezed_targets = squeezed[0]

    normal_xs = _center_peak_xs(normal_targets["center"])
    squeezed_xs = _center_peak_xs(squeezed_targets["center"])

    assert len(normal_xs) == 2
    assert len(squeezed_xs) == 2
    assert squeezed_xs[0] > normal_xs[0]
    assert squeezed_xs[1] < normal_xs[1]
    assert float(squeezed_targets["keypoints"].max().item()) >= 0.999
    assert float(squeezed_targets["center_offset_mask"].sum().item()) > 0.0
