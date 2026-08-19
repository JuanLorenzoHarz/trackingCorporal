"""Teste do loader COCO usando um exemplo mínimo criado localmente."""

import json

import cv2
import numpy as np

from src.data.coco_pose_dataset import CocoPoseDataset
from src.pose.keypoints import NUM_KEYPOINTS


def test_coco_pose_dataset_shapes(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    image = np.full((100, 100, 3), 180, dtype=np.uint8)
    image_path = images_dir / "sample.jpg"
    assert cv2.imwrite(str(image_path), image)

    keypoints = []
    for index in range(NUM_KEYPOINTS):
        x = 30 + (index % 5) * 8
        y = 25 + (index // 5) * 15
        keypoints.extend([x, y, 2])

    annotations = {
        "images": [
            {
                "id": 1,
                "file_name": "sample.jpg",
                "width": 100,
                "height": 100,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "iscrowd": 0,
                "num_keypoints": NUM_KEYPOINTS,
                "bbox": [10, 10, 80, 80],
                "keypoints": keypoints,
            }
        ],
    }

    annotations_file = tmp_path / "person_keypoints.json"
    annotations_file.write_text(json.dumps(annotations), encoding="utf-8")

    dataset = CocoPoseDataset(
        images_dir=images_dir,
        annotations_file=annotations_file,
        input_size=256,
        heatmap_size=64,
    )

    image_tensor, heatmaps, visibility = dataset[0]

    assert len(dataset) == 1
    assert image_tensor.shape == (3, 256, 256)
    assert heatmaps.shape == (NUM_KEYPOINTS, 64, 64)
    assert visibility.shape == (NUM_KEYPOINTS, 1, 1)
    assert float(visibility.sum()) == NUM_KEYPOINTS
