"""Teste isolado da oclusão artificial usada no treinamento."""

import numpy as np

from src.data.coco_pose_dataset import CocoPoseDataset


def test_random_occlusion_changes_image_without_mutating_original():
    dataset = object.__new__(CocoPoseDataset)
    dataset.occlusion_probability = 1.0
    dataset.occlusion_min_size = 0.30
    dataset.occlusion_max_size = 0.30
    dataset._rng = np.random.default_rng(1234)

    image = np.full((100, 100, 3), 180, dtype=np.uint8)
    original = image.copy()

    result = dataset._apply_random_occlusion(image)

    assert np.array_equal(image, original)
    assert not np.array_equal(result, original)
