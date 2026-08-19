"""Testes do wrapper de câmera sem exigir webcam física."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.capture.camera import Camera


@patch("src.capture.camera.cv2.VideoCapture")
def test_camera_opens_and_releases(mock_video_capture):
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.get.side_effect = [1280.0, 720.0]
    mock_video_capture.return_value = capture

    with Camera(index=0, width=1280, height=720) as camera:
        assert camera.is_opened()
        assert camera.resolution == (1280, 720)

    capture.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.release.assert_called_once()


@patch("src.capture.camera.cv2.VideoCapture")
def test_camera_reads_frame(mock_video_capture):
    capture = MagicMock()
    capture.isOpened.return_value = True
    expected_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    capture.read.return_value = (True, expected_frame)
    mock_video_capture.return_value = capture

    camera = Camera(index=0)
    frame = camera.read()
    camera.release()

    assert np.array_equal(frame, expected_frame)


@patch("src.capture.camera.cv2.VideoCapture")
def test_camera_fails_when_device_cannot_open(mock_video_capture):
    capture = MagicMock()
    capture.isOpened.return_value = False
    mock_video_capture.return_value = capture

    with pytest.raises(RuntimeError):
        Camera(index=0)

    capture.release.assert_called_once()
