"""Captura de frames da webcam com OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


class Camera:
    """Pequeno wrapper responsável por abrir, ler e liberar uma webcam."""

    def __init__(
        self,
        index: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.index = index
        self._capture = cv2.VideoCapture(index)

        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(
                f"Não foi possível abrir a câmera de índice {index}."
            )

        if width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def resolution(self) -> tuple[int, int]:
        """Retorna a resolução realmente entregue pela câmera."""
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def is_opened(self) -> bool:
        """Indica se o dispositivo de captura continua aberto."""
        return self._capture.isOpened()

    def read(self) -> np.ndarray:
        """Lê e retorna um frame; falha explicitamente se a leitura não funcionar."""
        success, frame = self._capture.read()

        if not success or frame is None:
            raise RuntimeError(
                f"Não foi possível ler um frame da câmera de índice {self.index}."
            )

        return frame

    def release(self) -> None:
        """Libera o dispositivo de câmera."""
        if self._capture is not None:
            self._capture.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
