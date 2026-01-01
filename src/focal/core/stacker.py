"""Focus stacking algorithm using Laplacian pyramid fusion."""

from pathlib import Path
from typing import Callable
import cv2
import numpy as np


class FocusStacker:
    """Laplacian pyramid focus stacking."""

    def __init__(self, num_levels: int | None = None, kernel_size: int = 5):
        self.num_levels = num_levels
        self.kernel_size = kernel_size

    def stack(
        self,
        image_paths: list[Path],
        progress_callback: Callable[[int], None] | None = None
    ) -> np.ndarray:
        """
        Stack images using Laplacian pyramid method.

        Args:
            image_paths: List of paths to source images
            progress_callback: Optional callback for progress updates (0-100)

        Returns:
            Stacked result as numpy array (H, W, C) in uint8
        """
        if not image_paths:
            raise ValueError("No images to stack")

        if len(image_paths) == 1:
            return cv2.imread(str(image_paths[0]))

        # Load images
        images = []
        for i, path in enumerate(image_paths):
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            images.append(img)
            if progress_callback:
                progress_callback(int((i + 1) / len(image_paths) * 30))  # 0-30%

        # Convert to float32
        float_images = [img.astype(np.float32) for img in images]
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in float_images]

        # Determine pyramid levels
        num_levels = self.num_levels or self._compute_num_levels(images[0].shape)

        # Build pyramids
        laplacian_pyramids = []
        gaussian_pyramids_gray = []

        for i, (img, gray) in enumerate(zip(float_images, gray_images)):
            gauss = self._build_gaussian_pyramid(img, num_levels)
            gauss_gray = self._build_gaussian_pyramid(gray, num_levels)
            laplacian_pyramids.append(self._build_laplacian_pyramid(gauss))
            gaussian_pyramids_gray.append(gauss_gray)
            if progress_callback:
                progress_callback(30 + int((i + 1) / len(images) * 30))  # 30-60%

        # Fuse at each level
        fused_laplacian = []
        for level in range(num_levels):
            focus_measures = [
                self._compute_focus_measure(gauss_gray[level])
                for gauss_gray in gaussian_pyramids_gray
            ]
            weights = self._compute_weights(focus_measures)

            blended = np.zeros_like(laplacian_pyramids[0][level])
            for i, lap_pyr in enumerate(laplacian_pyramids):
                w = weights[i]
                if len(blended.shape) == 3:
                    w = w[:, :, np.newaxis]
                blended += lap_pyr[level] * w

            fused_laplacian.append(blended)
            if progress_callback:
                progress_callback(60 + int((level + 1) / num_levels * 30))  # 60-90%

        # Reconstruct
        result = self._reconstruct_from_laplacian(fused_laplacian)
        result = np.clip(result, 0, 255).astype(np.uint8)

        if progress_callback:
            progress_callback(100)

        return result

    def _compute_num_levels(self, shape: tuple[int, ...]) -> int:
        min_dim = min(shape[0], shape[1])
        levels = int(np.log2(min_dim / 16))
        return max(1, min(levels, 6))

    def _build_gaussian_pyramid(
        self, image: np.ndarray, levels: int
    ) -> list[np.ndarray]:
        pyramid = [image]
        current = image
        for _ in range(levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        return pyramid

    def _build_laplacian_pyramid(
        self, gaussian_pyramid: list[np.ndarray]
    ) -> list[np.ndarray]:
        laplacian = []
        for i in range(len(gaussian_pyramid) - 1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            diff = cv2.subtract(gaussian_pyramid[i], upsampled)
            laplacian.append(diff)
        laplacian.append(gaussian_pyramid[-1])
        return laplacian

    def _reconstruct_from_laplacian(
        self, laplacian_pyramid: list[np.ndarray]
    ) -> np.ndarray:
        current = laplacian_pyramid[-1]
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
            current = cv2.pyrUp(current, dstsize=size)
            current = cv2.add(current, laplacian_pyramid[i])
        return current

    def _compute_focus_measure(self, gray: np.ndarray) -> np.ndarray:
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        mean = cv2.boxFilter(laplacian, cv2.CV_32F, (self.kernel_size, self.kernel_size))
        sq_mean = cv2.boxFilter(
            laplacian * laplacian, cv2.CV_32F, (self.kernel_size, self.kernel_size)
        )
        variance = sq_mean - mean * mean
        return np.maximum(variance, 0).astype(np.float32)

    def _compute_weights(
        self, focus_measures: list[np.ndarray]
    ) -> list[np.ndarray]:
        stacked = np.stack(focus_measures, axis=0)
        total = np.sum(stacked, axis=0, keepdims=True) + 1e-10
        weights = stacked / total
        return [weights[i] for i in range(len(focus_measures))]
