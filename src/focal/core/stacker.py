"""Focus stacking algorithms: Laplacian pyramid and Complex wavelet fusion."""

from enum import Enum
from pathlib import Path
from typing import Callable
import cv2
import numpy as np

from focal.core import complex_wavelet
from focal.core.align import align_image
from focal.core.grayscale import compute_pca_weights, to_grayscale
from focal.core.reassign import build_color_map, reassign_colors


class StackAlgorithm(Enum):
    LAPLACIAN = "laplacian"
    COMPLEX_WAVELET = "complex_wavelet"


class FocusStacker:
    """Focus stacking with selectable algorithm."""

    def __init__(
        self,
        algorithm: StackAlgorithm = StackAlgorithm.LAPLACIAN,
        num_levels: int | None = None,
        kernel_size: int = 5,
        consistency: int = 2,
    ):
        self.algorithm = algorithm
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.consistency = consistency  # For complex_wavelet denoising (0-2)

    def stack(
        self,
        image_paths: list[Path],
        progress_callback: Callable[[int], None] | None = None
    ) -> np.ndarray:
        """
        Stack images using selected algorithm.

        Args:
            image_paths: List of paths to source images
            progress_callback: Optional callback for progress updates (0-100)

        Returns:
            Stacked result as numpy array (H, W, C) in uint8
        """
        if self.algorithm == StackAlgorithm.COMPLEX_WAVELET:
            return self._stack_complex_wavelet(image_paths, progress_callback)
        else:
            return self._stack_laplacian(image_paths, progress_callback)

    def _stack_laplacian(
        self,
        image_paths: list[Path],
        progress_callback: Callable[[int], None] | None = None
    ) -> np.ndarray:
        """Stack images using Laplacian pyramid method."""
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
                progress_callback(int((i + 1) / len(image_paths) * 30))

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
                progress_callback(30 + int((i + 1) / len(images) * 30))

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
                progress_callback(60 + int((level + 1) / num_levels * 30))

        # Reconstruct
        result = self._reconstruct_from_laplacian(fused_laplacian)
        result = np.clip(result, 0, 255).astype(np.uint8)

        if progress_callback:
            progress_callback(100)

        return result

    def _stack_complex_wavelet(
        self,
        image_paths: list[Path],
        progress_callback: Callable[[int], None] | None = None
    ) -> np.ndarray:
        """Stack images using complex_wavelet (Complex Daubechies) wavelet method."""
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
                progress_callback(int((i + 1) / len(image_paths) * 10))

        h, w = images[0].shape[:2]

        # Determine wavelet levels and expand images
        levels = complex_wavelet.levels_for_size((h, w))
        new_h, new_w = complex_wavelet.expand_to_valid_size((h, w), levels)

        # Expand images with reflection padding if needed
        expanded_images = []
        for img in images:
            if new_h != h or new_w != w:
                padded = cv2.copyMakeBorder(
                    img, 0, new_h - h, 0, new_w - w,
                    cv2.BORDER_REFLECT
                )
                expanded_images.append(padded)
            else:
                expanded_images.append(img)

        if progress_callback:
            progress_callback(15)

        # Determine reference image (middle of stack)
        ref_idx = len(expanded_images) // 2

        # Compute PCA weights from reference image for grayscale conversion
        pca_weights = compute_pca_weights(expanded_images[ref_idx])

        # Convert to grayscale using PCA weights
        grayscales = [to_grayscale(img, pca_weights) for img in expanded_images]

        if progress_callback:
            progress_callback(20)

        # Align images to reference
        ref_gray = grayscales[ref_idx]
        ref_color = expanded_images[ref_idx]

        aligned_colors = []
        aligned_grays = []

        for i, (gray, color) in enumerate(zip(grayscales, expanded_images)):
            if i == ref_idx:
                aligned_colors.append(color)
                aligned_grays.append(gray)
            else:
                aligned_color = align_image(ref_gray, ref_color, gray, color)
                aligned_gray = to_grayscale(aligned_color, pca_weights)
                aligned_colors.append(aligned_color)
                aligned_grays.append(aligned_gray)
            if progress_callback:
                progress_callback(20 + int((i + 1) / len(grayscales) * 20))

        # Decompose each aligned grayscale image
        wavelets = []
        for i, gray in enumerate(aligned_grays):
            wavelet = complex_wavelet.decompose(gray.astype(np.float32), levels)
            wavelets.append(wavelet)
            if progress_callback:
                progress_callback(40 + int((i + 1) / len(aligned_grays) * 20))

        # Merge wavelets
        merged_wavelet = complex_wavelet.merge_wavelets(wavelets, consistency=self.consistency)

        if progress_callback:
            progress_callback(65)

        # Reconstruct grayscale
        merged_gray = complex_wavelet.compose(merged_wavelet, levels)
        merged_gray = np.clip(merged_gray, 0, 255).astype(np.uint8)

        if progress_callback:
            progress_callback(70)

        # Reassign colors using color map
        color_map = build_color_map(aligned_grays, aligned_colors)

        if progress_callback:
            progress_callback(85)

        result = reassign_colors(merged_gray, color_map)

        if progress_callback:
            progress_callback(95)

        # Crop back to original size
        result = result[:h, :w]

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
