"""Unit tests for the FocusStacker class."""

import numpy as np
import pytest

from focal.core.stacker import FocusStacker, StackAlgorithm
from focal.core import complex_wavelet


class TestFocusStackerInit:
    """Tests for FocusStacker initialization."""

    def test_default_init(self):
        """Test default initialization."""
        stacker = FocusStacker()
        assert stacker.num_levels is None
        assert stacker.kernel_size == 5
        assert stacker.algorithm == StackAlgorithm.LAPLACIAN

    def test_custom_init(self):
        """Test initialization with custom parameters."""
        stacker = FocusStacker(num_levels=4, kernel_size=7)
        assert stacker.num_levels == 4
        assert stacker.kernel_size == 7

    def test_complex_wavelet_init(self):
        """Test initialization with Complex Wavelet algorithm."""
        stacker = FocusStacker(algorithm=StackAlgorithm.COMPLEX_WAVELET)
        assert stacker.algorithm == StackAlgorithm.COMPLEX_WAVELET
        assert stacker.consistency == 2


class TestComputeNumLevels:
    """Tests for automatic pyramid level computation."""

    def test_small_image(self):
        """Small images should get fewer levels."""
        stacker = FocusStacker()
        levels = stacker._compute_num_levels((64, 64, 3))
        assert levels >= 1
        assert levels <= 6

    def test_large_image(self):
        """Large images can have more levels."""
        stacker = FocusStacker()
        levels = stacker._compute_num_levels((2048, 2048, 3))
        assert levels >= 1
        assert levels <= 6

    def test_asymmetric_image(self):
        """Asymmetric images use the smaller dimension."""
        stacker = FocusStacker()
        levels = stacker._compute_num_levels((2048, 256, 3))
        # Should be based on 256, not 2048
        levels_square = stacker._compute_num_levels((256, 256, 3))
        assert levels == levels_square


class TestGaussianPyramid:
    """Tests for Gaussian pyramid building."""

    def test_pyramid_length(self):
        """Pyramid should have the specified number of levels."""
        stacker = FocusStacker()
        image = np.random.rand(128, 128, 3).astype(np.float32)
        pyramid = stacker._build_gaussian_pyramid(image, 4)
        assert len(pyramid) == 4

    def test_pyramid_sizes_decrease(self):
        """Each level should be smaller than the previous."""
        stacker = FocusStacker()
        image = np.random.rand(128, 128, 3).astype(np.float32)
        pyramid = stacker._build_gaussian_pyramid(image, 4)
        for i in range(len(pyramid) - 1):
            assert pyramid[i].shape[0] > pyramid[i + 1].shape[0]
            assert pyramid[i].shape[1] > pyramid[i + 1].shape[1]

    def test_first_level_is_original(self):
        """First level should be the original image."""
        stacker = FocusStacker()
        image = np.random.rand(128, 128, 3).astype(np.float32)
        pyramid = stacker._build_gaussian_pyramid(image, 4)
        np.testing.assert_array_equal(pyramid[0], image)


class TestLaplacianPyramid:
    """Tests for Laplacian pyramid building."""

    def test_laplacian_length(self):
        """Laplacian pyramid should match Gaussian pyramid length."""
        stacker = FocusStacker()
        image = np.random.rand(128, 128, 3).astype(np.float32)
        gaussian = stacker._build_gaussian_pyramid(image, 4)
        laplacian = stacker._build_laplacian_pyramid(gaussian)
        assert len(laplacian) == len(gaussian)

    def test_laplacian_last_is_gaussian_last(self):
        """Last level of Laplacian should be last Gaussian level."""
        stacker = FocusStacker()
        image = np.random.rand(128, 128, 3).astype(np.float32)
        gaussian = stacker._build_gaussian_pyramid(image, 4)
        laplacian = stacker._build_laplacian_pyramid(gaussian)
        np.testing.assert_array_equal(laplacian[-1], gaussian[-1])


class TestReconstruction:
    """Tests for Laplacian pyramid reconstruction."""

    def test_reconstruction_shape(self):
        """Reconstructed image should match original shape."""
        stacker = FocusStacker()
        image = np.random.rand(128, 128, 3).astype(np.float32)
        gaussian = stacker._build_gaussian_pyramid(image, 4)
        laplacian = stacker._build_laplacian_pyramid(gaussian)
        reconstructed = stacker._reconstruct_from_laplacian(laplacian)
        assert reconstructed.shape == image.shape

    def test_reconstruction_accuracy(self):
        """Reconstruction should be close to original (within float precision)."""
        stacker = FocusStacker()
        image = np.random.rand(128, 128, 3).astype(np.float32) * 255
        gaussian = stacker._build_gaussian_pyramid(image, 4)
        laplacian = stacker._build_laplacian_pyramid(gaussian)
        reconstructed = stacker._reconstruct_from_laplacian(laplacian)
        # Allow some tolerance for floating point operations
        np.testing.assert_allclose(reconstructed, image, atol=1.0)


class TestFocusMeasure:
    """Tests for focus measure computation."""

    def test_focus_measure_shape(self):
        """Focus measure should match input shape."""
        stacker = FocusStacker()
        gray = np.random.rand(64, 64).astype(np.float32)
        measure = stacker._compute_focus_measure(gray)
        assert measure.shape == gray.shape

    def test_focus_measure_non_negative(self):
        """Focus measure should be non-negative (it's a variance)."""
        stacker = FocusStacker()
        gray = np.random.rand(64, 64).astype(np.float32)
        measure = stacker._compute_focus_measure(gray)
        assert np.all(measure >= 0)

    def test_sharp_vs_blurry(self):
        """Sharp regions should have higher focus measure than blurry."""
        stacker = FocusStacker()
        # Create a sharp edge
        sharp = np.zeros((64, 64), dtype=np.float32)
        sharp[:, 32:] = 255
        # Create a smooth gradient (blurry)
        blurry = np.linspace(0, 255, 64).astype(np.float32)
        blurry = np.tile(blurry, (64, 1))

        sharp_measure = stacker._compute_focus_measure(sharp)
        blurry_measure = stacker._compute_focus_measure(blurry)

        # Sharp edge should have higher max focus measure
        assert np.max(sharp_measure) > np.max(blurry_measure)


class TestWeightComputation:
    """Tests for weight computation."""

    def test_weights_sum_to_one(self):
        """Weights should sum to 1 at each pixel."""
        stacker = FocusStacker()
        measures = [
            np.random.rand(32, 32).astype(np.float32),
            np.random.rand(32, 32).astype(np.float32),
            np.random.rand(32, 32).astype(np.float32),
        ]
        weights = stacker._compute_weights(measures)
        total = sum(weights)
        np.testing.assert_allclose(total, 1.0, atol=1e-5)

    def test_weights_match_count(self):
        """Should return same number of weights as focus measures."""
        stacker = FocusStacker()
        measures = [
            np.random.rand(32, 32).astype(np.float32),
            np.random.rand(32, 32).astype(np.float32),
        ]
        weights = stacker._compute_weights(measures)
        assert len(weights) == len(measures)

    def test_weights_favor_higher_focus(self):
        """Higher focus measure should get higher weight."""
        stacker = FocusStacker()
        # One image has high focus everywhere, one has low
        high_focus = np.ones((32, 32), dtype=np.float32) * 100
        low_focus = np.ones((32, 32), dtype=np.float32) * 1
        weights = stacker._compute_weights([high_focus, low_focus])
        # First weight should be higher everywhere
        assert np.all(weights[0] > weights[1])


class TestStackValidation:
    """Tests for stack input validation."""

    def test_empty_list_raises(self):
        """Empty image list should raise ValueError."""
        stacker = FocusStacker()
        with pytest.raises(ValueError, match="No images to stack"):
            stacker.stack([])


class TestComplexWavelet:
    """Tests for Complex Wavelet transform."""

    def test_levels_for_size(self):
        """Test level calculation for various image sizes."""
        # Aims for ~8 pixel image at lowest level
        # 64x64: 64 >> 5 = 2, which is < 8, so stays at MIN_LEVELS=5
        assert complex_wavelet.levels_for_size((64, 64)) == 5
        # 2048x2048: 2048 >> 5 = 64 > 8, so keeps incrementing
        # 2048 >> 8 = 8, so stops at 8 levels
        assert complex_wavelet.levels_for_size((2048, 2048)) == 8

    def test_expand_to_valid_size(self):
        """Test image expansion to valid wavelet size."""
        levels = 5
        h, w = complex_wavelet.expand_to_valid_size((100, 100), levels)
        assert h % (1 << levels) == 0
        assert w % (1 << levels) == 0
        assert h >= 100
        assert w >= 100

    def test_decompose_compose_roundtrip(self):
        """Test that decompose -> compose recovers the original."""
        image = np.random.rand(64, 64).astype(np.float32) * 255
        levels = 5
        wavelet = complex_wavelet.decompose(image, levels)
        reconstructed = complex_wavelet.compose(wavelet, levels)
        np.testing.assert_allclose(reconstructed, image, atol=1.0)

    def test_wavelet_shape(self):
        """Test that wavelet output has correct shape."""
        image = np.random.rand(64, 64).astype(np.float32)
        levels = 5
        wavelet = complex_wavelet.decompose(image, levels)
        assert wavelet.shape == (64, 64, 2)  # Complex: (H, W, 2)

    def test_merge_wavelets(self):
        """Test merging multiple wavelets."""
        image1 = np.random.rand(64, 64).astype(np.float32) * 255
        image2 = np.random.rand(64, 64).astype(np.float32) * 255
        levels = 5
        w1 = complex_wavelet.decompose(image1, levels)
        w2 = complex_wavelet.decompose(image2, levels)
        merged = complex_wavelet.merge_wavelets([w1, w2], consistency=0)
        assert merged.shape == w1.shape

    def test_merge_selects_max_magnitude(self):
        """Test that merge picks higher magnitude coefficients."""
        # Create two images: one with high energy, one with low
        # Use 64x64 to ensure valid size for 5 decomposition levels
        high = np.ones((64, 64), dtype=np.float32) * 200
        low = np.ones((64, 64), dtype=np.float32) * 50
        levels = complex_wavelet.levels_for_size((64, 64))
        w_high = complex_wavelet.decompose(high, levels)
        w_low = complex_wavelet.decompose(low, levels)
        merged = complex_wavelet.merge_wavelets([w_high, w_low], consistency=0)
        # Merged should be closer to high energy image
        reconstructed = complex_wavelet.compose(merged, levels)
        assert np.mean(reconstructed) > np.mean(low)
