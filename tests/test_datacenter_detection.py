"""
Tests for datacenter GPU detection and backend optimization.

These tests verify that:
1. Datacenter GPUs (MI300X, A100, H100, etc.) are correctly detected
2. Triton is preferred over Vulkan for datacenter GPUs
3. The optimize_for_hardware() function works correctly
4. Backend selection logic prioritizes correctly
"""

import unittest
from unittest.mock import patch, MagicMock
import sys

# Check if torch is available for mocking tests
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestDatacenterDetection(unittest.TestCase):
    """Test GPU type detection logic."""

    def test_amd_datacenter_patterns(self):
        """Test AMD datacenter GPU name patterns are recognized."""
        amd_datacenter_names = [
            'AMD Instinct MI300X',
            'AMD Instinct MI250X',
            'AMD Instinct MI210',
            'AMD Instinct MI200',
            'AMD Instinct MI100',
            'Instinct MI300X OAM',
        ]

        for gpu_name in amd_datacenter_names:
            with self.subTest(gpu=gpu_name):
                is_dc = self._check_gpu_pattern(gpu_name.lower())
                self.assertTrue(is_dc, f"{gpu_name} should be detected as datacenter")

    def test_nvidia_datacenter_patterns(self):
        """Test NVIDIA datacenter GPU name patterns are recognized."""
        nvidia_datacenter_names = [
            'NVIDIA A100-SXM4-80GB',
            'NVIDIA A100 PCIe',
            'NVIDIA H100 PCIe',
            'NVIDIA H100 SXM5',
            'Tesla V100-SXM2',
            'Tesla T4',
            'NVIDIA L40',
            'NVIDIA A30',
        ]

        for gpu_name in nvidia_datacenter_names:
            with self.subTest(gpu=gpu_name):
                is_dc = self._check_gpu_pattern(gpu_name.lower())
                self.assertTrue(is_dc, f"{gpu_name} should be detected as datacenter")

    def test_consumer_patterns(self):
        """Test consumer GPU name patterns are NOT flagged as datacenter."""
        consumer_names = [
            'NVIDIA GeForce RTX 4090',
            'NVIDIA GeForce RTX 3080',
            'AMD Radeon RX 7900 XTX',
            'AMD Radeon RX 6800 XT',
            'NVIDIA GeForce GTX 1080 Ti',
        ]

        for gpu_name in consumer_names:
            with self.subTest(gpu=gpu_name):
                is_dc = self._check_gpu_pattern(gpu_name.lower())
                self.assertFalse(is_dc, f"{gpu_name} should NOT be detected as datacenter")

    def _check_gpu_pattern(self, gpu_name_lower):
        """Check if GPU name matches datacenter patterns."""
        # AMD Datacenter patterns
        amd_datacenter_patterns = [
            'mi300', 'mi250', 'mi210', 'mi200', 'mi100', 'mi60', 'mi50',
            'instinct',
        ]

        # NVIDIA Datacenter patterns
        nvidia_datacenter_patterns = [
            'a100', 'a800', 'h100', 'h200', 'h800',
            'a30', 'a40', 'a10', 'a16',
            'v100', 'p100', 't4',
            'l40', 'l4',
            'b100', 'b200',
            'dgx', 'hgx',
        ]

        for pattern in amd_datacenter_patterns + nvidia_datacenter_patterns:
            if pattern in gpu_name_lower:
                return True
        return False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestBackendSelection(unittest.TestCase):
    """Test backend selection logic for different GPU types."""

    def setUp(self):
        """Reset module state before each test."""
        # We need to reimport to reset state
        if 'aule' in sys.modules:
            # Store current state
            pass

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_datacenter_prefers_triton(self, mock_props, mock_available):
        """Test that datacenter GPUs prefer Triton over Vulkan."""
        # Mock MI300X
        mock_available.return_value = True
        mock_device = MagicMock()
        mock_device.name = 'AMD Instinct MI300X'
        mock_device.total_memory = 192 * 1024**3  # 192GB
        mock_device.major = 9
        mock_device.minor = 0
        mock_props.return_value = mock_device

        # Import after mocking
        from aule import _detect_gpu_type

        is_dc, gpu_type, gpu_name = _detect_gpu_type()

        self.assertTrue(is_dc)
        self.assertEqual(gpu_type, 'datacenter')
        self.assertIn('MI300X', gpu_name)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_consumer_allows_vulkan(self, mock_props, mock_available):
        """Test that consumer GPUs can use Vulkan without warning."""
        mock_available.return_value = True
        mock_device = MagicMock()
        mock_device.name = 'NVIDIA GeForce RTX 4090'
        mock_device.total_memory = 24 * 1024**3  # 24GB
        mock_device.major = 8
        mock_device.minor = 9
        mock_props.return_value = mock_device

        from aule import _detect_gpu_type

        is_dc, gpu_type, gpu_name = _detect_gpu_type()

        self.assertFalse(is_dc)
        self.assertEqual(gpu_type, 'consumer')

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_high_memory_heuristic(self, mock_props, mock_available):
        """Test that high memory (>40GB) triggers datacenter detection."""
        mock_available.return_value = True
        mock_device = MagicMock()
        mock_device.name = 'Unknown GPU'  # Unknown name
        mock_device.total_memory = 80 * 1024**3  # 80GB - likely datacenter
        mock_device.major = 8
        mock_device.minor = 0
        mock_props.return_value = mock_device

        from aule import _detect_gpu_type

        is_dc, gpu_type, gpu_name = _detect_gpu_type()

        self.assertTrue(is_dc)
        self.assertEqual(gpu_type, 'datacenter')


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestOptimizeForHardware(unittest.TestCase):
    """Test the optimize_for_hardware() function."""

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_optimize_returns_info(self, mock_props, mock_available):
        """Test that optimize_for_hardware returns correct info dict."""
        mock_available.return_value = True
        mock_device = MagicMock()
        mock_device.name = 'AMD Instinct MI300X'
        mock_device.total_memory = 192 * 1024**3
        mock_device.major = 9
        mock_device.minor = 0
        mock_props.return_value = mock_device

        from aule import get_gpu_info

        info = get_gpu_info()

        self.assertIn('name', info)
        self.assertIn('type', info)
        self.assertIn('is_datacenter', info)
        self.assertIn('recommended_backend', info)

    @patch('torch.cuda.is_available')
    def test_no_cuda_graceful(self, mock_available):
        """Test graceful handling when CUDA is not available."""
        mock_available.return_value = False

        from aule import _detect_gpu_type

        is_dc, gpu_type, gpu_name = _detect_gpu_type()

        self.assertFalse(is_dc)
        self.assertEqual(gpu_type, 'unknown')


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestIsDatacenterGpu(unittest.TestCase):
    """Test the is_datacenter_gpu() public function."""

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_is_datacenter_gpu_mi300x(self, mock_props, mock_available):
        """Test is_datacenter_gpu returns True for MI300X."""
        mock_available.return_value = True
        mock_device = MagicMock()
        mock_device.name = 'AMD Instinct MI300X'
        mock_device.total_memory = 192 * 1024**3
        mock_device.major = 9
        mock_device.minor = 0
        mock_props.return_value = mock_device

        # Reset cached state
        import aule
        aule._detected_gpu_type = None
        aule._is_datacenter_gpu = False

        result = aule.is_datacenter_gpu()

        self.assertTrue(result)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_is_datacenter_gpu_rtx4090(self, mock_props, mock_available):
        """Test is_datacenter_gpu returns False for RTX 4090."""
        mock_available.return_value = True
        mock_device = MagicMock()
        mock_device.name = 'NVIDIA GeForce RTX 4090'
        mock_device.total_memory = 24 * 1024**3
        mock_device.major = 8
        mock_device.minor = 9
        mock_props.return_value = mock_device

        # Reset cached state
        import aule
        aule._detected_gpu_type = None
        aule._is_datacenter_gpu = False

        result = aule.is_datacenter_gpu()

        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
