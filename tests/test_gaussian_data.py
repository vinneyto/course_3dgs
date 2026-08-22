import unittest

import torch

from metal_renderer.data import GaussianData
from util import evaluate_sh


class GaussianDataTest(unittest.TestCase):
    def test_flat_level_four_layout_matches_existing_sh_evaluation(self):
        torch.manual_seed(0)
        n = 5
        positions = torch.randn(n, 3, dtype=torch.float32)
        f_dc = torch.randn(n, 3, dtype=torch.float32)
        f_rest = torch.randn(n, 45, dtype=torch.float32)
        opacity_raw = torch.randn(n, 1, dtype=torch.float32)
        sigma = torch.eye(3, dtype=torch.float32).expand(n, -1, -1).clone()
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, 3] = torch.tensor([0.3, -0.2, 0.1])

        data = GaussianData.from_flat_tensors(
            positions=positions,
            f_dc=f_dc,
            f_rest=f_rest,
            opacity_raw=opacity_raw,
            sigma=sigma,
        )

        self.assertEqual(data.sh_levels, 4)
        self.assertEqual(data.sh_coefficients.shape, (n, 16, 3))
        torch.testing.assert_close(
            data.evaluate_color(c2w),
            evaluate_sh(f_dc, f_rest, positions, c2w),
        )

    def test_lower_level_selects_prefix_from_each_color_block(self):
        positions = torch.zeros(1, 3, dtype=torch.float32)
        f_dc = torch.tensor([[100.0, 200.0, 300.0]], dtype=torch.float32)
        f_rest = torch.arange(45, dtype=torch.float32).reshape(1, 45)
        opacity_raw = torch.zeros(1, 1, dtype=torch.float32)
        sigma = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)

        data = GaussianData.from_flat_tensors(
            positions=positions,
            f_dc=f_dc,
            f_rest=f_rest,
            opacity_raw=opacity_raw,
            sigma=sigma,
            sh_levels=2,
        )

        expected = torch.tensor(
            [
                [100.0, 200.0, 300.0],
                [0.0, 15.0, 30.0],
                [1.0, 16.0, 31.0],
                [2.0, 17.0, 32.0],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(data.sh_coefficients[0], expected)


if __name__ == "__main__":
    unittest.main()
