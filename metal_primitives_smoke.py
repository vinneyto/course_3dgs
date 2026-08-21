import numpy as np
import torch

from metal_renderer import _exclusive_scan_int32, _lsd_radix_sort


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal smoke checks require MPS")

    device = torch.device("mps")

    counts = torch.tensor([3, 0, 7, 2, 1], dtype=torch.int32, device=device)
    scanned = _exclusive_scan_int32(counts)
    expected_scan = torch.tensor([0, 3, 3, 10, 12], dtype=torch.int32)
    torch.testing.assert_close(scanned.cpu(), expected_scan)

    tile_cpu = np.array([2, 0, 2, 1, 0, 1, 2, 0], dtype=np.int32)
    depth_cpu = np.array([4.0, 3.0, 1.0, 8.0, 2.0, 1.5, 2.0, 1.0], dtype=np.float32)
    ids_cpu = np.arange(tile_cpu.size, dtype=np.int32)

    tile_ids = torch.from_numpy(tile_cpu).to(device)
    depth_bits = torch.from_numpy(depth_cpu.view(np.int32)).to(device)
    gaussian_ids = torch.from_numpy(ids_cpu).to(device)

    sorted_tile, _, sorted_ids, _ = _lsd_radix_sort(
        tile_ids,
        depth_bits,
        gaussian_ids,
        num_tiles=3,
    )

    expected_ids = sorted(
        range(tile_cpu.size),
        key=lambda i: (int(tile_cpu[i]), float(depth_cpu[i])),
    )

    assert sorted_ids.cpu().tolist() == expected_ids
    assert sorted_tile.cpu().tolist() == [int(tile_cpu[i]) for i in expected_ids]

    print("Metal exclusive scan smoke check passed")
    print("Metal radix sort smoke check passed")


if __name__ == "__main__":
    main()
