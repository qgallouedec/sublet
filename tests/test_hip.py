"""Bounded regressions for HIP virtual-memory aliases on large-BAR hardware."""
import pytest
import torch

import sublet


@pytest.mark.skipif(not torch.version.hip, reason="HIP backend regression")
@pytest.mark.parametrize("size", [64 * 1024**2, 256 * 1024**2])
def test_hip_scalar_reads_and_survivors_across_repeated_loans(size):
    cache = sublet.empty(size, dtype=torch.uint8).fill_(37)
    for _ in range(3):
        torch.cuda.empty_cache()  # request a fresh segment from the hook on every loan
        with sublet.lend(cache, keep=cache[:size // 2]) as loan:
            borrower = torch.empty(size // 2, dtype=torch.uint8, device="cuda").fill_(23)
            # PyTorch directly dereferences caching-allocator memory on large-BAR GPUs.
            assert borrower[0].item() == 23
            assert borrower[:4].view(torch.float32).fill_(1.5).item() == 1.5
        # Live allocations move out of the cache at the end without changing addresses.
        assert loan.moved_bytes == size // 2
        assert bool((borrower[4:] == 23).all())
        assert borrower[:4].view(torch.float32).item() == 1.5
        assert bool((cache[:size // 2] == 37).all())
        del borrower
