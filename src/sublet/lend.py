import ctypes
import gc
import importlib.util
import sys
import warnings
import weakref

import torch

from . import _compat

PAGE = 2 * 1024**2  # the caching allocator requests segments in multiples of 2 MiB

_lib = None
_regions = weakref.WeakValueDictionary()  # address -> _Region
_active = {}  # device index -> Loan


def _load():
    global _lib
    if _lib is None:
        backend = "HIP" if torch.version.hip else "CUDA"
        module = "sublet._allocator_hip" if torch.version.hip else "sublet._allocator"
        spec = importlib.util.find_spec(module)
        if spec is None:
            raise ImportError(
                f"sublet's {backend} allocator library is missing: reinstall with pip"
                + (" using SUBLET_BUILD_HIP=1 ROCM_PATH=/opt/rocm" if torch.version.hip else "")
            )
        lib = ctypes.CDLL(spec.origin)
        if lib.sublet_init() != 0:
            raise RuntimeError(f"sublet could not initialize the {backend} driver")
        p, address = ctypes.c_void_p, ctypes.c_uint64
        lib.sublet_create.argtypes, lib.sublet_create.restype = [ctypes.c_int, ctypes.c_size_t], p
        lib.sublet_destroy.argtypes = [p]
        lib.sublet_base.argtypes, lib.sublet_base.restype = [p], address
        lib.sublet_lent.argtypes = [p, p]
        lib.sublet_backs.argtypes, lib.sublet_backs.restype = [p, address], ctypes.c_int
        lib.sublet_begin.argtypes = [p, p]
        lib.sublet_stop.argtypes = [p]
        lib.sublet_move_out.argtypes, lib.sublet_move_out.restype = [p, p, ctypes.c_int64], ctypes.c_int64
        lib.sublet_blank_out.argtypes = [p]
        lib.sublet_end.argtypes = [p, p, p]
        if lib.sublet_hook() != 0:
            raise RuntimeError(
                f"sublet could not hook PyTorch's {backend} caching allocator: this PyTorch build is not supported"
            )
        _lib = lib
    return _lib


class _Region:
    """The memory behind a tensor made by `empty`, exposed through `__cuda_array_interface__`.

    sublet maps its pages itself, so that loans can hand them out. It lives as long as a view of the tensor, and as long
    as the segments of the borrower's memory made from it.
    """

    def __init__(self, nbytes, device):
        self.handle = None
        lib = _load()
        self._destroy = lib.sublet_destroy  # module globals may be gone at interpreter exit
        self.handle = lib.sublet_create(device.index, nbytes)
        if not self.handle:
            raise torch.OutOfMemoryError(f"sublet could not allocate {nbytes} bytes on {device}")
        self.device, self.npages, self.base = device, nbytes // PAGE, lib.sublet_base(self.handle)
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (self.base, False), "version": 3}

    def lent_pages(self):
        """For inspection: whether each 2 MiB page was lent during the current loan (or the last one, after it)."""
        out = torch.empty(self.npages, dtype=torch.uint8)
        _lib.sublet_lent(self.handle, out.data_ptr())
        return out.bool()

    def backs(self, address):
        """For inspection: whether the memory at `address` is the tensor's, lent to the borrower."""
        return bool(_lib.sublet_backs(self.handle, address))

    def __del__(self):
        if self.handle:
            self._destroy(self.handle)  # once the last segment made from it is freed


def empty(*size, dtype=None, device=None) -> torch.Tensor:
    """Like `torch.empty`, but its memory can be lent with `lend`.

    Args:
        *size (`int` or `tuple` of `int`):
            Shape of the tensor.
        dtype (`torch.dtype`, *optional*):
            Defaults to `torch.get_default_dtype()`.
        device (`torch.device`, `str` or `int`, *optional*):
            CUDA device, defaults to the current one.
    """
    meta = torch.empty(*size, dtype=dtype, device="meta")
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device or "cuda")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _compat.check_supported(device)
    nbytes = meta.numel() * meta.element_size()
    region = _Region(max(1, -(-nbytes // PAGE)) * PAGE, device)
    _regions[region.base] = region
    return torch.as_tensor(region, device=device)[:nbytes].view(meta.dtype).view(meta.shape)


def lend(tensor: torch.Tensor, keep=None) -> "Loan":
    """Lends the memory of `tensor` to every CUDA allocation on its device, from now until the loan ends.

    ```python
    with sublet.lend(cache, keep=cache[3]) as loan:
        ...  # any PyTorch code
    ```

    Or `loan = sublet.lend(cache, keep=cache[3])` and, later, `loan.end()`. The borrower's memory comes from the tensor's
    memory that is not kept, then from free GPU memory, then from kept memory, copied to the CPU first and back when the
    loan ends. Allocations still alive then are moved out of the tensor's memory. The tensor's owner must not use it
    during the loan, and one loan per device runs at a time.

    Args:
        tensor (`torch.Tensor`):
            A tensor made by `empty`, or a view of one: its whole memory is lent.
        keep (`torch.Tensor` or `list` of `torch.Tensor`, *optional*):
            Views of `tensor` to get back, bit-identical, such as `tensor[3]` or `tensor[:, 2:5]`. Defaults to all of it;
            `[]` gets nothing back. The rest of the tensor may have changed.
    """
    return Loan(tensor, keep)


class LeakError(RuntimeError):
    """Allocations made during a loan were still alive when it ended, and the GPU had no memory to move them out of
    the lent tensor."""


class Loan:
    """A loan of a tensor's memory, made by `lend`. After it ends:

    - `copied_bytes`: kept memory the borrower used, copied to the CPU and back.
    - `extra_bytes`: the most GPU memory the borrower took beyond the tensor.
    - `moved_bytes`: memory of allocations still alive at the end, moved out of the tensor.
    """

    def __init__(self, tensor, keep):
        region = _regions.get(tensor.untyped_storage().data_ptr()) if tensor.is_cuda else None
        if region is None:
            raise ValueError("lend needs a tensor made by sublet.empty, or a view of one")
        device = region.device
        if device.index in _active:
            raise RuntimeError(f"a loan is already active on {device}; loans on one device cannot be nested")
        _compat.check_allocator_settings()
        kept = _kept_pages(region, keep).to(torch.uint8)
        torch.cuda.synchronize(device)  # the owner's last kernels may still use the memory
        _lib.sublet_begin(region.handle, kept.data_ptr())
        _active[device.index] = self
        self._region = region
        self.copied_bytes = self.extra_bytes = self.moved_bytes = None

    def end(self):
        """Ends the loan: what `keep` covers is back, and the borrower's allocations still alive are moved out of the
        tensor's memory. Does nothing if the loan has ended already.

        Raises `LeakError` if the GPU has no room to move some of them: they keep their addresses but lose their data.
        While an exception is being handled, in an `except` or `finally` block, it warns instead, not to hide it.
        """
        stuck = self._end()
        if stuck and sys.exc_info()[1] is None:
            raise LeakError(_leak_message(stuck))
        if stuck:
            warnings.warn(_leak_message(stuck), stacklevel=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end()
        return False

    def _end(self):
        region, self._region = self._region, None
        if region is None:
            return []
        device, stuck = region.device, []
        try:
            torch.cuda.synchronize(device)  # the borrower's kernels may still use the memory
            _lib.sublet_stop(region.handle)
            _compat.clear_cublas_workspaces()
            torch.cuda.empty_cache()  # PyTorch gives back the segments it holds nothing in
            stuck = _move_out(region)
        finally:
            out = (ctypes.c_int64 * 4)()
            _lib.sublet_end(region.handle, torch.cuda.current_stream(device).cuda_stream, out)
            del _active[device.index]
        self.copied_bytes, self.extra_bytes, self.moved_bytes, self._gathered_bytes = out
        return stuck


def _move_out(region):
    """Moves the segments still holding pages of the region out of it. Returns the allocations in those that could not
    be, which no longer hold data."""
    stuck = _stuck_segments(region)
    if stuck:  # garbage cycles may hold some of them, and PyTorch may hold GPU memory it does not use
        gc.collect()
        torch.cuda.empty_cache()
        stuck = _stuck_segments(region)
    if not stuck:
        return []
    blocks = [
        block
        for segment in torch.cuda.memory_snapshot()
        if segment["address"] in stuck
        for block in segment["blocks"]
        if block["state"] == "active_allocated"
    ]
    _lib.sublet_blank_out(region.handle)
    return blocks


def _stuck_segments(region):
    """Moves what it can; returns the addresses of the segments it could not move."""
    addresses = (ctypes.c_uint64 * 1024)()
    n = _lib.sublet_move_out(region.handle, addresses, len(addresses))
    return set(addresses[: min(n, len(addresses))]) if n else set()


def _kept_pages(region, keep):
    if keep is None:
        return torch.ones(region.npages, dtype=torch.bool)
    edges = torch.zeros(region.npages + 1, dtype=torch.int64)  # +1 where a kept range starts, -1 after it ends
    for view in [keep] if isinstance(keep, torch.Tensor) else keep:
        if view.untyped_storage().data_ptr() != region.base:
            raise ValueError(
                "keep must hold views of the lent tensor, such as tensor[3] or tensor[:, 2:5] (indexing with a list "
                "or a mask makes a copy)"
            )
        if view.numel():
            starts, length = _byte_runs(view)
            edges.index_add_(0, starts // PAGE, torch.ones_like(starts))
            edges.index_add_(0, (starts + length - 1) // PAGE + 1, -torch.ones_like(starts))
    return edges.cumsum(0)[:-1] > 0


def _byte_runs(view):
    """Offsets in its storage of the contiguous runs of `view`, and their length, in bytes."""
    shape, stride, run = list(view.shape), list(view.stride()), 1
    while shape and (shape[-1] == 1 or stride[-1] == run):
        run *= shape.pop()
        stride.pop()
    starts = torch.tensor([view.storage_offset()])
    for size, step in zip(shape, stride):
        starts = (starts[:, None] + torch.arange(size) * step).flatten()
    return starts * view.element_size(), run * view.element_size()


def _leak_message(stuck):
    lines = [
        f"{len(stuck)} allocation(s) made during the loan ({sum(b['size'] for b in stuck)} bytes) were still alive when "
        "it ended, and the GPU had no memory to move them out of the lent tensor: they keep their addresses but no "
        "longer hold data. Free them before the loan ends, or leave more GPU memory free."
    ]
    for block in stuck[:10]:
        lines.append(f"  - {block['requested_size']} bytes")
        lines += [f"      {f['filename']}:{f['line']} {f['name']}" for f in block.get("frames", []) if f["filename"].endswith(".py")][:6]
    if not any(block.get("frames") for block in stuck):
        lines.append("Call torch.cuda.memory._record_memory_history() before the loan to see where they were allocated.")
    return "\n".join(lines)
