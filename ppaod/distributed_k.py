"""
``torch.distributed`` helpers for k-point data parallelism.

Launch with::

    torchrun --nproc_per_node=N --module ppaod.run_ppaod ...

Uses NCCL when CUDA is available on the process device, else gloo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str


def is_dist_env() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def init_dist(device_str: str = "cpu") -> DistInfo:
    """
    Initialize the process group from ``torchrun`` env vars.

    When not launched under ``torchrun`` / ``WORLD_SIZE==1``, returns a
    single-process stub (no ``init_process_group``).
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world <= 1:
        if device_str.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(device_str if ":" in device_str else "cuda:0")
        else:
            device = torch.device("cpu" if device_str == "cpu" else device_str)
        return DistInfo(
            enabled=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device=device,
            backend="none",
        )

    use_cuda = device_str.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistInfo(
        enabled=True,
        rank=rank,
        world_size=world,
        local_rank=local_rank,
        device=device,
        backend=backend,
    )


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def destroy_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def shard_kpoints(Nk: int, rank: int, world_size: int) -> list[int]:
    """Contiguous block sharding of ``range(Nk)``."""
    if world_size <= 1:
        return list(range(Nk))
    # Even split with remainder on lower ranks
    base = Nk // world_size
    rem = Nk % world_size
    start = rank * base + min(rank, rem)
    n = base + (1 if rank < rem else 0)
    return list(range(start, start + n))


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce; no-op if not distributed."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_max_int(value: int, device: torch.device) -> int:
    t = torch.tensor([value], dtype=torch.int64, device=device)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def all_gather_v_list(
    V_local: list[torch.Tensor],
    global_k_indices: list[int],
    Nk_global: int,
    *,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """
    All-gather Löwdin ``V`` matrices onto every rank with autograd support.

    Returns a Python list ``V_full[ik]`` of shape ``(Nb[ik], J)`` for
    ``ik = 0…Nk_global-1``. Gradients flow back only into locally owned
    entries of ``V_local``.
    """
    if not V_local:
        raise ValueError("V_local must be non-empty on every rank that owns k")

    device = device or V_local[0].device
    J = V_local[0].shape[1]
    world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

    if world == 1:
        # Identity: ensure list is indexed by global k
        if list(global_k_indices) == list(range(Nk_global)):
            return V_local
        out: list[torch.Tensor | None] = [None] * Nk_global
        for V, gk in zip(V_local, global_k_indices):
            out[gk] = V
        assert all(v is not None for v in out)
        return out  # type: ignore[return-value]

    # Pad local V to common max_Nb (global)
    Nb_loc = [V.shape[0] for V in V_local]
    max_local = max(Nb_loc)
    max_Nb_t = torch.tensor([max_local], dtype=torch.int64, device=device)
    dist.all_reduce(max_Nb_t, op=dist.ReduceOp.MAX)
    max_Nb = int(max_Nb_t.item())

    V_pad = torch.zeros(
        (len(V_local), max_Nb, J),
        dtype=V_local[0].dtype,
        device=device,
    )
    for i, V in enumerate(V_local):
        nb = V.shape[0]
        V_pad[i, :nb, :] = V

    # Autograd-friendly all_gather on the padded stack
    V_pad_full = _AllGatherPadded.apply(V_pad, Nk_global, global_k_indices)

    # Nb per global k via object gather (metadata only)
    meta = {"idx": list(global_k_indices), "Nb": Nb_loc}
    metas: list[dict] = [None] * world  # type: ignore[list-item]
    dist.all_gather_object(metas, meta)
    Nb_all = [0] * Nk_global
    for m in metas:
        for gk, nb in zip(m["idx"], m["Nb"]):
            Nb_all[gk] = nb

    V_full: list[torch.Tensor] = []
    for ik in range(Nk_global):
        nb = Nb_all[ik]
        V_full.append(V_pad_full[ik, :nb, :])
    return V_full


class _AllGatherPadded(torch.autograd.Function):
    """All-gather ``(n_local, max_Nb, J)`` into ``(Nk, max_Nb, J)`` with grads."""

    @staticmethod
    def forward(
        ctx,
        V_pad_local: torch.Tensor,
        Nk_global: int,
        global_k_indices: list[int],
    ) -> torch.Tensor:
        world = dist.get_world_size()
        rank = dist.get_rank()
        device = V_pad_local.device
        dtype = V_pad_local.dtype
        max_Nb = V_pad_local.shape[1]
        J = V_pad_local.shape[2]

        pack = {
            "V": V_pad_local.detach().cpu(),
            "idx": list(global_k_indices),
            "rank": rank,
        }
        gather_list: list[dict] = [None] * world  # type: ignore[list-item]
        dist.all_gather_object(gather_list, pack)

        V_full = torch.zeros((Nk_global, max_Nb, J), dtype=dtype, device=device)
        for p in gather_list:
            Vp = p["V"].to(device=device, dtype=dtype)
            for i_loc, gk in enumerate(p["idx"]):
                V_full[gk] = Vp[i_loc]

        ctx.global_k_indices = list(global_k_indices)
        ctx.Nk_global = Nk_global
        ctx.max_Nb = max_Nb
        ctx.J = J
        ctx.n_local = V_pad_local.shape[0]
        return V_full

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor):
        g = grad_full.contiguous()
        if g.is_complex():
            gv = torch.view_as_real(g).contiguous()
            dist.all_reduce(gv, op=dist.ReduceOp.SUM)
            g = torch.view_as_complex(gv)
        else:
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
        grads = [g[gk] for gk in ctx.global_k_indices]
        if grads:
            g_local = torch.stack(grads, dim=0)
        else:
            g_local = torch.zeros(
                (0, ctx.max_Nb, ctx.J),
                dtype=grad_full.dtype,
                device=grad_full.device,
            )
        return g_local, None, None


def is_rank0(info: DistInfo | None = None) -> bool:
    if info is None:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return int(os.environ.get("RANK", "0")) == 0
    return info.rank == 0


def print0(*args, info: DistInfo | None = None, **kwargs) -> None:
    if is_rank0(info):
        print(*args, **kwargs)
