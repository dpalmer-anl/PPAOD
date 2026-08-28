"""Adam optimization of radial coefficients ``c_ml`` for ``Ω_P = α (Ω_I/J) + (1−α) Ω_D``."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .distributed_k import (
    DistInfo,
    all_gather_v_list,
    all_reduce_sum_,
    is_rank0,
    print0,
)
from .file_io.wannier90_files import WannierDataset
from .model.omega_D import EnergyGrid, omega_P
from .model.omega_I import build_V_list
from .model.projection import ProjectionCache, max_condition_number


@dataclass
class OptimizeResult:
    theta: dict[int, torch.Tensor]
    history: list[float] = field(default_factory=list)
    history_I: list[float] = field(default_factory=list)
    history_D: list[float] = field(default_factory=list)
    V_list: list[torch.Tensor] = field(default_factory=list)
    omega_final: float = float("nan")
    omega_I_final: float = float("nan")
    omega_D_final: float = float("nan")
    steps: int = 0


def _converged(history: list[float], tol: float, window: int = 5) -> bool:
    if len(history) < window + 1:
        return False
    return abs(history[-1] - history[-1 - window]) < tol


def _eval_omega_P(
    theta: dict[int, torch.Tensor],
    cache: ProjectionCache,
    data: WannierDataset,
    mmn_win: dict[tuple[int, int], torch.Tensor],
    grid: EnergyGrid,
    *,
    alpha: float,
    check_V: bool,
    dist_info: DistInfo | None,
    Nk_global: int,
    k_owners: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """
    Returns ``(loss_for_backward, Ω_I/J_global, Ω_D_global, V_full)``.

    Under DDP, ``loss_for_backward`` is the *local partitioned* contribution
    (already scaled by ``1/N_k``); after ``backward``, callers must SUM-allreduce
    ``θ.grad``. Logged Ω values are global (SUM of partitions).
    """
    V_local = build_V_list(theta, cache, check_V=check_V)
    if dist_info is not None and dist_info.enabled:
        V_full = all_gather_v_list(
            V_local,
            list(cache.global_k_indices or k_owners),
            Nk_global,
            device=dist_info.device,
        )
        oP_loc, oI_loc, oD_loc, V_full = omega_P(
            theta,
            cache,
            data,
            mmn_win=mmn_win,
            grid=grid,
            alpha=alpha,
            check_V=False,
            V_full=V_full,
            k_owners=k_owners,
            Nk_global=Nk_global,
        )
        # Global metrics for logging (detached)
        oI = all_reduce_sum_(oI_loc.detach().clone())
        oD = all_reduce_sum_(oD_loc.detach().clone())
        oP_log = all_reduce_sum_(oP_loc.detach().clone())
        # Stash global Ω_P on the local tensor for history via .detach later;
        # backward uses oP_loc.
        oP_loc_for_bw = oP_loc
        # Attach global value for reading without affecting graph:
        oP_loc_for_bw_ret = oP_loc_for_bw
        # Caller reads float(loss) after sync — return oP_log for history and
        # oP_loc for backward. We return a pair encoded as: loss_bw, and store
        # globals in oI/oD; for history value we allreduce separately in
        # optimize loop.
        return oP_loc_for_bw_ret, oI, oD, V_full

    return omega_P(
        theta,
        cache,
        data,
        mmn_win=mmn_win,
        grid=grid,
        alpha=alpha,
        check_V=check_V,
        Nk_global=Nk_global,
        k_owners=k_owners if k_owners else None,
    )



def optimize_theta(
    theta0: dict[int, torch.Tensor],
    cache: ProjectionCache,
    data: WannierDataset,
    mmn_win: dict[tuple[int, int], torch.Tensor],
    *,
    grid: EnergyGrid,
    alpha: float = 0.8,
    lr: float = 1e-2,
    max_steps: int = 200,
    tol: float = 1e-8,
    l2: float = 1e-6,
    grad_clip: float = 10.0,
    log_every: int = 5,
    cond_warn: float = 1e6,
    dist_info: DistInfo | None = None,
    Nk_global: int | None = None,
    k_owners: list[int] | None = None,
    compile_proj: bool = False,
) -> OptimizeResult:
    """
    Minimize ``Ω_P(θ) + l2 ||θ||²`` with Adam + ReduceLROnPlateau.

    Under ``torch.distributed``, k-points are sharded; ``V`` is all-gathered
    each step with correct autograd; ``θ.grad`` is SUM-allreduced.
    """
    del compile_proj  # reserved for optional torch.compile(build_A)
    Nk = Nk_global if Nk_global is not None else len(data.amn)
    owners = (
        k_owners
        if k_owners is not None
        else (
            list(cache.global_k_indices)
            if cache.global_k_indices is not None
            else list(range(len(cache.c_win)))
        )
    )

    theta = {
        l: torch.nn.Parameter(c0.detach().clone().to(dtype=torch.float64))
        for l, c0 in theta0.items()
    }
    opt = torch.optim.Adam(theta.values(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-5
    )

    history: list[float] = []
    history_I: list[float] = []
    history_D: list[float] = []
    V_list: list[torch.Tensor] = []
    for step in range(max_steps):
        opt.zero_grad(set_to_none=True)
        loss, oI, oD, V_list = _eval_omega_P(
            theta,
            cache,
            data,
            mmn_win,
            grid,
            alpha=alpha,
            check_V=(step % log_every == 0),
            dist_info=dist_info,
            Nk_global=Nk,
            k_owners=owners,
        )
        reg = sum((c**2).sum() for c in theta.values())
        if dist_info is not None and dist_info.enabled:
            total = loss + (l2 / dist_info.world_size) * reg
        else:
            total = loss + l2 * reg
        total.backward()
        # Partitioned loss ⇒ partial θ grads; SUM to recover the global gradient.
        if dist_info is not None and dist_info.enabled:
            for p in theta.values():
                if p.grad is not None:
                    all_reduce_sum_(p.grad)
        torch.nn.utils.clip_grad_norm_(list(theta.values()), grad_clip)
        opt.step()

        # Global Ω for history/logging
        if dist_info is not None and dist_info.enabled:
            val = float(all_reduce_sum_(loss.detach().clone()).item())
        else:
            val = float(loss.detach().item())
        history.append(val)
        history_I.append(float(oI.detach().item()))
        history_D.append(float(oD.detach().item()))
        sched.step(val)

        if step % log_every == 0 or step == max_steps - 1:
            cond = max_condition_number(theta, cache)
            print0(
                f"step {step:4d}  Ω_P={val:.8f}  Ω_I/J={history_I[-1]:.8f}  "
                f"Ω_D={history_D[-1]:.8f}  lr={opt.param_groups[0]['lr']:.2e}  "
                f"cond(A†A)≤{cond:.2e}",
                info=dist_info,
            )
            if cond > cond_warn and is_rank0(dist_info):
                print(f"  WARNING: condition number {cond:.2e} exceeds {cond_warn:.0e}")

        if _converged(history, tol):
            print0(
                f"Converged at step {step} (|ΔΩ_P| < {tol} over window)",
                info=dist_info,
            )
            break

    theta_final = {l: p.detach().clone() for l, p in theta.items()}
    with torch.no_grad():
        oP_loc, oI, oD, V_list = _eval_omega_P(
            theta_final,
            cache,
            data,
            mmn_win,
            grid,
            alpha=alpha,
            check_V=True,
            dist_info=dist_info,
            Nk_global=Nk,
            k_owners=owners,
        )
        if dist_info is not None and dist_info.enabled:
            oP = all_reduce_sum_(oP_loc.detach().clone())
        else:
            oP = oP_loc
    return OptimizeResult(
        theta=theta_final,
        history=history,
        history_I=history_I,
        history_D=history_D,
        V_list=[V.detach().cpu() for V in V_list],
        omega_final=float(oP.item()),
        omega_I_final=float(oI.item()),
        omega_D_final=float(oD.item()),
        steps=len(history),
    )
