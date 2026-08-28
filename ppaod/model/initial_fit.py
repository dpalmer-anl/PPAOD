"""Online / closed-form least-squares fit of ``c_ml`` to Wannier90 ``.amn``."""

from __future__ import annotations

import torch

from .projection import ProjectionCache


def _accumulate_normal_eqs(
    cache: ProjectionCache,
    A_target: list[torch.Tensor],
    *,
    l: int,
    band_idx: list | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Accumulate ``XᵀX`` and ``Xᵀy`` for one angular momentum ``l``."""
    orb_idx = [n for n, ol in enumerate(cache.orb_l) if ol == l]
    n_basis = cache.bases[l].n_basis
    XtX = torch.zeros((n_basis, n_basis), dtype=torch.float64, device=device)
    xty = torch.zeros((n_basis,), dtype=torch.float64, device=device)
    for ik in range(len(cache.c_win)):
        gk = cache.global_k_of(ik)
        At = A_target[gk]
        if band_idx is not None:
            # band_idx may be full-mesh or local-aligned
            if len(band_idx) == len(cache.c_win):
                At = At[band_idx[ik]]
            else:
                At = At[band_idx[gk]]
        elif cache.band_idx_global is not None:
            At = At[cache.band_idx_global[gk]]
        cw = cache.c_win[ik]
        T = cache.T_l[ik][l]
        for n in orb_idx:
            ang = cache.angular_phase[ik][n]
            # B[m,i] = Σ_G conj(c[m,G]) T[i,G] ang[G]  → design for c_i
            B = torch.einsum(
                "mG,iG,G->mi",
                cw.conj(),
                T.to(dtype=torch.complex128),
                ang,
            )
            target = At[:, n].to(device=device)
            # Stack real/imag as float equations: 2*Nb rows
            Br = B.real.to(dtype=torch.float64)
            Bi = B.imag.to(dtype=torch.float64)
            yr = target.real.to(dtype=torch.float64)
            yi = target.imag.to(dtype=torch.float64)
            XtX = XtX + Br.T @ Br + Bi.T @ Bi
            xty = xty + Br.T @ yr + Bi.T @ yi
    return XtX, xty, n_basis


def fit_theta0_to_amn(
    cache: ProjectionCache,
    A_target: list[torch.Tensor],
    *,
    band_idx: list | None = None,
    ridge: float = 1e-2,
    all_reduce: bool = False,
) -> dict[int, torch.Tensor]:
    """
    Solve per-``l`` ridge linear least squares via normal equations::

        (XᵀX + λI) c = Xᵀ y

    Accumulated online over k (no giant design matrix). When ``all_reduce`` is
    True, ``XᵀX`` / ``Xᵀy`` are summed across ``torch.distributed`` ranks.
    """
    device = cache.c_win[0].device
    unique_l = sorted(set(cache.orb_l))
    theta0: dict[int, torch.Tensor] = {}

    for l in unique_l:
        XtX, xty, n_basis = _accumulate_normal_eqs(
            cache, A_target, l=l, band_idx=band_idx, device=device
        )
        if all_reduce:
            from ..distributed_k import all_reduce_sum_

            all_reduce_sum_(XtX)
            all_reduce_sum_(xty)
        XtX = XtX + ridge * torch.eye(n_basis, dtype=XtX.dtype, device=XtX.device)
        sol = torch.linalg.solve(XtX, xty)
        theta0[l] = sol.detach().to(device=device, dtype=torch.float64)
    return theta0


def amn_match_report(
    theta: dict[int, torch.Tensor],
    cache: ProjectionCache,
    A_target: list[torch.Tensor],
    band_idx: list,
) -> dict[str, float]:
    """RMS / max |A(θ)−A_target| over windowed bands (sanity check for Step 2)."""
    from .projection import build_A

    errs = []
    for ik in range(len(cache.c_win)):
        A = build_A(theta, cache, ik)
        gk = cache.global_k_of(ik)
        if len(band_idx) == len(cache.c_win):
            At = A_target[gk][band_idx[ik]].to(device=A.device)
        else:
            At = A_target[gk][band_idx[gk]].to(device=A.device)
        errs.append((A - At).abs().reshape(-1))
    e = torch.cat(errs)
    out = {
        "rms": float(torch.sqrt(torch.mean(e**2)).item()),
        "max": float(e.max().item()),
        "mean": float(e.mean().item()),
    }
    # Optional distributed mean of rms² weighted by n — keep local for reports
    return out
