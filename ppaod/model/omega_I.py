"""Differentiable gauge-invariant spread ``Ω_I`` (Souza–Marzari–Vanderbilt)."""

from __future__ import annotations

import torch

from ..file_io.wannier90_files import WannierDataset
from .projection import ProjectionCache, build_A, check_isometry, lowdin


def build_V_list(
    theta: dict[int, torch.Tensor],
    cache: ProjectionCache,
    *,
    check_V: bool = False,
    isometry_tol: float = 1e-8,
) -> list[torch.Tensor]:
    """Löwdin isometries ``V(k)`` for every k stored in ``cache`` (local shard)."""
    V_list: list[torch.Tensor] = []
    for ik_local in range(len(cache.c_win)):
        A = build_A(theta, cache, ik_local)
        V = lowdin(A)
        if check_V:
            err = check_isometry(V)
            if err > isometry_tol and err > 1e-5:
                gk = cache.global_k_of(ik_local)
                raise RuntimeError(f"V†V ≠ I at k={gk}: ||err||_F={err:.3e}")
        V_list.append(V)
    return V_list


def omega_I_from_V(
    V_full: list[torch.Tensor],
    data: WannierDataset,
    *,
    mmn_win: dict[tuple[int, int], torch.Tensor] | None = None,
    band_idx: list | None = None,
    k_owners: list[int] | None = None,
    Nk_global: int | None = None,
) -> torch.Tensor:
    """
    Gauge-invariant SMV ``Ω_I`` from a full-mesh ``V`` list.

        Ω_I = (1/N_k) Σ_{k,b} w_b [ J − ‖ V_k† M^{k,b} V_{k+b} ‖_F² ]

    equivalent to ``Tr(P M P M†)`` with ``P = V V†``, without forming ``P``.

    Parameters
    ----------
    k_owners :
        If set, only sum edges whose home k-index is in ``k_owners``
        (distributed partition; each edge counted once).
    Nk_global :
        Global mesh size for the ``1/N_k`` factor (defaults to ``len(V_full)``).
    """
    Nk = Nk_global if Nk_global is not None else len(V_full)
    J = V_full[0].shape[1]
    device = V_full[0].device
    total = torch.zeros((), dtype=torch.float64, device=device)
    owners = set(k_owners) if k_owners is not None else None

    for ik in range(len(V_full)):
        if owners is not None and ik not in owners:
            continue
        for b_idx, kb in data.neighbors[ik]:
            if mmn_win is not None:
                Mkb = mmn_win[(ik, b_idx)]
            else:
                if band_idx is None:
                    raise ValueError("band_idx required when mmn_win is None")
                idx_k = band_idx[ik]
                idx_kb = band_idx[kb]
                Mfull = data.mmn[(ik, b_idx)]
                Mkb = Mfull[idx_k][:, idx_kb]
            wb = data.weights[b_idx]
            # Thin form: Tr(P M P M†) = ‖V† M V'‖_F²
            Mtilde = V_full[ik].conj().T @ Mkb.to(device=device) @ V_full[kb]
            term = (Mtilde.abs() ** 2).sum().real
            total = total + wb * (J - term)
    return total / Nk


def omega_I(
    theta: dict[int, torch.Tensor],
    cache: ProjectionCache,
    data: WannierDataset,
    *,
    mmn_win: dict[tuple[int, int], torch.Tensor] | None = None,
    check_V: bool = False,
    isometry_tol: float = 1e-8,
    V_full: list[torch.Tensor] | None = None,
    k_owners: list[int] | None = None,
    Nk_global: int | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Differentiable ``Ω_I(θ)`` and Löwdin isometries.

    When ``V_full`` is provided (e.g. after distributed all-gather), only the
    local edges in ``k_owners`` contribute to the sum. Otherwise ``V`` is built
    from ``cache`` for all local k and treated as the full mesh.
    """
    if V_full is None:
        V_local = build_V_list(
            theta, cache, check_V=check_V, isometry_tol=isometry_tol
        )
        # Map local cache slots → global V list (contiguous single-rank default)
        if cache.global_k_indices is None:
            V_full = V_local
            owners = k_owners
        else:
            Nk = Nk_global if Nk_global is not None else (max(cache.global_k_indices) + 1)
            # Placeholder list; only local entries are real tensors with grad
            V_full = V_local  # serial path when indices are a permutation of range
            if list(cache.global_k_indices) != list(range(len(V_local))):
                raise RuntimeError(
                    "Sharded cache requires pre-gathered V_full; call omega_P "
                    "with distributed all_gather first."
                )
            owners = k_owners
        oI = omega_I_from_V(
            V_full,
            data,
            mmn_win=mmn_win,
            band_idx=cache.band_idx_global,
            k_owners=owners,
            Nk_global=Nk_global,
        )
        return oI, V_full

    oI = omega_I_from_V(
        V_full,
        data,
        mmn_win=mmn_win,
        band_idx=cache.band_idx_global,
        k_owners=k_owners,
        Nk_global=Nk_global,
    )
    return oI, V_full


def slice_mmn_to_window(
    data: WannierDataset,
    band_idx: list,
    *,
    k_owners: list[int] | None = None,
) -> dict[tuple[int, int], torch.Tensor]:
    """
    Restrict MMN blocks to outer-window band indices at each (k,b).

    If ``k_owners`` is set, only edges with home k in that set are kept.
    """
    out: dict[tuple[int, int], torch.Tensor] = {}
    Nk = len(band_idx)
    owners = set(k_owners) if k_owners is not None else None
    for ik in range(Nk):
        if owners is not None and ik not in owners:
            continue
        for b_idx, kb in data.neighbors[ik]:
            idx_k = band_idx[ik]
            idx_kb = band_idx[kb]
            M = data.mmn[(ik, b_idx)]
            out[(ik, b_idx)] = M[idx_k][:, idx_kb]
    return out
