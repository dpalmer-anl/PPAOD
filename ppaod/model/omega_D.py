"""
Differentiable LPDOS mismatch ``Ω_D`` and combined objective ``Ω_P``.

DFT local / orbital-projected DOS at k (broadened)::

    LPDOS_DFT_{i k}(E) = Σ_m |V_{m i}(k)|² δ_σ(E − ε_{m k}^{DFT})

TB / disentangled Mulliken LPDOS from ``H(k) = V† E V``::

    H(k) C = C diag(ε^{TB}),   LPDOS_TB_{i k}(E) = Σ_ν |C_{i ν}(k)|² δ_σ(E − ε_ν^{TB})

TB peak positions use the Hellmann–Feynman surrogate (see
``hellmann_feynman_eigh``) so gradients do not flow through ``linalg.eigh``,
matching ``fit_tetb_pdos.py``.

Then::

    Ω_D = (1/(N_k J)) Σ_{k,i} KL( LPDOS_TB_{ik} \\| LPDOS_DFT_{ik} )

where each LPDOS is normalized to a unit PDF (∫ ρ dE = 1) before the KL.

For the multiobjective loss, Ω_I is reported / used **per Wannier**
(``Ω_I^{tot}/J``), matching mean-KL ``Ω_D`` at O(1) across system sizes::

    Ω_P = α (Ω_I^{tot}/J) + (1 − α) Ω_D
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from ..file_io.wannier90_files import WannierDataset
from .omega_I import build_V_list, omega_I_from_V
from .projection import ProjectionCache


@dataclass
class EnergyGrid:
    """Shared energy mesh (eV) for broadened LPDOS / KL."""

    E: torch.Tensor  # (nE,)
    dE: float
    sigma: float


def make_energy_grid(
    eig: NDArray[np.float64],
    band_idx: list[NDArray[np.int64]],
    *,
    sigma: float = 0.15,
    pad: float = 1.0,
    nE: int = 256,
    device: torch.device | None = None,
) -> EnergyGrid:
    """Build a uniform energy grid spanning the outer-window DFT eigenvalues."""
    emin = min(float(eig[ik, idx].min()) for ik, idx in enumerate(band_idx))
    emax = max(float(eig[ik, idx].max()) for ik, idx in enumerate(band_idx))
    E = torch.linspace(emin - pad, emax + pad, nE, dtype=torch.float64, device=device)
    dE = float(E[1] - E[0])
    return EnergyGrid(E=E, dE=dE, sigma=sigma)


def gaussian_dos(
    energies: torch.Tensor,
    weights: torch.Tensor,
    grid: EnergyGrid,
) -> torch.Tensor:
    """
    Broadened DOS ``Σ_j w_j δ_σ(E − ε_j)`` on ``grid.E``.

    Parameters
    ----------
    energies : (nb,)
    weights : (nb,) non-negative (need not sum to 1)

    Returns
    -------
    (nE,)
    """
    x = (grid.E[:, None] - energies[None, :]) / grid.sigma
    g = torch.exp(-0.5 * x * x) / (grid.sigma * np.sqrt(2.0 * np.pi))
    return (g * weights[None, :]).sum(dim=1)


def gaussian_dos_batched(
    energies: torch.Tensor,
    weights: torch.Tensor,
    grid: EnergyGrid,
) -> torch.Tensor:
    """
    Batched Gaussian DOS.

    Parameters
    ----------
    energies : (nb,)
    weights : (n_orb, nb)

    Returns
    -------
    rho : (n_orb, nE)
    """
    # g: (nE, nb); weights: (J, nb) → (J, nE)
    x = (grid.E[:, None] - energies[None, :]) / grid.sigma
    g = torch.exp(-0.5 * x * x) / (grid.sigma * np.sqrt(2.0 * np.pi))
    return weights.to(dtype=torch.float64) @ g.T.to(dtype=torch.float64)


def normalize_pdf(rho: torch.Tensor, dE: float, eps: float = 1e-12) -> torch.Tensor:
    """Normalize a non-negative density to a discrete PDF (sum ρ dE = 1)."""
    if rho.ndim == 1:
        z = (rho.clamp_min(0.0).sum() * dE).clamp_min(eps)
        return rho.clamp_min(0.0) / z
    z = (rho.clamp_min(0.0).sum(dim=-1, keepdim=True) * dE).clamp_min(eps)
    return rho.clamp_min(0.0) / z


def kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dE: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """``KL(p\\|q) = Σ p log(p/q) dE`` for normalized PDFs on a uniform mesh."""
    p_ = p.clamp_min(eps)
    q_ = q.clamp_min(eps)
    if p_.ndim == 1:
        p_ = p_ / (p_.sum() * dE)
        q_ = q_ / (q_.sum() * dE)
        return torch.sum(p_ * torch.log(p_ / q_)) * dE
    # (J, nE)
    p_ = p_ / (p_.sum(dim=-1, keepdim=True) * dE)
    q_ = q_ / (q_.sum(dim=-1, keepdim=True) * dE)
    return torch.sum(p_ * torch.log(p_ / q_), dim=-1) * dE


def hellmann_feynman_eigh(
    H: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hermitian eigensolve with Hellmann–Feynman energy surrogate.

    Matches ``fit_tetb_pdos._tb_refine_forward_evecs`` for ``S = I``::

        (ε0, C) = eigh(H)          # eigenvectors / values detached
        E_surr  = ε0 + ⟨C|H|C⟩ − ε0⟨C|C⟩   # live H, frozen C

    Gradients of ``E_surr`` w.r.t. ``H`` follow HF (``|c⟩⟨c|``) and do **not**
    route through ``linalg.eigh``. Mulliken weights should use the detached
    ``C`` so the LPDOS weight graph also avoids eigh autograd.

    Returns
    -------
    E_surr : (J,) real
    C_det : (J, J) complex, columns = eigenvectors (detached)
    """
    Hh = 0.5 * (H + H.conj().T)
    evals, evecs = torch.linalg.eigh(Hh.detach())
    E0 = evals.real.detach()
    C_det = evecs.detach()
    HC = Hh @ C_det
    diag_HC = torch.sum(C_det.conj() * HC, dim=0).real
    diag_CC = torch.sum(C_det.conj() * C_det, dim=0).real
    E_surr = E0 + (diag_HC - E0 * diag_CC)
    return E_surr, C_det


def lpdos_dft_tb_at_k(
    V: torch.Tensor,
    eps_dft: torch.Tensor,
    grid: EnergyGrid,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-orbital LPDOS arrays at one k (vectorized over orbitals).

    Returns
    -------
    rho_dft : (J, nE)
    rho_tb : (J, nE)
    eps_tb : (J,)  (HF surrogate)
    C : (J, J) detached eigenvectors (columns)
    """
    eps = eps_dft.to(dtype=V.dtype)
    Hk = V.conj().T @ (eps[:, None] * V)
    eps_tb, C = hellmann_feynman_eigh(Hk)
    eps_dft_r = eps_dft.to(dtype=torch.float64)
    # weights_dft[i, m] = |V[m, i]|²  → shape (J, Nb)
    w_dft = (V.abs() ** 2).real.to(dtype=torch.float64).T
    # weights_tb[i, ν] = |C[i, ν]|²  (rows = orbitals, cols = eigenstates)
    w_tb = (C.abs() ** 2).real.to(dtype=torch.float64)
    rho_dft = gaussian_dos_batched(eps_dft_r, w_dft, grid)
    rho_tb = gaussian_dos_batched(eps_tb, w_tb, grid)
    return rho_dft, rho_tb, eps_tb, C


def omega_D(
    V_list: list[torch.Tensor],
    data: WannierDataset,
    band_idx: list[NDArray[np.int64]],
    grid: EnergyGrid,
    *,
    k_indices: list[int] | None = None,
    Nk_global: int | None = None,
    n_terms_global: int | None = None,
) -> torch.Tensor:
    """
    Mean KL over subspaces::

        Ω_D = (1/(N_k J)) Σ_{k,i} KL(LPDOS_TB_{ik} \\| LPDOS_DFT_{ik})

    Parameters
    ----------
    V_list :
        Either full-mesh V's, or a local shard (then pass ``k_indices``).
    k_indices :
        Global k index for each entry of ``V_list``. Default: ``range(len(V_list))``.
    n_terms_global :
        Denominator ``N_k J``. Defaults to ``len(k_indices)*J`` for a full local
        mesh, or ``Nk_global * J`` when provided (distributed partition).
    """
    device = V_list[0].device
    if k_indices is None:
        k_indices = list(range(len(V_list)))
    total = torch.zeros((), dtype=torch.float64, device=device)
    n_local = 0
    J = V_list[0].shape[1]
    for V, ik in zip(V_list, k_indices):
        idx = band_idx[ik]
        eps = torch.as_tensor(data.eig[ik, idx], dtype=torch.float64, device=V.device)
        rho_dft, rho_tb, _, _ = lpdos_dft_tb_at_k(V, eps, grid)
        p = normalize_pdf(rho_tb, grid.dE)
        q = normalize_pdf(rho_dft, grid.dE)
        kls = kl_divergence(p, q, grid.dE)  # (J,)
        total = total + kls.sum()
        n_local += int(J)
    if n_terms_global is not None:
        denom = max(n_terms_global, 1)
    elif Nk_global is not None:
        denom = max(Nk_global * J, 1)
    else:
        denom = max(n_local, 1)
    return total / denom


def omega_P(
    theta: dict[int, torch.Tensor],
    cache: ProjectionCache,
    data: WannierDataset,
    *,
    mmn_win: dict[tuple[int, int], torch.Tensor],
    grid: EnergyGrid,
    alpha: float = 0.8,
    check_V: bool = False,
    V_full: list[torch.Tensor] | None = None,
    k_owners: list[int] | None = None,
    Nk_global: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """
    Combined loss ``Ω_P = α (Ω_I^{tot}/J) + (1−α) Ω_D``.

    Returns ``(Ω_P, Ω_I/J, Ω_D, V_list)``.
    For distributed runs pass gathered ``V_full`` and ``k_owners``.
    """
    if V_full is None:
        V_local = build_V_list(theta, cache, check_V=check_V)
        if cache.global_k_indices is not None and list(cache.global_k_indices) != list(
            range(len(V_local))
        ):
            raise RuntimeError(
                "Sharded ProjectionCache requires V_full from all_gather_v_list"
            )
        V_full = V_local
        owners = k_owners
        k_for_D = list(range(len(V_full)))
        V_for_D = V_full
    else:
        owners = k_owners
        if owners is None:
            owners = (
                list(cache.global_k_indices)
                if cache.global_k_indices is not None
                else list(range(len(V_full)))
            )
        V_for_D = [V_full[ik] for ik in owners]
        k_for_D = list(owners)

    Nk = Nk_global if Nk_global is not None else len(V_full)
    oI_tot = omega_I_from_V(
        V_full,
        data,
        mmn_win=mmn_win,
        band_idx=cache.band_idx_global,
        k_owners=owners,
        Nk_global=Nk,
    )
    oD = omega_D(
        V_for_D,
        data,
        cache.band_idx_global if cache.band_idx_global is not None else cache.band_idx,
        grid,
        k_indices=k_for_D,
        Nk_global=Nk,
    )
    J = max(int(V_full[0].shape[1]), 1)
    oI = oI_tot / J
    oP = alpha * oI + (1.0 - alpha) * oD
    return oP, oI, oD, V_full


def channel_mask(orb_l: list[int], channel: str) -> list[int]:
    """Orbital indices for ``'s'`` (l=0) or ``'p'`` (l=1)."""
    if channel == "s":
        return [i for i, l in enumerate(orb_l) if l == 0]
    if channel == "p":
        return [i for i, l in enumerate(orb_l) if l == 1]
    raise ValueError(channel)


def find_mesh_k_index(kpts: NDArray[np.float64], target: NDArray[np.float64]) -> int:
    """Nearest mesh k (crystal coords), accounting for periodic images in [-0.5,0.5)."""

    def wrap(dk: np.ndarray) -> np.ndarray:
        return dk - np.round(dk)

    d2 = np.sum(wrap(kpts - target[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def lpdos_channel_curves(
    V: torch.Tensor,
    eps_dft: torch.Tensor,
    grid: EnergyGrid,
    orb_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sum LPDOS over a channel (e.g. all s or all p) at one k.

    Returns ``(E, rho_dft, rho_tb)`` as numpy arrays (not re-normalized).
    """
    rho_dft, rho_tb, _, _ = lpdos_dft_tb_at_k(V, eps_dft, grid)
    d = rho_dft[orb_indices].sum(dim=0).detach().cpu().numpy()
    t = rho_tb[orb_indices].sum(dim=0).detach().cpu().numpy()
    E = grid.E.detach().cpu().numpy()
    return E, d, t
