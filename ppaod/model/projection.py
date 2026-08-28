"""Build ``A(θ,k)``, Löwdin orthonormalization, and projectors ``P(k)``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from ..basis.bessel_basis import BesselBasis, make_bases, precompute_T_l_vectorized
from ..basis.spherical_harmonics import real_spherical_harmonic
from ..file_io.wannier90_files import TrialOrbital


@dataclass
class ProjectionCache:
    """θ-independent quantities for building ``A(θ,k)``."""

    # Per local k, per l: T_l[k] shape (n_basis[l], nG[k])
    T_l: list[dict[int, torch.Tensor]]
    # Per local k, per orbital n: Y_lm(q_hat) * phase * (4π i^l / √Ω) → complex (nG,)
    angular_phase: list[list[torch.Tensor]]
    # Wavefunction coeffs restricted to outer window: (Nb[k], nG[k])
    c_win: list[torch.Tensor]
    # Band indices into full DFT space for each *local* cache slot
    band_idx: list[NDArray[np.int64]]
    bases: dict[int, BesselBasis]
    orbitals: list[TrialOrbital]
    # map orbital index → l
    orb_l: list[int]
    omega: float  # cell volume Bohr^3
    J: int
    dtype: torch.dtype = torch.complex128
    # Global mesh indices for each local cache slot (None ⇒ identity 0..n-1)
    global_k_indices: list[int] | None = None
    # Full-mesh band_idx for MMN / Ω_D (length Nk_global); defaults to band_idx
    band_idx_global: list[NDArray[np.int64]] | None = None

    def __post_init__(self) -> None:
        if self.band_idx_global is None:
            self.band_idx_global = list(self.band_idx)

    def global_k_of(self, ik_local: int) -> int:
        if self.global_k_indices is None:
            return ik_local
        return int(self.global_k_indices[ik_local])


def cell_volume_bohr(real_lattice_ang: NDArray[np.float64]) -> float:
    A = real_lattice_ang / 0.52917720859
    return float(np.abs(np.linalg.det(A)))


def build_projection_cache(
    *,
    c_list: list[NDArray[np.complex128]],
    Gvecs_list: list[NDArray[np.float64]],
    k_cart_list: list[NDArray[np.float64]],
    orbitals: list[TrialOrbital],
    band_idx: list[NDArray[np.int64]],
    real_lattice_ang: NDArray[np.float64],
    r_c: float | dict[int, float],
    n_basis: int | dict[int, int],
    device: torch.device | None = None,
    global_k_indices: list[int] | None = None,
    band_idx_global: list[NDArray[np.int64]] | None = None,
) -> ProjectionCache:
    """
    Precompute ``T_l[k]``, ``Y_lm``, and ``exp(-i q·τ)`` for k-points in
    ``c_list``.

    For distributed runs, pass only local-k WFCs and set ``global_k_indices``
    (and full-mesh ``band_idx_global`` for MMN indexing).
    """
    device = device or torch.device("cpu")
    omega = cell_volume_bohr(real_lattice_ang)
    pref0 = 4.0 * np.pi / np.sqrt(omega)
    l_values = [orb.l for orb in orbitals]
    bases = make_bases(l_values, r_c=r_c, n_basis=n_basis)
    J = len(orbitals)
    Nk_loc = len(c_list)
    if global_k_indices is None:
        global_k_indices = list(range(Nk_loc))
    if len(global_k_indices) != Nk_loc:
        raise ValueError("global_k_indices length must match c_list")

    T_all: list[dict[int, torch.Tensor]] = []
    ang_all: list[list[torch.Tensor]] = []
    c_win: list[torch.Tensor] = []
    band_idx_local: list[NDArray[np.int64]] = []

    for i_loc in range(Nk_loc):
        gk = global_k_indices[i_loc]
        G = Gvecs_list[i_loc]
        k = k_cart_list[i_loc]
        q = G + k[None, :]
        q_abs = np.linalg.norm(q, axis=1)
        q_hat = np.zeros_like(q)
        mask = q_abs > 1e-14
        q_hat[mask] = q[mask] / q_abs[mask, None]

        T_k: dict[int, torch.Tensor] = {}
        for l, basis in bases.items():
            T_k[l] = precompute_T_l_vectorized(basis, q_abs).to(device=device)
        T_all.append(T_k)

        ang_k: list[torch.Tensor] = []
        for orb in orbitals:
            Y = real_spherical_harmonic(orb.l, orb.mr, q_hat)
            phase = np.exp(-1j * (q @ orb.tau_cart_bohr))
            il = ((-1j) ** orb.l)  # QE/Wannier90 plane-wave convention
            fac = pref0 * il * Y * phase
            ang_k.append(torch.as_tensor(fac, dtype=torch.complex128, device=device))
        ang_all.append(ang_k)

        # band_idx may be full-mesh or already local-aligned
        if len(band_idx) == Nk_loc and band_idx_global is not None:
            idx = band_idx[i_loc]
        elif band_idx_global is not None:
            idx = band_idx_global[gk]
        else:
            idx = band_idx[gk] if len(band_idx) > Nk_loc else band_idx[i_loc]
        band_idx_local.append(np.asarray(idx, dtype=np.int64))
        cw = torch.as_tensor(c_list[i_loc][idx], dtype=torch.complex128, device=device)
        c_win.append(cw)

    return ProjectionCache(
        T_l=T_all,
        angular_phase=ang_all,
        c_win=c_win,
        band_idx=band_idx_local,
        bases=bases,
        orbitals=orbitals,
        orb_l=[o.l for o in orbitals],
        omega=omega,
        J=J,
        global_k_indices=list(global_k_indices),
        band_idx_global=(
            list(band_idx_global) if band_idx_global is not None else list(band_idx)
        ),
    )


def radial_on_G(
    theta: dict[int, torch.Tensor],
    cache: ProjectionCache,
    ik: int,
) -> dict[int, torch.Tensor]:
    """``R̃_l(q_G) = Σ_i c_{i,l} T_l[k][i,G]`` for each ``l`` at k-point ``ik``."""
    out: dict[int, torch.Tensor] = {}
    for l, c_ml in theta.items():
        out[l] = torch.einsum("i,iG->G", c_ml.to(dtype=torch.float64), cache.T_l[ik][l])
    return out


def build_A(
    theta: dict[int, torch.Tensor],
    cache: ProjectionCache,
    ik: int,
) -> torch.Tensor:
    """
    Projection matrix ``A(θ,k)`` of shape ``(Nk[k], J)``::

        A[m,n] = Σ_G conj(c[m,G]) · φ̃_n(θ, k+G)
    """
    radial = radial_on_G(theta, cache, ik)
    nG = cache.c_win[ik].shape[1]
    J = cache.J
    device = cache.c_win[ik].device
    phi = torch.zeros((J, nG), dtype=torch.complex128, device=device)
    for n, l in enumerate(cache.orb_l):
        phi[n] = radial[l].to(dtype=torch.complex128) * cache.angular_phase[ik][n]
    # (Nk, nG) @ (nG, J)
    return cache.c_win[ik].conj() @ phi.transpose(0, 1)


def lowdin(
    A: torch.Tensor,
    *,
    eps: float = 1e-8,
    frozen: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Löwdin / polar orthonormalization ``V = A (A†A)^{-1/2}``.

    Implemented via thin SVD ``A = Z Σ W†`` → ``V = Z W†``, which is
    numerically stabler than an explicit eigen-inverse-sqrt near rank
    deficiency. Columns with singular value ``< eps`` are dropped from the
    inverse and replaced by a completion of the Stiefel frame (rare).
    """
    if frozen is None:
        Z, s, Vh = torch.linalg.svd(A, full_matrices=False)
        # A (A†A)^{-1/2} = Z Vh  (independent of Σ, as long as Σ > 0)
        # Keep all columns; if some s≈0 the corresponding directions in Z
        # are still orthonormal and form a valid subspace basis for the range.
        keep = s > eps
        if bool(keep.all()):
            return Z @ Vh
        # Rank-deficient: orthonormalize range(Z[:, keep]) then pad if needed
        Zk = Z[:, keep]
        J = A.shape[1]
        if Zk.shape[1] == J:
            # Still J columns despite tiny s — use QR on A
            Q, _ = torch.linalg.qr(A, mode="reduced")
            return Q
        Q, _ = torch.linalg.qr(A, mode="reduced")
        return Q

    Nf = frozen.shape[1]
    A_free = A[:, Nf:]
    A_free = A_free - frozen @ (frozen.conj().T @ A_free)
    Q, _ = torch.linalg.qr(A_free, mode="reduced")
    return torch.cat([frozen, Q], dim=1)


def projector(V: torch.Tensor) -> torch.Tensor:
    """``P = V V†``."""
    return V @ V.conj().T


def check_isometry(V: torch.Tensor, tol: float = 1e-8) -> float:
    """Return ``||V†V - I||_F``."""
    J = V.shape[1]
    I = torch.eye(J, dtype=V.dtype, device=V.device)
    return float(torch.linalg.norm((V.conj().T @ V - I).detach()).real)


def max_condition_number(
    theta: dict[int, torch.Tensor],
    cache: ProjectionCache,
) -> float:
    """Max over k of ``cond(A†A)``."""
    worst = 1.0
    for ik in range(len(cache.c_win)):
        A = build_A(theta, cache, ik)
        S = A.conj().T @ A
        w = torch.linalg.eigvalsh(0.5 * (S + S.conj().T)).real
        cond = float((w.max() / w.clamp_min(1e-30).min()).item())
        worst = max(worst, cond)
    return worst
