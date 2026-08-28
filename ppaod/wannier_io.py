"""Wannier90 ``.amn`` / ``.eig`` / ``.nnkp`` I/O and QE→TB orbital remapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray


def read_amn(path: str | Path) -> tuple[NDArray[np.complex128], int, int, int]:
    """
    Parse Wannier90 ``seedname.amn``.

    Returns
    -------
    A : (nk, nbands, norb) complex128
        ``A[k, n, m] = ⟨ψ_nk | φ_m⟩`` (QE ``pw2wannier90`` convention).
    num_bands, num_kpts, num_wann : int
    """
    path = Path(path)
    with path.open(encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        header = f.readline().split()
        num_bands, num_kpts, num_wann = map(int, header[:3])
        A = np.zeros((num_kpts, num_bands, num_wann), dtype=np.complex128)
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            ib = int(parts[0]) - 1
            iw = int(parts[1]) - 1
            ik = int(parts[2]) - 1
            A[ik, ib, iw] = complex(float(parts[3]), float(parts[4]))
    return A, num_bands, num_kpts, num_wann


def read_eigenvalues(path: str | Path) -> NDArray[np.float64]:
    """Parse Wannier90 ``seedname.eig`` → ``(nk, nbands)`` in eV."""
    path = Path(path)
    rows: list[tuple[int, int, float]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append((int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])))
    if not rows:
        raise ValueError(f"No eigenvalues in {path}")
    nbnd = max(r[0] for r in rows) + 1
    nk = max(r[1] for r in rows) + 1
    eig = np.zeros((nk, nbnd), dtype=np.float64)
    for ib, ik, e in rows:
        eig[ik, ib] = e
    return eig


def read_kpoints_nnkp(path: str | Path) -> NDArray[np.float64]:
    """Crystal-fractional k-points from ``seedname.nnkp`` → ``(nk, 3)``."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip().lower().startswith("begin kpoints"):
        i += 1
    if i >= len(lines):
        raise ValueError(f"No 'begin kpoints' block in {path}")
    i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    nk = int(lines[i].split()[0])
    i += 1
    kpts = np.zeros((nk, 3), dtype=np.float64)
    for ik in range(nk):
        while i < len(lines) and not lines[i].strip():
            i += 1
        parts = lines[i].split()
        kpts[ik] = [float(parts[0]), float(parts[1]), float(parts[2])]
        i += 1
    return kpts


def amn_phi_psi_tb_order(
    A_amn: NDArray[np.complex128] | torch.Tensor,
) -> NDArray[np.complex128] | torch.Tensor:
    """
    Convert AMN to ``⟨φ_m|ψ_n⟩`` in TB orbital order.

    QE ``atom_proj`` / PSWFC order per atom is ``s, pz, px, py``.
    TB model order per atom is ``s, px, py, pz``.
    """
    is_torch = isinstance(A_amn, torch.Tensor)
    A = A_amn
    # ⟨φ|ψ⟩ = ⟨ψ|φ⟩*
    A_phi = A.conj() if not is_torch else torch.conj(A)
    norb = A_phi.shape[-1]
    if norb % 4 != 0:
        raise ValueError(f"Expected 4 orbitals/atom, got norb={norb}")
    n_atoms = norb // 4
    # QE indices → TB: s,px,py,pz from s,pz,px,py
    qe_to_tb = [0, 2, 3, 1]
    if is_torch:
        out = torch.empty_like(A_phi)
    else:
        out = np.empty_like(A_phi)
    for a in range(n_atoms):
        base = 4 * a
        for t, q in enumerate(qe_to_tb):
            out[..., base + t] = A_phi[..., base + q]
    return out


def align_eigenvector_phase(
    c: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Multiply ``c`` by a global phase so ``⟨reference|c⟩`` is real and ≥ 0.

    The phase is computed from detached tensors so the matching gauge is a
    constant for autograd (see ``phase_alignment_choice.md``).
    """
    ov = (reference.detach().conj() * c.detach()).sum()
    if abs(complex(ov)) < 1e-14:
        return c
    phase = torch.exp(-1j * torch.angle(ov))
    return c * phase


def amn_psi_phi_tb_order(A_amn: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """``⟨ψ_nk|φ_m⟩`` in TB orbital order ``s,px,py,pz`` (from QE ``s,pz,px,py``)."""
    return np.asarray(amn_phi_psi_tb_order(A_amn).conj())


def lowdin_orthogonalize_batch(
    H_k: NDArray[np.complex128],
    S_k: NDArray[np.complex128],
    *,
    eig_floor: float = 1e-8,
) -> NDArray[np.complex128]:
    """
    Löwdin orthogonalization at each k::

        H_orth = S^{-1/2} H S^{-1/2}

    ``S`` eigenvalues below ``eig_floor`` are clipped before taking the inverse
    square root (needed when the projected overlap is nearly singular).
    """
    nk, norb, _ = H_k.shape
    H_orth = np.empty_like(H_k)
    for ik in range(nk):
        S = 0.5 * (S_k[ik] + S_k[ik].conj().T)
        H = 0.5 * (H_k[ik] + H_k[ik].conj().T)
        evals, evecs = np.linalg.eigh(S)
        evals = np.maximum(evals.real, eig_floor)
        s_inv_sqrt = (evecs * (evals ** (-0.5))) @ evecs.conj().T
        H_orth[ik] = s_inv_sqrt @ H @ s_inv_sqrt
        H_orth[ik] = 0.5 * (H_orth[ik] + H_orth[ik].conj().T)
    return H_orth


def band_projectability(A: NDArray[np.complex128]) -> NDArray[np.float64]:
    """
    ``P_nk = Σ_m |A_{nmk}|²`` with ``A[k,n,m] = ⟨ψ_nk|φ_m⟩`` (or same moduli
    for ``⟨φ|ψ⟩``).  Shape ``(nk, nbands)``.
    """
    return np.sum(np.abs(A) ** 2, axis=-1)


def projectability_threshold_for_n_bands(
    P: NDArray[np.float64],
    n_keep: int,
) -> float:
    """
    Largest projectability threshold ``T`` such that **every** k-point retains
    at least ``n_keep`` DFT bands with ``P_nk ≥ T``::

        T = min_k [ n_keep-th largest P_nk ]

    Typically ``n_keep = num_wann`` so the AMN projection always includes at
    least as many DFT bands as AO projectors.
    """
    if n_keep < 1:
        raise ValueError(f"n_keep must be >= 1, got {n_keep}")
    if P.ndim != 2:
        raise ValueError(f"P must be (nk, nbands), got shape {P.shape}")
    nk, nbnd = P.shape
    if n_keep > nbnd:
        raise ValueError(
            f"n_keep={n_keep} exceeds number of DFT bands nbnd={nbnd}"
        )
    # ascending sort → index -n_keep is the n_keep-th largest
    p_sorted = np.sort(P, axis=-1)
    return float(np.min(p_sorted[:, -n_keep]))


def projectability_band_mask(
    P: NDArray[np.float64],
    n_keep: int,
) -> tuple[NDArray[np.bool_], float]:
    """
    Boolean mask ``P >= T`` with ``T = projectability_threshold_for_n_bands``.

    Returns
    -------
    mask : (nk, nbands) bool
    threshold : float
    """
    thr = projectability_threshold_for_n_bands(P, n_keep)
    return P >= thr, thr


def build_Hk(
    A: NDArray[np.complex128],
    eig: NDArray[np.float64],
    *,
    band_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.complex128]:
    """
    Projected Hamiltonian at every k::

        H_ij(k) = Σ_n ⟨φ_i|ψ_nk⟩ ε_nk ⟨ψ_nk|φ_j⟩

    With ``A[k,n,m] = ⟨ψ_nk|φ_m⟩`` this is ``H(k) = A† diag(ε) A``.
    If ``band_mask`` is given ``(nk, nbands)``, only bands with ``True`` enter
    the sum (low-projectability bands dropped).

    Returns
    -------
    H_k : (nk, norb, norb) complex128
    """
    if band_mask is None:
        weighted = A * eig[:, :, None]
        return np.einsum("kni,knj->kij", A.conj(), weighted)
    m = band_mask.astype(np.float64)[:, :, None]
    weighted = A * eig[:, :, None] * m
    return np.einsum("kni,knj->kij", A.conj() * m, weighted)


def build_Sk(
    A: NDArray[np.complex128],
    *,
    band_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.complex128]:
    """
    Projected overlap ``S(k) = A† A`` with ``A[k,n,m] = ⟨ψ_nk|φ_m⟩``.
    Optional ``band_mask`` restricts the band sum as in ``build_Hk``.

    Returns
    -------
    S_k : (nk, norb, norb) complex128
    """
    if band_mask is None:
        return np.einsum("kni,knj->kij", A.conj(), A)
    m = band_mask.astype(np.float64)[:, :, None]
    return np.einsum("kni,knj->kij", A.conj() * m, A * m)


def fourier_transform(
    M_k: NDArray[np.complex128],
    kpts_frac: NDArray[np.float64],
    R_frac: NDArray[np.int64] | NDArray[np.float64],
) -> NDArray[np.complex128]:
    """
    Lattice-gauge Fourier transform (crystal-fractional k and R)::

        M(R) = (1/N_k) Σ_k exp(-2 π i k · R) M(k)

    Parameters
    ----------
    M_k : (nk, n, n)
    kpts_frac : (nk, 3)
    R_frac : (nR, 3)

    Returns
    -------
    M_R : (nR, n, n) complex128
    """
    phase = np.exp(-2j * np.pi * (np.asarray(R_frac, dtype=np.float64) @ kpts_frac.T))
    return np.einsum("rk,kij->rij", phase, M_k) / M_k.shape[0]


def inverse_fourier_transform(
    M_R: NDArray[np.complex128],
    kpts_frac: NDArray[np.float64],
    R_frac: NDArray[np.int64] | NDArray[np.float64],
) -> NDArray[np.complex128]:
    """
    Bloch sum (inverse of ``fourier_transform``)::

        M(k) = Σ_R exp(+2 π i k · R) M(R)

    ``kpts_frac`` may be any set of crystal-fractional k (e.g. a band path).
    """
    phase = np.exp(+2j * np.pi * (kpts_frac @ np.asarray(R_frac, dtype=np.float64).T))
    return np.einsum("kr,rij->kij", phase, M_R)


def mp_R_vectors(n1: int, n2: int, n3: int) -> NDArray[np.int64]:
    """
    Full FFT dual lattice for an ``mp_grid = n1 n2 n3`` Monkhorst–Pack mesh:
    ``R = (i,j,k)`` with ``i∈[0,n1)``, etc.  Shape ``(n1*n2*n3, 3)``.
    """
    Rs = [
        (i, j, k)
        for i in range(n1)
        for j in range(n2)
        for k in range(n3)
    ]
    return np.asarray(Rs, dtype=np.int64)


def read_mp_grid_win(path: str | Path) -> tuple[int, int, int]:
    """Parse ``mp_grid = N1 N2 N3`` from a Wannier90 ``.win`` file."""
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip().lower()
        if s.startswith("mp_grid"):
            # mp_grid = 19 19 1   or   mp_grid : 19 19 1
            rhs = s.split("=", 1)[-1] if "=" in s else s.split(":", 1)[-1]
            parts = rhs.replace(",", " ").split()
            return int(parts[0]), int(parts[1]), int(parts[2])
    raise ValueError(f"No mp_grid in {path}")


def infer_mp_grid_from_kpoints(kpts_frac: NDArray[np.float64]) -> tuple[int, int, int]:
    """
    Infer Monkhorst–Pack dimensions from a regular fractional k-mesh.
    Falls back to unique counts along each crystal axis.
    """
    dims = []
    for ax in range(3):
        vals = np.unique(np.round(kpts_frac[:, ax], decimals=10))
        dims.append(max(int(vals.size), 1))
    return int(dims[0]), int(dims[1]), int(dims[2])
