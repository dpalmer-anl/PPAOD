"""
Minimal PyTorch implementation of band disentanglement
(Souza, Marzari, Vanderbilt, PRB 65, 035109 (2001)).

Notation (matches the derivation):
    M            number of DFT bands in the outer window (m, n index these)
    N            number of target Wannier functions (i, j index these), N <= M
    Nk           number of k-points on the mesh
    A[k]         (M, N) complex tensor: A_mi(k) = <psi_mk | g_i>  (trial-orbital projections)
    mmn[(k,b)]   (M, M) complex tensor: M_mn^(k,b) = <u_mk | u_{n,k+b}>  (DFT-band overlaps)
    neighbors[k] list of (b_idx, k_plus_b_idx) pairs -- the b-shell geometry (from .nnkp)
    weights      (Nb,) real tensor of finite-difference weights w_b

Output:
    U[k]  (M, N) complex, isometry (U^dagger U = I_N) defining the optimal
          N-dim disentangled subspace S(k) subset F(k) at each k.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
QE_DIR = HERE / "qe_inner_window/"
SEEDNAME = "carbon_wannier"
BAND_PATH_POINTS = 30
BAND_SYMMETRY_POINTS = (
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (2.0 / 3.0, 1.0 / 3.0, 0.0),
    (0.0, 0.0, 0.0),
)

from .band_io import parse_qe_bands_out
from .wannier_io import (
    build_Hk,
    build_Sk,
    fourier_transform,
    infer_mp_grid_from_kpoints,
    inverse_fourier_transform,
    read_amn,
    read_eigenvalues,
    read_kpoints_nnkp,
    read_mp_grid_win,
)
from scipy.io import FortranFile  # noqa: E402



class Disentangler:
    def __init__(self, amn, mmn, neighbors, weights, num_wann,
                 num_iter=500, mix=0.5, tol=1e-10, verbose=True, lr=0.05):
        self.amn = amn                # list[Nk] of (M, N) complex
        self.mmn = mmn                # dict {(k_idx, b_idx): (M, M) complex}
        self.neighbors = neighbors    # list[Nk] of (b_idx, k_plus_b_idx)
        self.weights = weights        # (Nb,) real
        self.N = num_wann
        self.Nk = len(amn)
        self.num_iter = num_iter
        self.mix = mix                # unused by run() (kept for backward compat)
        self.tol = tol
        self.verbose = verbose
        self.lr = lr                  # Adam learning rate for run()
        self._cdtype = amn[0].dtype

    # ---- Step 1: initial guess via projection + Lowdin orthonormalization ----
    # A = Z D V (SVD)  =>  A S^{-1/2} = Z * 1_{MxN} * V   (paper, Sec. III D)
    def lowdin_guess(self):
        U0 = []
        for A in self.amn:
            Z, _, Vh = torch.linalg.svd(A, full_matrices=False)  # Z:(M,N), Vh:(N,N)
            U0.append(Z @ Vh)
        return U0

    @staticmethod
    def projector(U):
        # P_k = U(k) U(k)^dagger, an (M, M) Hermitian idempotent matrix
        return U @ U.conj().T

    # ---- Step 2: build the Z-matrix (Eq. 21) and diagonalize (Eq. 17/19) ----
    def build_Z_matrix(self, k_idx, P_list):
        M = self.amn[k_idx].shape[0]
        Zk = torch.zeros(M, M, dtype=self._cdtype)
        for b_idx, kb_idx in self.neighbors[k_idx]:
            Mkb = self.mmn[(k_idx, b_idx)]
            wb = self.weights[b_idx]
            # Z^(k) += w_b M^(k,b) P^(k+b) M^(k,b)^dagger
            Zk = Zk + wb * (Mkb @ P_list[kb_idx] @ Mkb.conj().T)
        return Zk

    def update_subspace(self, P_list):
        U_new = []
        for k_idx in range(self.Nk):
            Zk = self.build_Z_matrix(k_idx, P_list)
            Zk = 0.5 * (Zk + Zk.conj().T)          # enforce exact Hermiticity
            evals, evecs = torch.linalg.eigh(Zk)   # ascending eigenvalues
            U_new.append(evecs[:, -self.N:])       # keep N largest-eigenvalue vectors
        return U_new

    # ---- Gauge-invariant spread Omega_I (Eq. 7), evaluated on the current subspace ----
    def omega_I_loss(self, U_list) -> torch.Tensor:
        """Differentiable Omega_I (real scalar tensor, keeps the autograd graph)."""
        total = torch.zeros((), dtype=torch.float64)
        for k_idx in range(self.Nk):
            for b_idx, kb_idx in self.neighbors[k_idx]:
                Mkb = self.mmn[(k_idx, b_idx)]
                Mtilde = U_list[k_idx].conj().T @ Mkb @ U_list[kb_idx]   # (N, N)
                wb = self.weights[b_idx]
                total = total + wb * (self.N - (Mtilde.abs() ** 2).sum())
        return total / self.Nk

    def omega_I(self, U_list) -> float:
        return float(self.omega_I_loss(U_list).detach())

    @staticmethod
    def _retract(W: torch.Tensor) -> torch.Tensor:
        """Project (M, N) onto the Stiefel manifold (U^dagger U = I_N) via QR."""
        Q, _ = torch.linalg.qr(W, mode="reduced")
        return Q

    @staticmethod
    def align_to_reference(U: torch.Tensor, U_ref: torch.Tensor) -> torch.Tensor:
        """
        Right-multiply ``U`` by a unitary ``V`` that maximizes
        ``Re Tr(U_ref^dagger U V)`` (polar decomposition of ``U^dagger U_ref``).

        This is a pure within-subspace gauge change: the projector
        ``P = U U^dagger`` and the band eigenvalues of ``U^dagger eps U`` are
        unchanged, but neighboring ``H(k)`` matrices become continuous so an
        ``H(R)`` Fourier transform is well-defined.
        """
        M = U.conj().T @ U_ref
        W, _, Vh = torch.linalg.svd(M, full_matrices=False)
        return U @ (W @ Vh)

    def smooth_gauge(
        self,
        U_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Make the within-subspace gauge of ``U(k)`` continuous across the mesh.

        Adam/SMV minimize the gauge-invariant Omega_I, so the raw U(k) have
        essentially random relative gauges. Mesh eigenvalues and Omega_I stay
        correct, but ``H(k) = U^{+} ε U`` then jumps discontinuously between
        neighboring k-points and the ``H(R)`` Fourier transform produces wildly
        oscillating bands. Wannier90 inherits a smoother gauge from its
        iterative eigenvector updates seeded by the same projections.

        Fix: at each k independently, right-multiply ``U(k)`` by the unitary
        that maximizes ``Re Tr(U0(k)† U(k) V)``, where ``U0(k)`` is the
        continuous Löwdin projection seed ``A(k) S(k)^{-1/2}``. This is the
        polar factor of ``U(k)† U0(k)`` and pins the disentangled frame to the
        same continuous gauge Wannier90 starts from. Neighbor-refinement and
        Γ-rooted parallel transport are deliberately avoided -- both were
        found to re-introduce branch cuts on this mesh.
        """
        U0 = self.lowdin_guess()
        return [
            self.align_to_reference(U_list[ik].detach(), U0[ik].detach())
            for ik in range(self.Nk)
        ]

    # ---- Step 3: direct Adam minimization of Omega_I over the Stiefel manifold ----
    def run(self):
        """
        Minimize Omega_I(U) directly with PyTorch's Adam optimizer, instead of
        the SMV fixed-point/mixing iteration (``build_Z_matrix`` +
        ``update_subspace``). Each U(k) is parametrized by an unconstrained
        (M, N) complex matrix W(k) and retracted onto the Stiefel manifold
        (U(k)^dagger U(k) = I_N) via a QR decomposition at every step, so
        Omega_I stays differentiable end-to-end and Adam converges in far
        fewer iterations than the alpha-mixed SCF loop.

        After convergence the returned ``U(k)`` are gauge-smoothed across the
        mesh (see ``smooth_gauge``) so that ``H(k)=U^dagger eps U`` can be
        Fourier transformed to a short-ranged ``H(R)``.
        """
        W_list = [u.clone().detach().requires_grad_(True) for u in self.lowdin_guess()]
        opt = torch.optim.Adam(W_list, lr=self.lr)

        prev_omega = None
        U = [self._retract(W.detach()) for W in W_list]
        if self.verbose:
            print(f"iter   0: Omega_I = {self.omega_I(U):.8f}")

        for it in range(1, self.num_iter + 1):
            opt.zero_grad()
            U = [self._retract(W) for W in W_list]
            loss = self.omega_I_loss(U)
            loss.backward()
            opt.step()
            omega = float(loss.detach())

            if self.verbose and (it % 10 == 0 or it == 1):
                print(f"iter {it:4d}: Omega_I = {omega:.8f}")

            if prev_omega is not None and abs(omega - prev_omega) < self.tol:
                if self.verbose:
                    print(f"Converged at iter {it}, Omega_I = {omega:.8f}")
                break
            prev_omega = omega

        with torch.no_grad():
            U = [self._retract(W) for W in W_list]
            U = self.smooth_gauge(U)
            if self.verbose:
                print(f"after gauge smooth: Omega_I = {self.omega_I(U):.8f}")
        return U   # list[Nk] of (M, N) isometries defining S(k)


# --------------------------------------------------------------------------- #
# Wannier90 I/O (local to this test; reuses read_amn from wannier_io)
# --------------------------------------------------------------------------- #


def read_mmn(path: str | Path) -> tuple[dict[tuple[int, int], torch.Tensor], int, int, int]:
    """
    Parse Wannier90 ``seedname.mmn``.

    Returns
    -------
    mmn : dict {(ik, ib): (nbands, nbands) complex128}
        Overlap ``M_mn^(k,b) = <u_mk | u_n,k+b>``. ``ib`` is 0..nntot-1 in
        the same order as the ``nnkpts`` block.
    num_bands, num_kpts, nntot : int
    """
    path = Path(path)
    with path.open(encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        header = f.readline().split()
        num_bands, num_kpts, nntot = map(int, header[:3])
        mmn: dict[tuple[int, int], torch.Tensor] = {}
        for ik in range(num_kpts):
            for ib in range(nntot):
                parts = f.readline().split()
                if len(parts) < 5:
                    raise ValueError(f"Bad MMN block header at k={ik}, b={ib} in {path}")
                # ik+1, ikp+1, G1 G2 G3 — order matches nnkpts
                mat = np.empty((num_bands, num_bands), dtype=np.complex128)
                for n in range(num_bands):
                    for m in range(num_bands):
                        re, im = map(float, f.readline().split()[:2])
                        mat[m, n] = complex(re, im)
                mmn[(ik, ib)] = torch.from_numpy(mat)
    return mmn, num_bands, num_kpts, nntot


def _parse_nnkp_block(lines: list[str], tag: str) -> list[str]:
    begin = f"begin {tag}"
    end = f"end {tag}"
    i = 0
    while i < len(lines) and lines[i].strip().lower() != begin:
        i += 1
    if i >= len(lines):
        raise ValueError(f"No '{begin}' block")
    i += 1
    block: list[str] = []
    while i < len(lines) and lines[i].strip().lower() != end:
        if lines[i].strip():
            block.append(lines[i])
        i += 1
    return block


def read_nnkp_neighbors(
    path: str | Path,
) -> tuple[list[list[tuple[int, int]]], torch.Tensor]:
    """
    Parse ``nnkpts`` + lattice from ``seedname.nnkp``.

    Returns
    -------
    neighbors : list[Nk] of (b_idx, k_plus_b_idx)  (0-based)
    weights : (nntot,) float64 finite-difference weights w_b (Ang^2),
        solved from Σ_b w_b b_α b_β = δ_αβ with equal |b| sharing a weight.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    recip_lines = _parse_nnkp_block(lines, "recip_lattice")
    recip = np.array(
        [[float(x) for x in row.split()[:3]] for row in recip_lines[:3]],
        dtype=np.float64,
    )

    k_lines = _parse_nnkp_block(lines, "kpoints")
    nk = int(k_lines[0].split()[0])
    kpts = np.array(
        [[float(x) for x in row.split()[:3]] for row in k_lines[1 : 1 + nk]],
        dtype=np.float64,
    )

    nn_lines = _parse_nnkp_block(lines, "nnkpts")
    nntot = int(nn_lines[0].split()[0])
    neighbors: list[list[tuple[int, int]]] = [[] for _ in range(nk)]
    bvecs = np.zeros((nntot, 3), dtype=np.float64)
    row = 1
    for ik in range(nk):
        for ib in range(nntot):
            parts = nn_lines[row].split()
            row += 1
            k_from = int(parts[0]) - 1
            k_to = int(parts[1]) - 1
            G = np.array([int(parts[2]), int(parts[3]), int(parts[4])], dtype=np.float64)
            if k_from != ik:
                raise ValueError(f"nnkpts order mismatch at k={ik}, b={ib}")
            neighbors[ik].append((ib, k_to))
            if ik == 0:
                # b = (k' + G - k) in cartesian reciprocal (Ang^-1)
                dk = kpts[k_to] + G - kpts[ik]
                bvecs[ib] = dk @ recip

    weights = _bvector_weights(bvecs)
    return neighbors, torch.from_numpy(weights)


def _bvector_weights(bvecs: np.ndarray, tol: float = 1e-5) -> np.ndarray:
    """Shell-constrained least-squares solution of Σ_b w_b b_α b_β = δ_αβ."""
    Nb = bvecs.shape[0]
    norms = np.linalg.norm(bvecs, axis=1)
    shells: list[float] = []
    shell_of = np.empty(Nb, dtype=np.int64)
    for i, n in enumerate(norms):
        for s, sn in enumerate(shells):
            if abs(n - sn) < tol:
                shell_of[i] = s
                break
        else:
            shell_of[i] = len(shells)
            shells.append(float(n))
    nshell = len(shells)
    eqs = []
    rhs = []
    for a in range(3):
        for b in range(a, 3):
            row = np.zeros(nshell, dtype=np.float64)
            for i in range(Nb):
                row[shell_of[i]] += bvecs[i, a] * bvecs[i, b]
            eqs.append(row)
            rhs.append(1.0 if a == b else 0.0)
    w_shell, *_ = np.linalg.lstsq(np.asarray(eqs), np.asarray(rhs), rcond=None)
    return w_shell[shell_of]


def load_disentangle_data(
    qe_dir: Path | str = QE_DIR,
    seed: str = SEEDNAME,
) -> tuple[
    list[torch.Tensor],
    dict[tuple[int, int], torch.Tensor],
    list[list[tuple[int, int]]],
    torch.Tensor,
    int,
    np.ndarray,
    np.ndarray,
]:
    """Load A(k), M(k,b), neighbors, w_b, ε(k), kpts from a finished job."""
    qe_dir = Path(qe_dir)
    amn_path = qe_dir / f"{seed}.amn"
    mmn_path = qe_dir / f"{seed}.mmn"
    nnkp_path = qe_dir / f"{seed}.nnkp"
    eig_path = qe_dir / f"{seed}.eig"
    for p in (amn_path, mmn_path, nnkp_path, eig_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    A, num_bands, num_kpts, num_wann = read_amn(amn_path)
    mmn, nb_m, nk_m, nntot = read_mmn(mmn_path)
    neighbors, weights = read_nnkp_neighbors(nnkp_path)
    eig = read_eigenvalues(eig_path)
    kpts = read_kpoints_nnkp(nnkp_path)

    if (nb_m, nk_m) != (num_bands, num_kpts):
        raise ValueError(
            f"AMN/MMN size mismatch: amn=({num_bands},{num_kpts}) "
            f"mmn=({nb_m},{nk_m})"
        )
    if eig.shape != (num_kpts, num_bands):
        raise ValueError(f"eig shape {eig.shape} != ({num_kpts}, {num_bands})")
    if len(neighbors) != num_kpts or kpts.shape[0] != num_kpts:
        raise ValueError(f"nnkp Nk mismatch with amn Nk={num_kpts}")
    if len(weights) != nntot:
        raise ValueError(f"nntot mismatch: mmn={nntot}, weights={len(weights)}")

    amn_list = [torch.from_numpy(A[ik]) for ik in range(num_kpts)]
    return amn_list, mmn, neighbors, weights, num_wann, eig, kpts


def build_disentangled_Hk(
    U_list: list[torch.Tensor],
    eig: np.ndarray,
) -> np.ndarray:
    """
    Subspace Hamiltonian on the Wannier mesh::

        H_{ij}(k) = sum_m U^*_{m i}(k) ε_m(k) U_{m j}(k)

    Returns ``(Nk, N, N)`` complex Hermitian matrices.
    """
    Nk = len(U_list)
    N = U_list[0].shape[1]
    H_k = np.empty((Nk, N, N), dtype=np.complex128)
    for ik, U in enumerate(U_list):
        eps = torch.as_tensor(eig[ik], dtype=U.real.dtype, device=U.device)
        Hk = U.conj().T @ (eps[:, None] * U)
        Hk = 0.5 * (Hk + Hk.conj().T)
        H_k[ik] = Hk.detach().cpu().numpy()
    return H_k


def build_Hk_from_U(
    U: np.ndarray,
    eig: np.ndarray,
) -> np.ndarray:
    """
    ``H(k) = U(k)† diag(ε(k)) U(k)`` for ``U`` shaped ``(nk, nbands, nwann)``.
    """
    nk, nb, nw = U.shape
    if eig.shape != (nk, nb):
        raise ValueError(f"eig shape {eig.shape} != U bands {(nk, nb)}")
    H_k = np.empty((nk, nw, nw), dtype=np.complex128)
    for ik in range(nk):
        Hk = U[ik].conj().T @ (eig[ik][:, None] * U[ik])
        H_k[ik] = 0.5 * (Hk + Hk.conj().T)
    return H_k


def _chk_read_str(f: FortranFile) -> str:
    return f.read_record("c").tobytes().decode("ascii", errors="replace").strip()


def _chk_read_complex(f: FortranFile) -> np.ndarray:
    a = np.asarray(f.read_record("f8"), dtype=np.float64)
    return a[0::2] + 1j * a[1::2]


def read_chk_u_dis(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """
    Read Wannier90 ``seedname.chk`` and return the disentanglement isometry.

    Returns
    -------
    U_dis : (nk, num_bands, num_wann) complex128
        Full-band-space ``U_matrix_opt`` (``U_dis``), with rows outside the
        outer window set to zero.  Does **not** fold in the MLWF unitary
        ``U_matrix``.
    kpts : (nk, 3) crystal-fractional k-mesh from the checkpoint
    mp_grid : (n1, n2, n3)
    """
    path = Path(path)
    with FortranFile(path, "r") as f:
        _ = _chk_read_str(f)  # header comment
        num_bands = int(f.read_record("i4")[0])
        num_exclude = int(f.read_record("i4")[0])
        # Empty implied-DO still writes a record when num_exclude == 0.
        _ = f.read_record("i4")
        if num_exclude < 0:
            raise ValueError(f"Bad num_exclude_bands={num_exclude} in {path}")
        _ = f.read_record("f8").reshape((3, 3), order="F")  # real_lattice
        _ = f.read_record("f8").reshape((3, 3), order="F")  # recip_lattice
        num_kpts = int(f.read_record("i4")[0])
        mp_grid = tuple(int(x) for x in f.read_record("i4")[:3])
        kpts = np.asarray(f.read_record("f8"), dtype=np.float64).reshape((num_kpts, 3))
        _nntot = int(f.read_record("i4")[0])
        num_wann = int(f.read_record("i4")[0])
        _ = _chk_read_str(f)  # checkpoint tag
        have_dis = bool(int(f.read_record("i4")[0]))
        if not have_dis:
            raise ValueError(f"{path} has have_disentangled = F; no U_dis")
        _omega_i = float(f.read_record("f8")[0])
        lwindow = np.asarray(f.read_record("i4"), dtype=np.int32).reshape(
            (num_kpts, num_bands)
        ).astype(bool)
        ndimwin = np.asarray(f.read_record("i4"), dtype=np.int32)
        # Written ((u(i,j,k), i=bands), j=wann), k) → reshape (k, wann, bands)
        u_opt = _chk_read_complex(f).reshape((num_kpts, num_wann, num_bands)).swapaxes(
            1, 2
        )

    U_dis = np.zeros((num_kpts, num_bands, num_wann), dtype=np.complex128)
    for ik in range(num_kpts):
        idx = np.flatnonzero(lwindow[ik])
        nd = int(ndimwin[ik])
        if idx.size != nd:
            raise ValueError(
                f"lwindow/ndimwin mismatch at k={ik}: "
                f"|win|={idx.size}, ndimwin={nd}"
            )
        U_dis[ik, idx, :] = u_opt[ik, :nd, :]
    return U_dis, kpts, mp_grid


def build_projected_Hk_Sk(
    A_amn: np.ndarray,
    eig: np.ndarray,
    *,
    n_atoms: int,
    n_orb_per_atom: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    AMN-projected Hamiltonian and overlap on the Wannier mesh (all DFT bands)::

        H_nn'(k) = Σ_m A^*_{m n}(k) ε_m(k) A_{m n'}(k)
        S_nn'(k) = Σ_m A^*_{m n}(k) A_{m n'}(k)

    ``A`` is reordered to TB ``s,px,py,pz`` within each sp³ shell (supports
    4 or 8 orbitals/atom). Equivalent to ``H = A† diag(ε) A``, ``S = A† A``.
    """
    A = amn_psi_phi_tb_order_flexible(
        A_amn, n_atoms=n_atoms, n_orb_per_atom=n_orb_per_atom
    )
    H_k = build_Hk(A, eig, band_mask=None)
    S_k = build_Sk(A, band_mask=None)
    return H_k, S_k


def centered_mp_R_vectors(n1: int, n2: int, n3: int) -> np.ndarray:
    """Centered FFT dual of an ``mp_grid`` mesh: R_α ∈ [-n_α//2, n_α//2)."""

    def axis(n: int) -> range:
        return range(-(n // 2), n - (n // 2))

    return np.asarray(
        [(i, j, k) for i in axis(n1) for j in axis(n2) for k in axis(n3)],
        dtype=np.int64,
    )


def parse_bands_out_kpath(path: Path | str) -> np.ndarray:
    """
    Crystal-fractional k-path from the header of a QE ``bands.out``.

    QE prints cart. coords in units of ``2π/alat`` together with the reciprocal
    axes ``b(i)`` in the same units.  Crystal coords follow ``k_cart = B @ k_frac``
    with columns of ``B`` equal to ``b(1), b(2), b(3)``.
    """
    import re

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # Stop before band eigenvalues so we only see the geometry header.
    head = text.split("End of band structure calculation", 1)[0]

    bvecs: list[list[float]] = []
    for m in re.finditer(
        r"b\(\s*[123]\s*\)\s*=\s*\(\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\)",
        head,
    ):
        bvecs.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    if len(bvecs) != 3:
        raise ValueError(f"Could not parse 3 reciprocal axes from {path}")
    # Columns of B are b1,b2,b3
    B = np.array(bvecs, dtype=np.float64).T
    B_inv = np.linalg.inv(B)

    nk_m = re.search(r"number of k points\s*=\s*(\d+)", head)
    if nk_m is None:
        raise ValueError(f"Could not find number of k points in {path}")
    nk = int(nk_m.group(1))

    k_cart = []
    for m in re.finditer(
        r"k\(\s*\d+\s*\)\s*=\s*\(\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\)",
        head,
    ):
        k_cart.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        if len(k_cart) >= nk:
            break
    if len(k_cart) != nk:
        raise ValueError(
            f"Expected {nk} k-points in {path} header, found {len(k_cart)}"
        )

    k_frac = (B_inv @ np.asarray(k_cart, dtype=np.float64).T).T
    return k_frac


def bands_from_HR(
    H_R: np.ndarray,
    R_frac: np.ndarray,
    kpts_frac: np.ndarray,
    S_R: np.ndarray | None = None,
) -> np.ndarray:
    """
    Diagonalize Bloch-summed H(k) along ``kpts_frac``.

    If ``S_R`` is given (AMN-projected non-orthogonal basis), solve the
    generalized problem ``H(k) c = ε S(k) c``; otherwise ordinary ``eigvalsh``.
    """
    H_path = inverse_fourier_transform(H_R, kpts_frac, R_frac)
    S_path = (
        None
        if S_R is None
        else inverse_fourier_transform(S_R, kpts_frac, R_frac)
    )
    nk, n = H_path.shape[0], H_path.shape[1]
    E = np.empty((nk, n), dtype=np.float64)
    for ik in range(nk):
        H = 0.5 * (H_path[ik] + H_path[ik].conj().T)
        if S_path is None:
            E[ik] = np.linalg.eigvalsh(H).real
        else:
            S = 0.5 * (S_path[ik] + S_path[ik].conj().T)
            # Clip tiny / negative overlap eigenvalues for numerical stability.
            se, U = np.linalg.eigh(S)
            se = np.maximum(se.real, 1e-8)
            s_inv_sqrt = (U * (se ** (-0.5))) @ U.conj().T
            H_orth = s_inv_sqrt @ H @ s_inv_sqrt
            H_orth = 0.5 * (H_orth + H_orth.conj().T)
            E[ik] = np.linalg.eigvalsh(H_orth).real
    return E


def plot_bands_comparison(
    dft_bands: np.ndarray,
    out_path: Path,
    *,
    proj_bands: np.ndarray | None = None,
    dis_bands: np.ndarray | None = None,
    w90_dis_bands: np.ndarray | None = None,
    emin: float = -25.0,
    emax: float = 30.0,
) -> Path:
    """Overlay DFT, AMN-projected, local, and/or W90 U_dis bands on one path."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x_dft = np.arange(dft_bands.shape[0])
    n_ref = dft_bands.shape[0]

    for b in range(dft_bands.shape[1]):
        ax.plot(
            x_dft,
            dft_bands[:, b],
            color="#1f77b4",
            lw=0.7,
            alpha=0.35,
            label="DFT (bands.out)" if b == 0 else None,
            zorder=2,
        )
    if proj_bands is not None:
        x = np.arange(proj_bands.shape[0])
        n_ref = max(n_ref, proj_bands.shape[0])
        for b in range(proj_bands.shape[1]):
            ax.plot(
                x,
                proj_bands[:, b],
                color="#9467bd",
                lw=1.4,
                alpha=0.9,
                label=r"AMN-proj $H{=}A^\dagger\varepsilon A$, $S{=}A^\dagger A$ (via $H(R),S(R)$)" if b == 0 else None,
                zorder=3,
            )
    if dis_bands is not None:
        x = np.arange(dis_bands.shape[0])
        n_ref = max(n_ref, dis_bands.shape[0])
        for b in range(dis_bands.shape[1]):
            ax.plot(
                x,
                dis_bands[:, b],
                color="#d62728",
                lw=1.6,
                alpha=0.95,
                label=r"local $H{=}U^\dagger\varepsilon U$ (via $H(R)$)" if b == 0 else None,
                zorder=4,
            )
    if w90_dis_bands is not None:
        x = np.arange(w90_dis_bands.shape[0])
        n_ref = max(n_ref, w90_dis_bands.shape[0])
        for b in range(w90_dis_bands.shape[1]):
            ax.plot(
                x,
                w90_dis_bands[:, b],
                color="#2ca02c",
                lw=1.5,
                alpha=0.95,
                ls="--",
                label=r"W90 $U_{\mathrm{dis}}$ $H{=}U^\dagger\varepsilon U$ (via $H(R)$)"
                if b == 0
                else None,
                zorder=5,
            )

    n_seg = len(BAND_SYMMETRY_POINTS) - 1
    n_k = BAND_PATH_POINTS if n_ref <= 1 else max(1, (n_ref - 1) // n_seg)
    tick_pos = [i * n_k for i in range(n_seg + 1)]
    tick_pos[-1] = n_ref - 1
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([r"$\Gamma$", "M", "K", r"$\Gamma$"])
    for xp in tick_pos[1:-1]:
        ax.axvline(xp, color="0.7", lw=0.6, zorder=1)

    ax.set_xlim(0, n_ref - 1)
    ax.set_ylim(emin, emax)
    ax.set_ylabel("Energy (eV)")
    ax.set_xlabel("k-path")
    ax.set_title("Projected / disentangled bands vs DFT (via H(R))")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.axhline(0.0, color="0.85", lw=0.5, zorder=0)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def load_geometry_from_scf_in(
    scf_in: Path | str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Parse ``CELL_PARAMETERS`` / ``ATOMIC_POSITIONS`` (angstrom) from a QE ``.in``."""
    lines = Path(scf_in).read_text(encoding="utf-8", errors="replace").splitlines()

    def find_line(tag: str) -> int:
        for i, ln in enumerate(lines):
            if ln.strip().upper().startswith(tag):
                return i
        raise ValueError(f"No '{tag}' block in {scf_in}")

    i_cell = find_line("CELL_PARAMETERS")
    if "angstrom" not in lines[i_cell].lower():
        raise ValueError(f"Expected 'CELL_PARAMETERS angstrom' in {scf_in}")
    cell = np.array(
        [[float(x) for x in lines[i_cell + 1 + r].split()[:3]] for r in range(3)],
        dtype=np.float64,
    )

    i_pos = find_line("ATOMIC_POSITIONS")
    if "angstrom" not in lines[i_pos].lower():
        raise ValueError(f"Expected 'ATOMIC_POSITIONS angstrom' in {scf_in}")
    keywords = ("K_POINTS", "CELL_PARAMETERS", "ATOMIC_SPECIES", "ATOMIC_FORCES")
    pos: list[list[float]] = []
    r = i_pos + 1
    while r < len(lines):
        s = lines[r].strip()
        if not s or s.upper().startswith(keywords):
            break
        parts = s.split()
        if len(parts) < 4:
            break
        pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
        r += 1
    if not pos:
        raise ValueError(f"No atoms parsed from ATOMIC_POSITIONS in {scf_in}")
    return np.asarray(pos, dtype=np.float64), cell, len(pos)


ALLOWED_ORB_PER_ATOM = (4, 8)


def infer_n_orb_per_atom(num_wann: int, n_atoms: int) -> int:
    """
    Orbitals per atom from ``num_wann / n_atoms``.

    Allowed values are 4 (one sp³ shell: s,px,py,pz) or 8 (two radial
    shells × sp³). Raises if ``num_wann`` is not divisible by ``n_atoms``
    or the quotient is not in ``ALLOWED_ORB_PER_ATOM``.
    """
    if n_atoms < 1:
        raise ValueError(f"n_atoms must be ≥ 1, got {n_atoms}")
    if num_wann < 1:
        raise ValueError(f"num_wann must be ≥ 1, got {num_wann}")
    if num_wann % n_atoms != 0:
        raise ValueError(
            f"num_wann={num_wann} is not divisible by n_atoms={n_atoms}"
        )
    n_orb = num_wann // n_atoms
    if n_orb not in ALLOWED_ORB_PER_ATOM:
        raise ValueError(
            f"Inferred {n_orb} orbitals/atom from num_wann={num_wann}, "
            f"n_atoms={n_atoms}; expected one of {ALLOWED_ORB_PER_ATOM}"
        )
    return int(n_orb)


def amn_psi_phi_tb_order_flexible(
    A_amn: np.ndarray,
    *,
    n_atoms: int,
    n_orb_per_atom: int,
) -> np.ndarray:
    """
    ``⟨ψ|φ⟩`` with each sp³ shell reordered QE ``s,pz,px,py`` → TB ``s,px,py,pz``.

    Supports ``n_orb_per_atom`` in ``{4, 8}`` (one or two radial shells per atom,
    orbitals grouped contiguously by atom then by shell).
    """
    A = np.asarray(A_amn)
    expect = n_atoms * n_orb_per_atom
    if A.shape[-1] != expect:
        raise ValueError(
            f"A last dim {A.shape[-1]} != n_atoms*n_orb_per_atom={expect}"
        )
    if n_orb_per_atom not in ALLOWED_ORB_PER_ATOM:
        raise ValueError(
            f"n_orb_per_atom={n_orb_per_atom} not in {ALLOWED_ORB_PER_ATOM}"
        )
    if n_orb_per_atom % 4 != 0:
        raise ValueError(f"n_orb_per_atom={n_orb_per_atom} must be a multiple of 4")
    n_shells = n_orb_per_atom // 4
    # QE shell indices → TB: s,px,py,pz from s,pz,px,py
    qe_to_tb = (0, 2, 3, 1)
    out = np.empty_like(A)
    for a in range(n_atoms):
        for s in range(n_shells):
            base = a * n_orb_per_atom + 4 * s
            for t, q in enumerate(qe_to_tb):
                out[..., base + t] = A[..., base + q]
    return out


def atom_block_frobenius_vs_distance(
    H_R: np.ndarray,
    R_frac: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    *,
    n_orb_per_atom: int | None = None,
    min_dist: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Frobenius norms of atom-pair blocks ``H_{ij}(R)`` vs interatomic distance.

    Wannier orbitals are grouped as ``n_orb_per_atom`` consecutive indices per
    atom (4 or 8). If ``n_orb_per_atom`` is omitted it is inferred from
    ``H_R.shape[-1] / n_atoms``. Onsite ``i=j, R=0`` is omitted.
    """
    nR, n_orb, _ = H_R.shape
    n_atoms = positions.shape[0]
    if n_orb_per_atom is None:
        n_orb_per_atom = infer_n_orb_per_atom(n_orb, n_atoms)
    elif n_orb != n_atoms * n_orb_per_atom:
        raise ValueError(
            f"n_orb={n_orb} != n_atoms*n_orb_per_atom="
            f"{n_atoms * n_orb_per_atom}; cannot map orbitals to atoms"
        )
    R_cart = R_frac.astype(np.float64) @ cell
    dists: list[float] = []
    norms: list[float] = []
    for iR in range(nR):
        Rc = R_cart[iR]
        R_is_zero = abs(R_frac[iR]).max() < 1e-12
        for ia in range(n_atoms):
            for ja in range(n_atoms):
                if ia == ja and R_is_zero:
                    continue
                d = float(np.linalg.norm(positions[ja] + Rc - positions[ia]))
                if d < min_dist:
                    continue
                oi, oj = ia * n_orb_per_atom, ja * n_orb_per_atom
                block = H_R[iR, oi : oi + n_orb_per_atom, oj : oj + n_orb_per_atom]
                dists.append(d)
                norms.append(float(np.linalg.norm(block)))
    return np.asarray(dists, dtype=np.float64), np.asarray(norms, dtype=np.float64)


def plot_hopping_vs_distance(
    series: list[dict],
    out_path: Path,
    *,
    dmax: float | None = None,
) -> Path:
    """
    Atom-centered hopping plot: ``‖H_ij(R)‖_F`` vs interatomic distance.

    Each entry of ``series`` needs ``label``, ``color``, ``dist_atom``,
    ``hop_atom``.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for s in series:
        ax.scatter(
            s["dist_atom"],
            s["hop_atom"],
            s=18,
            alpha=0.6,
            color=s["color"],
            label=s["label"],
            edgecolors="none",
            zorder=3,
        )
    ax.set_xlabel(r"Interatomic distance $|\mathbf{r}_j + \mathbf{R} - \mathbf{r}_i|$ (Å)")
    ax.set_ylabel(r"$\|H_{ij}(R)\|_F$ (eV)")
    ax.set_title("Real-space disentangled hoppings (atom-centered blocks)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlim(left=0.0)
    if dmax is not None:
        ax.set_xlim(0.0, dmax)
    ax.axhline(0.0, color="0.85", lw=0.5, zorder=0)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build and plot disentangled / AMN-projected bands from a QE+W90 "
            "directory against DFT bands.out. Choose the disentanglement "
            "source with --source."
        )
    )
    p.add_argument(
        "--qe-dir",
        type=Path,
        default=HERE / "qe",
        help="Directory with .amn/.eig/.chk/.win and bands.out (default: ./qe).",
    )
    p.add_argument(
        "--seed",
        type=str,
        default=SEEDNAME,
        help=f"Wannier90 seedname (default: {SEEDNAME}).",
    )
    p.add_argument(
        "--source",
        choices=("w90", "local", "both"),
        default="both",
        help=(
            "Disentanglement source to plot: "
            "'w90' = U_dis from seed.chk (Wannier90), "
            "'local' = Adam-SMV Disentangler in this script (needs .mmn), "
            "'both' = overlay both (default)."
        ),
    )
    p.add_argument(
        "--num-iter",
        type=int,
        default=200,
        help="Adam iterations for local SMV (default: 200).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Adam learning rate for local SMV (default: 0.05).",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Convergence tolerance on Delta Omega_I for local SMV.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: <qe-dir>/w90_bands_vs_dft.png).",
    )
    p.add_argument(
        "--hop-out",
        type=Path,
        default=None,
        help="Hopping PNG (default: <qe-dir>/disentangled_hoppings.png).",
    )
    p.add_argument(
        "--hop-dmax",
        type=float,
        default=None,
        help="Optional x-axis cutoff (Ang) for the atom-centered hopping panel.",
    )
    p.add_argument(
        "--with-amn-proj",
        action="store_true",
        help="Also overlay AMN-projected (non-orthogonal) bands via H(R),S(R).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    qe_dir = Path(args.qe_dir).resolve()
    seed = args.seed
    use_w90 = args.source in ("w90", "both")
    use_local = args.source in ("local", "both")
    use_amn_proj = bool(args.with_amn_proj)

    print(f"Loading Wannier90 data from {qe_dir}")
    print(f"  seed={seed}  disentanglement source: {args.source}")
    A_amn, num_bands_amn, num_kpts_amn, num_wann = read_amn(qe_dir / f"{seed}.amn")
    eig = read_eigenvalues(qe_dir / f"{seed}.eig")
    kpts = read_kpoints_nnkp(qe_dir / f"{seed}.nnkp")
    print(f"  AMN/EIG: Nk={num_kpts_amn}, M={num_bands_amn}, N={num_wann}")

    positions, cell, n_atoms = load_geometry_from_scf_in(qe_dir / "scf.in")
    n_orb_per_atom = infer_n_orb_per_atom(num_wann, n_atoms)
    print(
        f"  geometry: n_atoms={n_atoms}, n_orb/atom={n_orb_per_atom} "
        f"(from num_wann/n_atoms)"
    )

    win_path = qe_dir / f"{seed}.win"
    if win_path.is_file():
        mp_grid = read_mp_grid_win(win_path)
    else:
        mp_grid = infer_mp_grid_from_kpoints(kpts)
    R_frac = centered_mp_R_vectors(*mp_grid)

    H_R_proj = None
    S_R_proj = None
    if use_amn_proj:
        H_k_proj, S_k_proj = build_projected_Hk_Sk(
            A_amn,
            eig,
            n_atoms=n_atoms,
            n_orb_per_atom=n_orb_per_atom,
        )
        H_R_proj = fourier_transform(H_k_proj, kpts, R_frac)
        S_R_proj = fourier_transform(S_k_proj, kpts, R_frac)
        print(
            f"projected H(k)/S(k) {H_k_proj.shape} -> H(R)/S(R) {H_R_proj.shape} "
            f"(all DFT bands)"
        )

    H_R_w90 = None
    R_frac_chk = None
    if use_w90:
        chk_path = qe_dir / f"{seed}.chk"
        if not chk_path.is_file():
            raise FileNotFoundError(
                f"--source={args.source} requires {chk_path}. "
                "Run wannier90.x first, or pass --source local."
            )
        U_dis_w90, kpts_chk, mp_chk = read_chk_u_dis(chk_path)
        if eig.shape != (U_dis_w90.shape[0], U_dis_w90.shape[1]):
            raise ValueError(
                f".eig shape {eig.shape} incompatible with U_dis "
                f"{U_dis_w90.shape[:2]}"
            )
        R_frac_chk = centered_mp_R_vectors(*mp_chk)
        # H(k) = U_dis^\dagger(k) diag(eps) U_dis(k)  (chk U_matrix_opt)
        H_k_w90 = build_Hk_from_U(U_dis_w90, eig)
        H_R_w90 = fourier_transform(H_k_w90, kpts_chk, R_frac_chk)
        print(
            f"W90 U_dis H(k) {H_k_w90.shape} -> H(R) {H_R_w90.shape}, "
            f"mp_grid={mp_chk}, ||U\u2020U-I||_G="
            f"{np.linalg.norm(U_dis_w90[0].conj().T @ U_dis_w90[0] - np.eye(U_dis_w90.shape[2])):.2e}"
        )

    H_R_dis = None
    if use_local:
        mmn_path = qe_dir / f"{seed}.mmn"
        if not mmn_path.is_file():
            raise FileNotFoundError(
                f"--source={args.source} requires {mmn_path} for local SMV."
            )
        mmn_mb = mmn_path.stat().st_size / (1024 ** 2)
        print(f"Loading MMN ({mmn_mb:.1f} MB) for local Adam-SMV ...")
        amn, mmn, neighbors, weights, num_wann2, eig2, kpts2 = load_disentangle_data(
            qe_dir, seed
        )
        model = Disentangler(
            amn, mmn, neighbors, weights, num_wann=num_wann2,
            num_iter=args.num_iter, tol=args.tol, lr=args.lr,
        )
        U_opt = model.run()
        H_k_dis = build_disentangled_Hk(U_opt, eig2)
        H_R_dis = fourier_transform(H_k_dis, kpts2, R_frac)
        print(f"local dis H(k) {H_k_dis.shape} -> H(R) {H_R_dis.shape}")

    bands_out = qe_dir / "bands.out"
    path_k = parse_bands_out_kpath(bands_out)
    proj_path = (
        bands_from_HR(H_R_proj, R_frac, path_k, S_R=S_R_proj)
        if H_R_proj is not None
        else None
    )
    w90_path = (
        bands_from_HR(H_R_w90, R_frac_chk, path_k) if H_R_w90 is not None else None
    )
    dis_path = (
        bands_from_HR(H_R_dis, R_frac, path_k) if H_R_dis is not None else None
    )
    print(f"Path from bands.out: {path_k.shape}")
    if proj_path is not None:
        print(f"  projected FT bands: {proj_path.shape}")
    if w90_path is not None:
        print(f"  W90 U_dis FT bands: {w90_path.shape}")
    if dis_path is not None:
        print(f"  local dis FT bands: {dis_path.shape}")

    dft = parse_qe_bands_out(bands_out)
    if dft is None:
        raise RuntimeError(f"Could not parse DFT bands from {bands_out}")
    print(f"DFT bands.out: {dft.shape}")

    out_png = args.out if args.out is not None else qe_dir / "w90_bands_vs_dft.png"
    plot_bands_comparison(
        dft,
        out_png,
        proj_bands=proj_path,
        dis_bands=dis_path,
        w90_dis_bands=w90_path,
    )
    print(f"Wrote {out_png}")

    hop_series: list[dict] = []
    if H_R_w90 is not None:
        d_atom, h_atom = atom_block_frobenius_vs_distance(
            H_R_w90,
            R_frac_chk,
            positions,
            cell,
            n_orb_per_atom=n_orb_per_atom,
        )
        hop_series.append(
            {
                "label": r"W90 $U_{\mathrm{dis}}$",
                "color": "#2ca02c",
                "dist_atom": d_atom,
                "hop_atom": h_atom,
            }
        )
        print(
            f"  W90 hoppings: {h_atom.size} atom blocks, "
            f"d in [{d_atom.min():.2f},{d_atom.max():.2f}] A, "
            f"max||H||_F={h_atom.max():.3f} eV"
        )

    if H_R_dis is not None:
        d_atom, h_atom = atom_block_frobenius_vs_distance(
            H_R_dis,
            R_frac,
            positions,
            cell,
            n_orb_per_atom=n_orb_per_atom,
        )
        hop_series.append(
            {
                "label": "local Adam-SMV",
                "color": "#d62728",
                "dist_atom": d_atom,
                "hop_atom": h_atom,
            }
        )
        print(
            f"  local hoppings: {h_atom.size} atom blocks, "
            f"d in [{d_atom.min():.2f},{d_atom.max():.2f}] A, "
            f"max||H||_F={h_atom.max():.3f} eV"
        )

    if hop_series:
        hop_png = (
            args.hop_out if args.hop_out is not None else qe_dir / "disentangled_hoppings.png"
        )
        plot_hopping_vs_distance(hop_series, hop_png, dmax=args.hop_dmax)
        print(f"Wrote {hop_png}")
    else:
        print("No disentangled H(R) available; skipping hopping plot.")
