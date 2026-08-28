"""Save ``Ω_I`` history, ``V(k)``, ``H(k)``, ``H(R)``, and comparison plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from .band_io import parse_qe_bands_out
from .disentangle import (
    bands_from_HR,
    centered_mp_R_vectors,
    parse_bands_out_kpath,
    plot_bands_comparison,
)
from .sk_transforms import sk_integrals_from_block
from .wannier_io import fourier_transform

# Wannier90 (l, mr) angular labels (user-guide convention).
_W90_ORB_NAME: dict[tuple[int, int], str] = {
    (0, 1): "s",
    (1, 1): "pz",
    (1, 2): "px",
    (1, 3): "py",
    (2, 1): "dz2",
    (2, 2): "dxz",
    (2, 3): "dyz",
    (2, 4): "dx2-y2",
    (2, 5): "dxy",
}

# SK orbital order used by ``sk_integrals_from_block``.
_SK_ORDER = ("s", "px", "py", "pz")
_BOND_LABELS = (
    r"$V_{ss\sigma}$",
    r"$V_{sp\sigma}$",
    r"$V_{pp\sigma}$",
    r"$V_{pp\pi}$",
)
_BOND_KEYS = ("ss_sigma", "sp_sigma", "pp_sigma", "pp_pi")



def orbital_name(l: int, mr: int) -> str:
    """Human-readable angular label for a W90 projection ``(l, mr)``."""
    return _W90_ORB_NAME.get((int(l), int(mr)), f"l{l}_mr{mr}")


def projection_labels(orbitals: list) -> list[str]:
    """Ordered angular labels for the trial-orbital set (from ``.nnkp`` / ``.win``)."""
    return [orbital_name(int(o.l), int(o.mr)) for o in orbitals]


def is_complete_sp3(
    orbitals: list,
    real_lattice_ang: NDArray[np.float64],
    *,
    tol: float = 1e-8,
) -> bool:
    """
    True when every crystallo-graphic site carries a full ``s,px,py,pz`` shell.

    Used to gate Slater–Koster 4×4 bond-integral analysis, which requires
    complete sp³ atoms. Subsets (e.g. ``C:pz`` only) return False.
    """
    try:
        group_sp3_atoms(orbitals, real_lattice_ang, tol=tol)
    except ValueError:
        return False
    return True


def pair_channel(name_i: str, name_j: str) -> str:
    """
    Undirected hopping channel for the legend (``s-px`` ≡ ``px-s``).
    """
    a, b = sorted((name_i, name_j))
    return f"{a}-{b}"


def build_HR_from_Hk(
    H_k: NDArray[np.complex128],
    kpts_crystal: NDArray[np.float64],
    mp_grid: tuple[int, int, int],
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """
    ``H(k) → H(R)`` on the centered Monkhorst–Pack lattice shells.

    Returns ``(H_R, R_frac)`` with shapes ``(nR, J, J)`` and ``(nR, 3)``.
    """
    R_frac = centered_mp_R_vectors(*mp_grid)
    H_R = fourier_transform(H_k, kpts_crystal, R_frac)
    return H_R, R_frac


def hoppings_vs_distance(
    H_R: NDArray[np.complex128],
    R_frac: NDArray[np.float64],
    *,
    orbitals: list,
    real_lattice_ang: NDArray[np.float64],
    min_dist: float = 0.0,
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Collect ``Re[H_ij(R)]`` vs orbital–orbital distance, grouped by channel.

    Distance is ``|τ_j + R − τ_i|`` in Å. Hermitian partners
    ``(i,j,R)`` / ``(j,i,−R)`` are deduplicated. Returns
    ``{channel: (dist, ReH)}``.
    """
    nR, J, _ = H_R.shape
    if len(orbitals) != J:
        raise ValueError(f"len(orbitals)={len(orbitals)} != J={J}")
    cell = np.asarray(real_lattice_ang, dtype=np.float64)
    pos = np.array([np.asarray(o.tau_crystal, dtype=np.float64) @ cell for o in orbitals])
    names = [orbital_name(int(o.l), int(o.mr)) for o in orbitals]
    R_cart = R_frac.astype(np.float64) @ cell

    buckets: dict[str, list[tuple[float, float]]] = {}
    for iR in range(nR):
        Rf = tuple(np.round(R_frac[iR], decimals=10))
        Rfm = tuple(np.round(-R_frac[iR], decimals=10))
        Rc = R_cart[iR]
        for i in range(J):
            for j in range(J):
                # Keep one of each Hermitian pair
                if (Rf, i, j) > (Rfm, j, i):
                    continue
                d = float(np.linalg.norm(pos[j] + Rc - pos[i]))
                if d < min_dist:
                    continue
                ch = pair_channel(names[i], names[j])
                val = float(H_R[iR, i, j].real)
                buckets.setdefault(ch, []).append((d, val))

    out: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for ch, pairs in buckets.items():
        arr = np.asarray(pairs, dtype=np.float64)
        out[ch] = (arr[:, 0], arr[:, 1])
    return out


def plot_HR_vs_distance(
    out_dir: Path | str,
    *,
    H_R: NDArray[np.complex128],
    R_frac: NDArray[np.float64],
    orbitals: list,
    real_lattice_ang: NDArray[np.float64],
    dmax: float = 6.0,
) -> Path:
    """
    One subplot per orbital channel (``s-px``, ``s-py``, …):
    ``Re[H_ij(R)]`` vs orbital distance, limited to ``d ≤ dmax`` (Å).
    """
    out_dir = Path(out_dir)
    series = hoppings_vs_distance(
        H_R, R_frac, orbitals=orbitals, real_lattice_ang=real_lattice_ang
    )
    channels = sorted(series.keys())
    n = len(channels)
    if n == 0:
        raise RuntimeError("No H(R) hoppings to plot")

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 2.6 * nrows),
        sharex=True,
        squeeze=False,
    )
    cmap = plt.get_cmap("tab20")
    for idx, ch in enumerate(channels):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        d, h = series[ch]
        mask = d <= dmax
        ax.scatter(
            d[mask],
            h[mask],
            s=14,
            alpha=0.75,
            color=cmap(idx % 20),
            edgecolors="none",
            zorder=3,
        )
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
        ax.set_title(ch, fontsize=10)
        ax.set_xlim(0.0, dmax)
        if r == nrows - 1:
            ax.set_xlabel(r"$|\boldsymbol{\tau}_j+\mathbf{R}-\boldsymbol{\tau}_i|$ (Å)")
        if c == 0:
            ax.set_ylabel(r"$\mathrm{Re}\,H_{ij}(\mathbf{R})$ (eV)")

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        rf"Real-space hoppings $H(\mathbf{{R}})$ by channel  ($d\leq{dmax:g}$ Å)",
        fontsize=11,
    )
    fig.tight_layout()
    out = out_dir / "H_R_vs_distance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def group_sp3_atoms(
    orbitals: list,
    real_lattice_ang: NDArray[np.float64],
    *,
    tol: float = 1e-8,
) -> tuple[NDArray[np.float64], list[list[int]]]:
    """
    Group trial orbitals into sp³ atoms by crystal site.

    Returns
    -------
    positions_ang : (n_atoms, 3)
    atom_orbs : list of length-4 global orbital index lists in **SK order**
        ``(s, px, py, pz)``.
    """
    cell = np.asarray(real_lattice_ang, dtype=np.float64)
    sites: list[NDArray[np.float64]] = []
    by_site: list[dict[str, int]] = []
    for ig, o in enumerate(orbitals):
        tau = np.asarray(o.tau_crystal, dtype=np.float64)
        name = orbital_name(int(o.l), int(o.mr))
        matched = None
        for ia, tau0 in enumerate(sites):
            if np.linalg.norm(tau - tau0) < tol:
                matched = ia
                break
        if matched is None:
            sites.append(tau.copy())
            by_site.append({})
            matched = len(sites) - 1
        if name in by_site[matched]:
            raise ValueError(
                f"Duplicate orbital '{name}' on atom {matched} "
                f"(indices {by_site[matched][name]}, {ig})"
            )
        by_site[matched][name] = ig

    atom_orbs: list[list[int]] = []
    for ia, mapping in enumerate(by_site):
        missing = [n for n in _SK_ORDER if n not in mapping]
        if missing:
            raise ValueError(f"Atom {ia} missing orbitals {missing}; have {sorted(mapping)}")
        atom_orbs.append([mapping[n] for n in _SK_ORDER])

    positions = np.array([tau @ cell for tau in sites], dtype=np.float64)
    return positions, atom_orbs


def bond_integrals_vs_distance(
    H_R: NDArray[np.complex128],
    R_frac: NDArray[np.float64],
    *,
    orbitals: list,
    real_lattice_ang: NDArray[np.float64],
    min_dist: float = 1e-6,
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Invert each atom–atom 4×4 block ``H_{ij}(R)`` to Slater–Koster integrals.

    Uses ``sk_integrals_from_block`` (Wikipedia convention) after reordering
    orbitals to ``(s, px, py, pz)``. Hermitian partners are deduplicated.
    Onsite ``i=j, R=0`` is skipped.

    Returns ``{ss_sigma|sp_sigma|pp_sigma|pp_pi: (dist_Å, V)}``.
    """
    positions, atom_orbs = group_sp3_atoms(orbitals, real_lattice_ang)
    n_atoms = positions.shape[0]
    cell = np.asarray(real_lattice_ang, dtype=np.float64)
    R_cart = R_frac.astype(np.float64) @ cell
    nR = R_frac.shape[0]

    buckets: dict[str, list[tuple[float, float]]] = {k: [] for k in _BOND_KEYS}
    for iR in range(nR):
        Rf = tuple(np.round(R_frac[iR], decimals=10))
        Rfm = tuple(np.round(-R_frac[iR], decimals=10))
        Rc = R_cart[iR]
        for ia in range(n_atoms):
            for ja in range(n_atoms):
                if (Rf, ia, ja) > (Rfm, ja, ia):
                    continue
                bond = positions[ja] + Rc - positions[ia]
                d = float(np.linalg.norm(bond))
                if d < min_dist:
                    continue
                ii = atom_orbs[ia]
                jj = atom_orbs[ja]
                block = np.asarray(H_R[iR][np.ix_(ii, jj)].real, dtype=np.float64)
                vals = sk_integrals_from_block(block, bond)
                if vals is None:
                    continue
                for key, v in zip(_BOND_KEYS, vals):
                    buckets[key].append((d, float(v)))

    out: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for key, pairs in buckets.items():
        if not pairs:
            out[key] = (
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
            continue
        arr = np.asarray(pairs, dtype=np.float64)
        out[key] = (arr[:, 0], arr[:, 1])
    return out


def plot_bond_integrals_vs_distance(
    out_dir: Path | str,
    *,
    H_R: NDArray[np.complex128],
    R_frac: NDArray[np.float64],
    orbitals: list,
    real_lattice_ang: NDArray[np.float64],
    dmax: float = 6.0,
) -> Path:
    """
    Four subplots: ``V_ssσ``, ``V_spσ``, ``V_ppσ``, ``V_ppπ`` extracted from
    each atom–atom 4×4 hopping block, vs interatomic distance (≤ ``dmax`` Å).
    """
    out_dir = Path(out_dir)
    series = bond_integrals_vs_distance(
        H_R, R_frac, orbitals=orbitals, real_lattice_ang=real_lattice_ang
    )
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), sharex=True)
    for ax, key, label, color in zip(axes.ravel(), _BOND_KEYS, _BOND_LABELS, colors):
        d, v = series[key]
        mask = d <= dmax
        ax.scatter(
            d[mask],
            v[mask],
            s=16,
            alpha=0.75,
            color=color,
            edgecolors="none",
            zorder=3,
        )
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(0.0, dmax)
        ax.set_ylabel(r"$V$ (eV)")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$|\mathbf{r}_j + \mathbf{R} - \mathbf{r}_i|$ (Å)")
    fig.suptitle(
        rf"Slater–Koster bond integrals from $4\times 4$ $H_{{ij}}(\mathbf{{R}})$ "
        rf"($d\leq{dmax:g}$ Å)",
        fontsize=11,
    )
    fig.tight_layout()
    out = out_dir / "bond_integrals_vs_distance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # Also stash numeric table for reuse
    np.savez(
        out_dir / "bond_integrals.npz",
        **{f"{k}_d": series[k][0] for k in _BOND_KEYS},
        **{f"{k}_V": series[k][1] for k in _BOND_KEYS},
    )
    return out


def build_Hk_from_V(
    V_list: list[torch.Tensor],
    eig: NDArray[np.float64],
    band_idx: list[NDArray[np.int64]],
) -> NDArray[np.complex128]:
    """
    Disentangled Bloch-gauge Hamiltonian (before MLWF unitary)::

        H(k) = V(k)† E(k) V(k)

    with ``E(k) = diag(ε_n)`` restricted to the outer window at ``k``.
    """
    Nk = len(V_list)
    J = V_list[0].shape[1]
    H_k = np.empty((Nk, J, J), dtype=np.complex128)
    for ik, V in enumerate(V_list):
        idx = band_idx[ik]
        eps = torch.as_tensor(eig[ik, idx], dtype=torch.float64)
        Hk = V.conj().T @ (eps.to(dtype=V.dtype)[:, None] * V)
        Hk = 0.5 * (Hk + Hk.conj().T)
        H_k[ik] = Hk.detach().cpu().numpy()
    return H_k


def check_interlacing(
    H_k: NDArray[np.complex128],
    eig: NDArray[np.float64],
    band_idx: list[NDArray[np.int64]],
    *,
    n_sample: int = 5,
    tol: float = 1e-4,
) -> list[str]:
    """Cauchy interlacing sanity check on a few k-points."""
    warnings: list[str] = []
    Nk = H_k.shape[0]
    sample = np.linspace(0, Nk - 1, num=min(n_sample, Nk), dtype=int)
    for ik in sample:
        eps_win = np.sort(eig[ik, band_idx[ik]].real)
        ev = np.linalg.eigvalsh(H_k[ik]).real
        # Every disentangled eigenvalue must lie within [ε_min, ε_max] of the window
        if ev.min() < eps_win.min() - tol or ev.max() > eps_win.max() + tol:
            warnings.append(
                f"k={ik}: H eigenvalues [{ev.min():.4f},{ev.max():.4f}] "
                f"outside window [{eps_win.min():.4f},{eps_win.max():.4f}]"
            )
        # Interlace: ε_j ≤ E_j ≤ ε_{j+M-N} roughly for sorted spectra
        M, N = eps_win.size, ev.size
        for j in range(N):
            lo = eps_win[j]
            hi = eps_win[j + (M - N)]
            if ev[j] < lo - tol or ev[j] > hi + tol:
                warnings.append(
                    f"k={ik}: interlacing violated at j={j}: "
                    f"E={ev[j]:.4f} not in [{lo:.4f},{hi:.4f}]"
                )
                break
    return warnings


def save_outputs(
    out_dir: Path | str,
    *,
    omega_history: list[float],
    omega_final: float,
    theta: dict[int, torch.Tensor],
    V_list: list[torch.Tensor],
    H_k: NDArray[np.complex128],
    kpts_crystal: NDArray[np.float64],
    meta: dict,
    history_I: list[float] | None = None,
    history_D: list[float] | None = None,
    H_R: NDArray[np.complex128] | None = None,
    R_frac: NDArray[np.float64] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "V": V_list,
        "H_k": torch.from_numpy(H_k),
        "kpts_crystal": torch.from_numpy(kpts_crystal),
        "theta": theta,
        "omega_P": omega_final,
        "note": (
            "H(k)=V† E V is the disentangled Bloch-gauge Hamiltonian "
            "(before maximal-localization unitary U(k)). Eigenvalues are "
            "gauge-invariant; individual matrix elements are not final MLWF hoppings. "
            "H_R / R_frac are the Fourier transform on the MP R-mesh."
        ),
    }
    if H_R is not None:
        payload["H_R"] = torch.from_numpy(np.asarray(H_R))
    if R_frac is not None:
        payload["R_frac"] = torch.from_numpy(np.asarray(R_frac, dtype=np.float64))
    torch.save(payload, out_dir / "V_H_theta.pt")

    np.save(out_dir / "omega_P_history.npy", np.asarray(omega_history, dtype=np.float64))
    if history_I is not None:
        np.save(out_dir / "omega_I_history.npy", np.asarray(history_I, dtype=np.float64))
    if history_D is not None:
        np.save(out_dir / "omega_D_history.npy", np.asarray(history_D, dtype=np.float64))
    np.save(out_dir / "H_k.npy", H_k)
    if H_R is not None:
        np.save(out_dir / "H_R.npy", H_R)
    if R_frac is not None:
        np.save(out_dir / "R_frac.npy", np.asarray(R_frac, dtype=np.float64))
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(omega_history, "-o", ms=3, label=r"$\Omega_P$")
    if history_I is not None:
        ax.plot(history_I, "--", lw=1.2, label=r"$\Omega_I/J$")
    if history_D is not None:
        ax.plot(history_D, ":", lw=1.2, label=r"$\Omega_D$")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("Objective")
    ax.set_title(rf"Final $\Omega_P$ = {omega_final:.6f}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "omega_P_history.png", dpi=150)
    plt.close(fig)
    return out_dir


def plot_lpdos_gamma_K(
    out_dir: Path | str,
    *,
    V_list: list[torch.Tensor],
    eig: NDArray[np.float64],
    band_idx: list[NDArray[np.int64]],
    kpts_crystal: NDArray[np.float64],
    orb_l: list[int],
    grid,
) -> Path | None:
    """
    LPDOS comparison at Γ and K for angular channels present in ``orb_l``.

    Columns are whatever of ``s`` / ``p`` appear among the projections
    (so ``C:pz`` alone yields a single ``p`` column). Returns ``None`` if no
    channel has orbitals.
    """
    from .model.omega_D import (
        EnergyGrid,
        channel_mask,
        find_mesh_k_index,
        lpdos_channel_curves,
    )

    out_dir = Path(out_dir)
    # Ensure LPDOS grid lives on the same device as V
    if isinstance(grid, EnergyGrid) and grid.E.device.type != "cpu":
        grid = EnergyGrid(E=grid.E.detach().cpu(), dE=grid.dE, sigma=grid.sigma)
    k_gamma = np.array([0.0, 0.0, 0.0])
    k_K = np.array([2.0 / 3.0, 1.0 / 3.0, 0.0])
    ik_G = find_mesh_k_index(kpts_crystal, k_gamma)
    ik_K = find_mesh_k_index(kpts_crystal, k_K)

    channels: list[tuple[str, list[int], str]] = []
    s_idx = channel_mask(orb_l, "s")
    p_idx = channel_mask(orb_l, "p")
    if s_idx:
        channels.append(("s", s_idx, r"$s$"))
    if p_idx:
        channels.append(("p", p_idx, r"$p$"))
    if not channels:
        return None

    nrows = 2
    ncols = len(channels)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 6.5),
        sharex=True,
        squeeze=False,
    )
    k_rows = [
        (ik_G, r"$\Gamma$"),
        (ik_K, r"$K$"),
    ]
    for r, (ik, k_lab) in enumerate(k_rows):
        for c, (_ch, orb_idx, ch_lab) in enumerate(channels):
            ax = axes[r, c]
            eps = torch.as_tensor(eig[ik, band_idx[ik]], dtype=torch.float64)
            V = V_list[ik]
            if not isinstance(V, torch.Tensor):
                V = torch.as_tensor(V)
            E, dft, tb = lpdos_channel_curves(V, eps, grid, orb_idx)
            ax.plot(E, dft, color="#1f77b4", lw=1.6, label="LPDOS DFT")
            ax.plot(E, tb, color="#d62728", lw=1.6, ls="--", label="LPDOS TB")
            ax.set_title(rf"{k_lab}, {ch_lab}")
            ax.set_ylabel("LPDOS (arb.)")
            if r == 1:
                ax.set_xlabel("Energy (eV)")
            ax.legend(frameon=False, fontsize=7)
    ch_tag = "_".join(c[0] for c in channels)
    fig.suptitle(
        rf"LPDOS DFT vs TB  |  mesh $k_\Gamma$={ik_G}, $k_K$={ik_K}  "
        rf"({kpts_crystal[ik_K]})",
        fontsize=10,
    )
    fig.tight_layout()
    out = out_dir / f"lpdos_gamma_K_{ch_tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_disentangled_vs_dft(
    out_dir: Path | str,
    *,
    H_k: NDArray[np.complex128],
    kpts_mesh: NDArray[np.float64],
    mp_grid: tuple[int, int, int],
    bands_out: Path | str,
    emin: float = -25.0,
    emax: float = 30.0,
    H_R: NDArray[np.complex128] | None = None,
    R_frac: NDArray[np.float64] | None = None,
) -> Path:
    """
    Band path: mesh ``H(k) → H(R) → H(k_path)`` eigenvalues overlaid on DFT
    eigenvalues from ``bands.out``.
    """
    out_dir = Path(out_dir)
    if H_R is None or R_frac is None:
        H_R, R_frac = build_HR_from_Hk(H_k, kpts_mesh, mp_grid)
    k_path = parse_bands_out_kpath(bands_out)
    dis_bands = bands_from_HR(H_R, R_frac, k_path)
    dft = parse_qe_bands_out(Path(bands_out))
    # parse_qe_bands_out may return (nk, nb) or richer — handle ndarray
    if isinstance(dft, tuple):
        dft_bands = dft[0]
    else:
        dft_bands = np.asarray(dft)
    return plot_bands_comparison(
        dft_bands,
        out_dir / "bands_disentangled_vs_dft.png",
        dis_bands=dis_bands,
        emin=emin,
        emax=emax,
    )
