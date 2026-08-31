"""Read Wannier90 ``.mmn``, ``.eig``, ``.win``, ``.amn``, ``.nnkp`` files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ..disentangle import (
    read_mmn,
    read_nnkp_neighbors,
)
from ..wannier_io import (
    read_amn,
    read_eigenvalues,
    read_kpoints_nnkp,
    read_mp_grid_win,
)

@dataclass
class TrialOrbital:
    """One Wannier90-style trial orbital (angular part + site; radial free)."""

    tau_crystal: NDArray[np.float64]  # (3,) crystal coords
    tau_cart_ang: NDArray[np.float64]  # (3,) Cartesian Angstrom
    l: int
    mr: int  # Wannier90 mr index within l
    zona: float = 1.0


@dataclass
class WannierDataset:
    """Bundled W90 inputs needed by the Bessel–SMV pipeline."""

    amn: list[torch.Tensor]  # Nk of (num_bands, J)
    mmn: dict[tuple[int, int], torch.Tensor]
    neighbors: list[list[tuple[int, int]]]
    weights: torch.Tensor
    eig: NDArray[np.float64]  # (Nk, num_bands) eV
    kpts_crystal: NDArray[np.float64]  # (Nk, 3)
    orbitals: list[TrialOrbital]
    num_wann: int
    num_bands: int
    mp_grid: tuple[int, int, int]
    dis_win_max: float | None
    dis_win_min: float | None
    dis_froz_max: float | None
    dis_froz_min: float | None
    real_lattice_ang: NDArray[np.float64]  # (3, 3) rows = a1,a2,a3
    recip_lattice_ang: NDArray[np.float64]  # (3, 3) rows = b1,b2,b3 (Ang^-1)


def _parse_win_scalar(text: str, key: str) -> float | None:
    m = re.search(rf"^\s*{key}\s*=\s*([^\s!]+)", text, flags=re.IGNORECASE | re.M)
    if m is None:
        return None
    return float(m.group(1).lower().replace("d", "e"))


def parse_win_windows(path: Path | str) -> dict[str, float | None]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return {
        "dis_win_max": _parse_win_scalar(text, "dis_win_max"),
        "dis_win_min": _parse_win_scalar(text, "dis_win_min"),
        "dis_froz_max": _parse_win_scalar(text, "dis_froz_max"),
        "dis_froz_min": _parse_win_scalar(text, "dis_froz_min"),
    }


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


def read_nnkp_lattice(path: Path | str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(real_lattice_ang, recip_lattice_ang)`` with row vectors."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    real = np.array(
        [[float(x) for x in row.split()[:3]] for row in _parse_nnkp_block(lines, "real_lattice")[:3]],
        dtype=np.float64,
    )
    recip = np.array(
        [[float(x) for x in row.split()[:3]] for row in _parse_nnkp_block(lines, "recip_lattice")[:3]],
        dtype=np.float64,
    )
    return real, recip


def read_nnkp_projections(
    path: Path | str,
    real_lattice_ang: NDArray[np.float64],
) -> list[TrialOrbital]:
    """
    Parse Wannier90 ``begin projections`` from ``.nnkp``.

    Each projection is two lines::
        x y z   l  mr  r
        z-axis(3)  x-axis(3)  zona
    Positions are crystal coords; converted to Cartesian Angstrom using the
    real lattice.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    block = _parse_nnkp_block(lines, "projections")
    nproj = int(block[0].split()[0])
    orbitals: list[TrialOrbital] = []
    row = 1
    for _ in range(nproj):
        parts = block[row].split()
        row += 1
        tau_crys = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
        l = int(parts[3])
        mr = int(parts[4])
        extras = block[row].split()
        row += 1
        zona = float(extras[6]) if len(extras) >= 7 else 1.0
        tau_cart = tau_crys @ real_lattice_ang
        orbitals.append(
            TrialOrbital(
                tau_crystal=tau_crys,
                tau_cart_ang=tau_cart,
                l=l,
                mr=mr,
                zona=zona,
            )
        )
    if len(orbitals) != nproj:
        raise ValueError(f"Expected {nproj} projections, got {len(orbitals)}")
    return orbitals


def window_band_mask(
    eig_k: NDArray[np.float64],
    dis_win_min: float | None,
    dis_win_max: float | None,
) -> NDArray[np.bool_]:
    """Boolean mask of bands inside the outer disentanglement window at one k."""
    mask = np.ones(eig_k.shape[0], dtype=bool)
    if dis_win_min is not None:
        mask &= eig_k >= dis_win_min
    if dis_win_max is not None:
        mask &= eig_k <= dis_win_max
    return mask


def filter_band_idx_by_orbital_weight(
    amn: list[torch.Tensor],
    band_idx: list[NDArray[np.int64]],
    *,
    threshold: float,
    num_wann: int,
) -> tuple[list[NDArray[np.int64]], dict[str, float | int]]:
    """
    Drop DFT bands with low total trial-orbital weight at each k::

        w_m(k) = Σ_i |A_{m i}(k)|² = Σ_i |⟨ψ_{m k}|g_i⟩|²

    Keep band ``m`` only if ``w_m(k) ≥ threshold``. Applied to the already
    window-restricted ``band_idx``; the same indices slice AMN, MMN, and ε.

    Returns
    -------
    filtered_idx
        Per-k global band indices after the weight cut.
    stats
        Summary counts (before/after, min retained, threshold).

    Raises
    ------
    RuntimeError
        If any k retains fewer than ``num_wann`` bands (must have Nb ≥ J).
    """
    if threshold < 0.0:
        raise ValueError(f"threshold must be ≥ 0, got {threshold}")
    filtered: list[NDArray[np.int64]] = []
    n_before: list[int] = []
    n_after: list[int] = []
    for ik, idx in enumerate(band_idx):
        idx = np.asarray(idx, dtype=np.int64)
        A = amn[ik][idx]
        # A: (Nb, J) → weight per DFT band
        w = (A.abs() ** 2).sum(dim=-1).real.detach().cpu().numpy()
        keep = w >= threshold
        new_idx = idx[keep]
        n_before.append(int(idx.size))
        n_after.append(int(new_idx.size))
        if new_idx.size < num_wann:
            msg = (
                f"ERROR: after orbital-weight filter (threshold={threshold}) "
                f"at k={ik} only {new_idx.size} DFT bands remain, but "
                f"num_wann J={num_wann}. Need Nb ≥ J at every k. "
                f"Lower --proj-threshold (orbital weight threshold) and retry."
            )
            print(f"  WARNING: {msg}")
            raise RuntimeError(msg)
        filtered.append(new_idx.astype(np.int64))

    stats: dict[str, float | int] = {
        "threshold": float(threshold),
        "n_before_min": int(min(n_before)),
        "n_before_max": int(max(n_before)),
        "n_after_min": int(min(n_after)),
        "n_after_max": int(max(n_after)),
        "n_after_mean": float(np.mean(n_after)),
        "n_removed_total": int(sum(n_before) - sum(n_after)),
    }
    return filtered, stats


def load_wannier_dataset(qe_dir: Path | str, seed: str) -> WannierDataset:
    qe_dir = Path(qe_dir)
    amn_path = qe_dir / f"{seed}.amn"
    mmn_path = qe_dir / f"{seed}.mmn"
    nnkp_path = qe_dir / f"{seed}.nnkp"
    eig_path = qe_dir / f"{seed}.eig"
    win_path = qe_dir / f"{seed}.win"
    for p in (amn_path, mmn_path, nnkp_path, eig_path, win_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    A, num_bands, num_kpts, num_wann = read_amn(amn_path)
    mmn, nb_m, nk_m, nntot = read_mmn(mmn_path)
    neighbors, weights = read_nnkp_neighbors(nnkp_path)
    eig = read_eigenvalues(eig_path)
    kpts = read_kpoints_nnkp(nnkp_path)
    mp_grid = read_mp_grid_win(win_path)
    wins = parse_win_windows(win_path)
    real_lat, recip_lat = read_nnkp_lattice(nnkp_path)
    orbitals = read_nnkp_projections(nnkp_path, real_lat)

    if (nb_m, nk_m) != (num_bands, num_kpts):
        raise ValueError("AMN/MMN size mismatch")
    if eig.shape != (num_kpts, num_bands):
        raise ValueError(f"eig shape {eig.shape} != ({num_kpts},{num_bands})")
    if len(orbitals) != num_wann:
        raise ValueError(f"num_wann={num_wann} but nnkp has {len(orbitals)} projections")
    if len(weights) != nntot:
        raise ValueError("nntot / weights mismatch")

    amn_list = [torch.from_numpy(A[ik].copy()) for ik in range(num_kpts)]
    return WannierDataset(
        amn=amn_list,
        mmn=mmn,
        neighbors=neighbors,
        weights=weights,
        eig=eig,
        kpts_crystal=kpts,
        orbitals=orbitals,
        num_wann=num_wann,
        num_bands=num_bands,
        mp_grid=mp_grid,
        dis_win_max=wins["dis_win_max"],
        dis_win_min=wins["dis_win_min"],
        dis_froz_max=wins["dis_froz_max"],
        dis_froz_min=wins["dis_froz_min"],
        real_lattice_ang=real_lat,
        recip_lattice_ang=recip_lat,
    )
