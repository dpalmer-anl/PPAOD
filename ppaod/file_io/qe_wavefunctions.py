"""Read QE plane-wave coefficients ``c[k]`` and G-vectors from ``prefix.save/wfc*.dat``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import FortranFile

# Bohr / Angstrom (CODATA-style; QE uses ~0.529177)
ANG_TO_BOHR = 1.0 / 0.52917720859


def read_alat_bohr(save_dir: Path | str) -> float:
    """Parse ``alat`` (Bohr) from ``data-file-schema.xml`` in a QE ``.save`` directory."""
    import re

    schema = Path(save_dir) / "data-file-schema.xml"
    text = schema.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'alat\s*=\s*"([^"]+)"', text)
    if m is None:
        raise ValueError(f"Could not find alat in {schema}")
    return float(m.group(1))


def read_wfc_dat(
    path: Path | str,
) -> tuple[int, NDArray[np.float64], NDArray[np.int32], NDArray[np.complex128], NDArray[np.float64]]:
    """
    Read one QE ``wfc#.dat`` (Fortran unformatted, QE ≥6 / 7.x layout).

    Returns
    -------
    ik : 1-based k-index as stored by QE
    xk : (3,) crystal-cart. k in units of ``2π/alat``
    mill : (nG, 3) Miller indices
    evc : (nbnd, nG) complex PW coefficients (normalized Σ_G |c|² = 1)
    b_matrix : (3, 3) with rows ``b1,b2,b3`` in units of ``2π/alat``
    """
    path = Path(path)
    with FortranFile(path, "r") as f:
        raw = f.read_record("u1")
        if len(raw) != 44:
            raise ValueError(f"Unexpected wfc header size {len(raw)} in {path}")
        ik = int(np.frombuffer(raw[0:4], "<i4")[0])
        xk = np.frombuffer(raw[4:28], "<f8").copy()
        ints = f.read_record("<i4")
        ngw, igwx, npol, nbnd = (int(x) for x in ints[:4])
        del ngw
        if npol != 1:
            raise NotImplementedError(f"npol={npol} not supported (need spinor handling)")
        bflat = f.read_record("<f8")
        b_matrix = np.asarray(bflat, dtype=np.float64).reshape(3, 3)
        mill = f.read_record("<i4").reshape(3, igwx, order="F").T.astype(np.int32)
        evc = np.empty((nbnd, igwx), dtype=np.complex128)
        for ib in range(nbnd):
            c = f.read_record("<c16")
            if c.shape[0] != igwx:
                raise ValueError(
                    f"{path}: band {ib}: expected igwx={igwx}, got {c.shape[0]}"
                )
            evc[ib] = c
    return ik, xk, mill, evc, b_matrix


def miller_to_G_cart_au(
    mill: NDArray[np.int32],
    b_matrix: NDArray[np.float64],
    alat_bohr: float | None = None,
) -> NDArray[np.float64]:
    """
    Convert Miller indices to Cartesian G (Bohr⁻¹).

    In QE ``wfc*.dat``, the reciprocal axes ``b1,b2,b3`` are written already
    in Cartesian Bohr⁻¹ (consistent with ``2π/|a_i|``), *not* in units of
    ``2π/alat`` that still need rescaling.
    """
    del alat_bohr  # kept for call-site compatibility
    return mill.astype(np.float64) @ b_matrix


def xk_to_k_cart_au(xk: NDArray[np.float64], alat_bohr: float | None = None) -> NDArray[np.float64]:
    """``xk`` from ``wfc*.dat`` → Cartesian ``k`` in Bohr⁻¹ (same units as ``b_i``)."""
    del alat_bohr
    return np.asarray(xk, dtype=np.float64).copy()


def load_all_wavefunctions(
    save_dir: Path | str,
    *,
    n_k: int | None = None,
    k_indices: list[int] | None = None,
) -> tuple[
    list[NDArray[np.complex128]],
    list[NDArray[np.float64]],
    list[NDArray[np.float64]],
    float,
    NDArray[np.float64],
]:
    """
    Load ``wfc*.dat`` from a QE ``.save`` directory.

    Parameters
    ----------
    n_k :
        If ``k_indices`` is None, load ``wfc1…wfc{n_k}`` (or all files).
    k_indices :
        0-based mesh indices to load (only those ``wfc{ik+1}.dat`` files).

    Returns
    -------
    c_list, Gvecs_list, k_cart_list, alat_bohr, b_matrix
    """
    save_dir = Path(save_dir)
    if k_indices is None:
        if n_k is None:
            files = sorted(save_dir.glob("wfc*.dat"), key=lambda p: int(p.stem[3:]))
            k_indices = [int(p.stem[3:]) - 1 for p in files]
        else:
            k_indices = list(range(n_k))

    alat = read_alat_bohr(save_dir)
    c_list: list[NDArray[np.complex128]] = []
    G_list: list[NDArray[np.float64]] = []
    k_list: list[NDArray[np.float64]] = []
    b_matrix: NDArray[np.float64] | None = None
    for ik0 in k_indices:
        ik = int(ik0) + 1
        path = save_dir / f"wfc{ik}.dat"
        if not path.is_file():
            raise FileNotFoundError(path)
        _ik_file, xk, mill, evc, bmat = read_wfc_dat(path)
        if b_matrix is None:
            b_matrix = bmat
        G_cart = miller_to_G_cart_au(mill, bmat, alat)
        k_cart = xk_to_k_cart_au(xk, alat)
        c_list.append(evc)
        G_list.append(G_cart)
        k_list.append(k_cart)
    assert b_matrix is not None
    return c_list, G_list, k_list, alat, b_matrix


def load_wavefunctions(
    save_dir: Path | str,
    k_indices: list[int],
    **kwargs,
) -> tuple[
    list[NDArray[np.complex128]],
    list[NDArray[np.float64]],
    list[NDArray[np.float64]],
    float,
    NDArray[np.float64],
]:
    """Load only selected 0-based k-indices (thin wrapper)."""
    return load_all_wavefunctions(save_dir, k_indices=list(k_indices), **kwargs)
