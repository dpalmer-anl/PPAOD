"""Real spherical harmonics for Wannier90 ``(l, mr)`` trial orbitals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def real_spherical_harmonic(
    l: int,
    mr: int,
    q_hat: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Real spherical harmonic ``Y_lm`` in Wannier90 ``mr`` convention.

    Parameters
    ----------
    l, mr :
        Angular momentum and Wannier90 ``mr`` index (1-based within ``l``).
    q_hat : (..., 3)
        Unit vectors (Cartesian). Zero vectors → Y=0 for l>0, Y_00 for l=0.

    Returns
    -------
    Y : (...) float
    """
    q_hat = np.asarray(q_hat, dtype=np.float64)
    x = q_hat[..., 0]
    y = q_hat[..., 1]
    z = q_hat[..., 2]
    # safe norms already unit, but guard zeros
    r = np.sqrt(x * x + y * y + z * z)
    # l = 0
    if l == 0:
        if mr != 1:
            raise ValueError(f"l=0 requires mr=1, got {mr}")
        return np.full(q_hat.shape[:-1], 1.0 / np.sqrt(4.0 * np.pi), dtype=np.float64)
    # direction cosines; at q=0 set to 0 (FT weight vanishes with radial anyway)
    with np.errstate(invalid="ignore", divide="ignore"):
        xh = np.where(r > 0, x / r, 0.0)
        yh = np.where(r > 0, y / r, 0.0)
        zh = np.where(r > 0, z / r, 0.0)

    if l == 1:
        pref = np.sqrt(3.0 / (4.0 * np.pi))
        # W90: mr=1→pz, mr=2→px, mr=3→py
        if mr == 1:
            return pref * zh
        if mr == 2:
            return pref * xh
        if mr == 3:
            return pref * yh
        raise ValueError(f"l=1 requires mr in 1..3, got {mr}")

    if l == 2:
        pref = np.sqrt(5.0 / (4.0 * np.pi))
        # W90 mr=1..5: dz2, dxz, dyz, dx2-y2, dxy
        if mr == 1:
            return pref * (3.0 * zh * zh - 1.0) / 2.0  # ~ Y20
        if mr == 2:
            return np.sqrt(15.0 / (4.0 * np.pi)) * xh * zh
        if mr == 3:
            return np.sqrt(15.0 / (4.0 * np.pi)) * yh * zh
        if mr == 4:
            return np.sqrt(15.0 / (16.0 * np.pi)) * (xh * xh - yh * yh)
        if mr == 5:
            return np.sqrt(15.0 / (4.0 * np.pi)) * xh * yh
        raise ValueError(f"l=2 requires mr in 1..5, got {mr}")

    raise NotImplementedError(f"Real Y_lm not implemented for l={l}")
