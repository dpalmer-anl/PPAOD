"""
Spherical-Bessel radial basis and PW × Bessel integral table ``T_l[k]``.

Radial expansion (particle-in-a-sphere BC)::

    R_l(r) = Σ_i c_{i,l} j_l(k_i r) ,   j_l(k_i r_c) = 0

PW transform kernel (independent of ``c_ml``)::

    T_l[k][i, G] = ∫_0^{r_c} j_l(k_i r) j_l(q_G r) r^2 dr
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import spherical_jn


def spherical_bessel_roots(l: int, n_basis: int, r_c: float) -> NDArray[np.float64]:
    """
    First ``n_basis`` positive roots ``k_i`` of ``j_l(k r_c) = 0``.

    Uses bracketed Brent search on successive lobes of ``j_l``.
    """
    if r_c <= 0:
        raise ValueError("r_c must be positive")
    roots: list[float] = []
    # j_l(x)=0 at x = k r_c; search in x > 0
    # Asymptotic zeros near π(n + l/2)
    x = 1e-8
    dx = 0.25
    f_prev = float(spherical_jn(l, x))
    while len(roots) < n_basis:
        x_next = x + dx
        f_next = float(spherical_jn(l, x_next))
        if f_prev == 0.0:
            # exact hit
            if x / r_c > 1e-12:
                roots.append(x / r_c)
            f_prev = f_next
            x = x_next
            continue
        if f_prev * f_next < 0.0:
            x_root = brentq(lambda t: float(spherical_jn(l, t)), x, x_next, xtol=1e-12)
            k = x_root / r_c
            if k > 1e-12 and (not roots or abs(k - roots[-1]) > 1e-10):
                roots.append(k)
            f_prev = f_next
            x = x_next
        else:
            f_prev = f_next
            x = x_next
        if x > np.pi * (n_basis + l + 20):
            # widen step if stuck
            dx = min(dx * 1.5, 1.0)
        if x > 1e5:
            raise RuntimeError(f"Failed to find {n_basis} roots for l={l}")
    return np.asarray(roots[:n_basis], dtype=np.float64)


def bessel_integral_closed(
    l: int,
    k_i: float,
    q: float,
    r_c: float,
    *,
    equal_tol: float = 1e-8,
) -> float:
    """
    Closed-form ``∫_0^{r_c} j_l(k_i r) j_l(q r) r^2 dr`` with ``j_l(k_i r_c)=0``.

    Lommel integral; equal-argument limit when ``|k_i - q|`` is tiny.
    """
    ki = float(k_i)
    qq = float(q)
    a = float(r_c)
    if abs(ki - qq) < equal_tol * max(1.0, abs(ki)):
        # ∫_0^a [j_l(ki r)]^2 r^2 dr = (a^3 / 2) [j_{l+1}(ki a)]^2
        # (since j_l(ki a)=0)
        return 0.5 * (a**3) * float(spherical_jn(l + 1, ki * a)) ** 2
    # Lommel: ∫_0^a r² j_l(ki r) j_l(q r) dr
    #   = a²/(q²-ki²) * (ki j_{l+1}(ki a) j_l(q a) - q j_{l+1}(q a) j_l(ki a))
    # with j_l(ki a)=0 → a² ki / (q² - ki²) * j_{l+1}(ki a) j_l(q a)
    #                  = a² ki / (ki² - q²) * j_{l+1}(ki a) j_l(q a) * (-1)
    # Verified vs trapezoid: use + a² ki / (ki² - q²) * … with scipy spherical_jn
    # (sign convention matches scipy j_{l+1}(k_i a) at particle-in-a-sphere roots).
    j_l_q = float(spherical_jn(l, qq * a))
    j_lp1_ki = float(spherical_jn(l + 1, ki * a))
    return (a**2) * ki / (ki * ki - qq * qq) * j_lp1_ki * j_l_q


def bessel_integral_quad(
    l: int,
    k_i: float,
    q: float,
    r_c: float,
    n_grid: int = 4000,
) -> float:
    """Trapezoidal reference for unit tests."""
    r = np.linspace(0.0, r_c, n_grid)
    f = spherical_jn(l, k_i * r) * spherical_jn(l, q * r) * (r**2)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(f, r))


@dataclass
class BesselBasis:
    """Per-``l`` radial Bessel basis."""

    l: int
    r_c: float
    k_nodes: NDArray[np.float64]  # (n_basis,)

    @property
    def n_basis(self) -> int:
        return int(self.k_nodes.shape[0])


def make_bases(
    l_values: list[int],
    r_c: float | dict[int, float],
    n_basis: int | dict[int, int],
) -> dict[int, BesselBasis]:
    """Build ``BesselBasis`` for each unique ``l``."""
    bases: dict[int, BesselBasis] = {}
    for l in sorted(set(l_values)):
        rc = float(r_c[l] if isinstance(r_c, dict) else r_c)
        nb = int(n_basis[l] if isinstance(n_basis, dict) else n_basis)
        nodes = spherical_bessel_roots(l, nb, rc)
        bases[l] = BesselBasis(l=l, r_c=rc, k_nodes=nodes)
    return bases


def precompute_T_l(
    basis: BesselBasis,
    q_abs: NDArray[np.float64],
) -> torch.Tensor:
    """
    ``T_l[i, G]`` for one k-point / one ``l``.

    Parameters
    ----------
    q_abs : (nG,) ``|k+G|`` in Bohr⁻¹
    """
    n_basis = basis.n_basis
    nG = q_abs.shape[0]
    T = np.empty((n_basis, nG), dtype=np.float64)
    for i, ki in enumerate(basis.k_nodes):
        for g in range(nG):
            T[i, g] = bessel_integral_closed(basis.l, float(ki), float(q_abs[g]), basis.r_c)
    return torch.from_numpy(T)


def precompute_T_l_vectorized(
    basis: BesselBasis,
    q_abs: NDArray[np.float64],
) -> torch.Tensor:
    """Vectorized over G for each basis index (much faster than Python double loop)."""
    q = np.asarray(q_abs, dtype=np.float64)
    a = basis.r_c
    l = basis.l
    rows = []
    for ki in basis.k_nodes:
        ki = float(ki)
        equal = np.abs(q - ki) < 1e-8 * max(1.0, abs(ki))
        j_lp1_ki = float(spherical_jn(l + 1, ki * a))
        # equal branch
        T_eq = 0.5 * (a**3) * (j_lp1_ki**2)
        # unequal
        j_l_q = spherical_jn(l, q * a)
        T_ne = (a**2) * ki / (ki * ki - q * q) * j_lp1_ki * j_l_q
        T_row = np.where(equal, T_eq, T_ne)
        # q → 0 and l>0 is fine; for l=0, j_0(0)=1
        rows.append(T_row)
    return torch.from_numpy(np.stack(rows, axis=0).astype(np.float64))
