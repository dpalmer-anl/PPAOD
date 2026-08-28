"""
Slater-Koster two-centre transforms for sp3 (s, p_x, p_y, p_z).

Formulas follow J.C. Slater & G.F. Koster (1954) as tabulated on
https://en.wikipedia.org/wiki/Tight_binding#Table_of_interatomic_matrix_elements

Direction cosines (l, m, n) of the interatomic vector from atom i → atom j:

    r⃗_{i→j} = d (l, m, n),   l² + m² + n² = 1

Bond integrals (functions of distance d only):

    V_ssσ, V_spσ, V_ppσ, V_ppπ

Wikipedia prototypes (cyclic permutations for y,z filled in below):

    E_{s,s}  = V_ssσ
    E_{s,x}  = l V_spσ
    E_{x,x}  = l² V_ppσ + (1 − l²) V_ppπ
    E_{x,y}  = lm V_ppσ − lm V_ppπ
    E_{x,z}  = ln V_ppσ − ln V_ppπ

Parity of p orbitals gives (same (l,m,n) from i→j):

    E_{x,s}  = −l V_spσ
    E_{y,s}  = −m V_spσ
    E_{z,s}  = −n V_spσ
"""

from __future__ import annotations

import numpy as np
import torch


def direction_cosines(vectors: torch.Tensor) -> torch.Tensor:
    """Unit direction cosines (l, m, n) for bond vectors i→j. Shape (n_pairs, 3)."""
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp(min=1e-12)
    return vectors / norms


def sk_block_sp3(
    l: torch.Tensor,
    m: torch.Tensor,
    n: torch.Tensor,
    v_ss_sigma: torch.Tensor,
    v_sp_sigma: torch.Tensor,
    v_pp_sigma: torch.Tensor,
    v_pp_pi: torch.Tensor,
    *,
    hermitian: bool = False,
) -> torch.Tensor:
    """
    Build 4×4 SK blocks for orbital order (s, p_x, p_y, p_z).

    All angular factors match the Wikipedia / Slater–Koster table.

    If ``hermitian=True`` (onsite neighbor sums), p–s uses the same sign as
    s–p so each contribution is Hermitian::

        E_{s,x} = E_{x,s} = l V_spσ

    Otherwise (hopping) E_{x,s} = −l V_spσ as in the two-centre table.
    """
    n_pairs = l.shape[0]
    dtype = l.dtype
    device = l.device

    l2 = l * l
    m2 = m * m
    n2 = n * n
    lm = l * m
    ln = l * n
    mn = m * n

    blocks = torch.zeros((n_pairs, 4, 4), dtype=dtype, device=device)

    # --- s–s ---
    # E_{s,s} = V_ssσ
    blocks[:, 0, 0] = v_ss_sigma

    # --- s–p and p–s ---
    # E_{s,x} = l V_spσ,  E_{s,y} = m V_spσ,  E_{s,z} = n V_spσ
    blocks[:, 0, 1] = l * v_sp_sigma
    blocks[:, 0, 2] = m * v_sp_sigma
    blocks[:, 0, 3] = n * v_sp_sigma
    # Hopping: E_{x,s} = −l V_spσ. Onsite Hermitian sum: E_{x,s} = +l V_spσ.
    sp_sign = 1.0 if hermitian else -1.0
    blocks[:, 1, 0] = sp_sign * l * v_sp_sigma
    blocks[:, 2, 0] = sp_sign * m * v_sp_sigma
    blocks[:, 3, 0] = sp_sign * n * v_sp_sigma

    # --- p–p ---
    # E_{x,x} = l² V_ppσ + (1−l²) V_ppπ  (and cyclic for y,y / z,z)
    blocks[:, 1, 1] = l2 * v_pp_sigma + (1.0 - l2) * v_pp_pi
    blocks[:, 2, 2] = m2 * v_pp_sigma + (1.0 - m2) * v_pp_pi
    blocks[:, 3, 3] = n2 * v_pp_sigma + (1.0 - n2) * v_pp_pi

    # E_{x,y} = lm V_ppσ − lm V_ppπ = lm (V_ppσ − V_ppπ)  (symmetric)
    d_xy = lm * (v_pp_sigma - v_pp_pi)
    d_xz = ln * (v_pp_sigma - v_pp_pi)
    d_yz = mn * (v_pp_sigma - v_pp_pi)
    blocks[:, 1, 2] = d_xy
    blocks[:, 2, 1] = d_xy
    blocks[:, 1, 3] = d_xz
    blocks[:, 3, 1] = d_xz
    blocks[:, 2, 3] = d_yz
    blocks[:, 3, 2] = d_yz

    return blocks


def sk_block_from_integrals(
    bond_vectors: torch.Tensor,
    integrals: dict[str, torch.Tensor],
    *,
    hermitian: bool = False,
) -> torch.Tensor:
    """
    Build SK blocks from bond vectors i→j and named integrals.

    Accepted keys (aliases): ss / ss_sigma, sp / sp_sigma,
    pp_sigma, pp_pi — each shape (n_pairs,).

    ``hermitian=True`` for onsite neighbor sums (see ``sk_block_sp3``).
    """
    l, m, n = direction_cosines(bond_vectors).unbind(dim=-1)
    v_ss = integrals.get("ss_sigma", integrals.get("ss"))
    v_sp = integrals.get("sp_sigma", integrals.get("sp"))
    v_pps = integrals["pp_sigma"]
    v_ppp = integrals["pp_pi"]
    if v_ss is None or v_sp is None:
        raise KeyError("integrals must include ss/ss_sigma and sp/sp_sigma")
    return sk_block_sp3(
        l, m, n, v_ss, v_sp, v_pps, v_ppp, hermitian=hermitian
    )


def sk_integrals_from_block(
    block: np.ndarray, bond_vector: np.ndarray
) -> np.ndarray | None:
    """
    Invert a 4×4 AO block to [V_ssσ, V_spσ, V_ppσ, V_ppπ] (Wikipedia convention).

    Returns None for near-zero bond vectors (onsite).
    """
    block = np.asarray(block, dtype=np.float64)
    bond = np.asarray(bond_vector, dtype=np.float64)
    dist = float(np.linalg.norm(bond))
    if dist < 1e-8:
        return None
    l, m, n = bond / dist
    nhat = np.array([l, m, n], dtype=np.float64)

    # E_{s,s} = V_ssσ
    v_ss = float(block[0, 0])

    # E_{s,α} = α̂ V_spσ  and  E_{α,s} = −α̂ V_spσ
    sp_from_s_row = float(l * block[0, 1] + m * block[0, 2] + n * block[0, 3])
    sp_from_p_col = float(-(l * block[1, 0] + m * block[2, 0] + n * block[3, 0]))
    v_sp = 0.5 * (sp_from_s_row + sp_from_p_col)

    # E_{αβ} = δ_{αβ} V_ppπ + α̂_α α̂_β (V_ppσ − V_ppπ)
    # ⇒ V_ppσ = n̂ᵀ H_pp n̂,  V_ppπ = (Tr H_pp − V_ppσ) / 2
    hpp = block[1:4, 1:4]
    v_pp_sigma = float(nhat @ hpp @ nhat)
    v_pp_pi = 0.5 * (float(np.trace(hpp)) - v_pp_sigma)
    return np.array([v_ss, v_sp, v_pp_sigma, v_pp_pi], dtype=np.float64)


def wikipedia_sk_element(
    orb_i: str,
    orb_j: str,
    l: float,
    m: float,
    n: float,
    *,
    v_ss_sigma: float,
    v_sp_sigma: float,
    v_pp_sigma: float,
    v_pp_pi: float,
) -> float:
    """
    Scalar SK matrix element E_{orb_i, orb_j} from the Wikipedia table
    (plus standard cyclic / parity completions for the full sp set).
    """
    orb_i = orb_i.lower()
    orb_j = orb_j.lower()
    cos = {"x": l, "y": m, "z": n}

    if orb_i == "s" and orb_j == "s":
        return v_ss_sigma
    if orb_i == "s" and orb_j in cos:
        return cos[orb_j] * v_sp_sigma
    if orb_j == "s" and orb_i in cos:
        return -cos[orb_i] * v_sp_sigma
    if orb_i in cos and orb_j in cos:
        ci, cj = cos[orb_i], cos[orb_j]
        if orb_i == orb_j:
            # E_{x,x} = l² V_ppσ + (1−l²) V_ppπ
            return ci * ci * v_pp_sigma + (1.0 - ci * ci) * v_pp_pi
        # E_{x,y} = lm V_ppσ − lm V_ppπ
        return ci * cj * (v_pp_sigma - v_pp_pi)
    raise ValueError(f"Unsupported orbital pair ({orb_i}, {orb_j})")
