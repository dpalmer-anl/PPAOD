"""Small band-structure readers used by the PPAOD output routines."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .wannier_io import inverse_fourier_transform


def parse_qe_bands_out(path: Path) -> np.ndarray | None:
    """Parse Quantum ESPRESSO ``bands.out`` into ``(nk, nbnd)`` eV."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if "End of band structure calculation" in text:
        text = text.split("End of band structure calculation", 1)[1]
    blocks = re.split(r"\n\s*k =", text)
    eigenvalues: list[list[float]] = []
    for block in blocks[1:]:
        if "bands (ev)" not in block.lower():
            continue
        body = re.split(
            r"bands \(ev\):", block, maxsplit=1, flags=re.IGNORECASE
        )[1]
        values: list[float] = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                if values:
                    break
                continue
            if re.search(r"[a-df-zA-DF-Z]", line):
                break
            values.extend(map(float, line.split()))
        if values:
            eigenvalues.append(values)
    if not eigenvalues:
        return None
    num_bands = min(len(values) for values in eigenvalues)
    return np.asarray(
        [values[:num_bands] for values in eigenvalues],
        dtype=float,
    )


def compute_hr_bands(
    H_R: np.ndarray,
    R_frac: np.ndarray,
    kpts_frac: np.ndarray,
) -> np.ndarray:
    """Bloch-sum ``H(R)`` and return sorted eigenvalues along a k-path."""
    H_k = inverse_fourier_transform(H_R, kpts_frac, R_frac)
    bands = np.empty((H_k.shape[0], H_k.shape[1]), dtype=float)
    for ik, H in enumerate(H_k):
        bands[ik] = np.linalg.eigvalsh(0.5 * (H + H.conj().T)).real
    return bands
