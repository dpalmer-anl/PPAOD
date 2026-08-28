from __future__ import annotations

import numpy as np

from ppaod.band_io import parse_qe_bands_out
from ppaod.wannier_io import (
    fourier_transform,
    inverse_fourier_transform,
    mp_R_vectors,
)


def test_qe_band_parser(tmp_path):
    bands = tmp_path / "bands.out"
    bands.write_text(
        """
 End of band structure calculation
     k = 0.0 0.0 0.0
                 bands (ev):
       -2.0  1.5

     k = 0.5 0.0 0.0
                 bands (ev):
       -1.0  2.0

""",
        encoding="utf-8",
    )
    np.testing.assert_allclose(
        parse_qe_bands_out(bands),
        [[-2.0, 1.5], [-1.0, 2.0]],
    )


def test_fourier_round_trip():
    kpts = np.asarray(
        [[i / 2, j / 2, 0.0] for i in range(2) for j in range(2)],
        dtype=float,
    )
    r_frac = mp_R_vectors(2, 2, 1)
    rng = np.random.default_rng(7)
    h_k = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
    h_r = fourier_transform(h_k, kpts, r_frac)
    np.testing.assert_allclose(
        inverse_fourier_transform(h_r, kpts, r_frac),
        h_k,
        atol=1e-12,
    )
