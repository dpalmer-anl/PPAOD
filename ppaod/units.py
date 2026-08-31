"""Unit conversions used at file-format boundaries.

PPAOD uses electron-volts for energies and Angstroms for real-space
quantities. Quantum ESPRESSO wavefunction files expose ``alat`` in Bohr and
reciprocal coordinates in units of ``2π/alat``; those values are converted as
they are loaded.
"""

from __future__ import annotations

BOHR_TO_ANGSTROM = 0.52917720859
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
