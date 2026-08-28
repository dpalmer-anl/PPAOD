# PPAOD methodology

PPAOD means **Parameterized Pseudo-Atomic Orbital Disentanglement**. Its
purpose is to construct a compact subspace from DFT Bloch eigenvectors that
has three properties:

1. it is made from physically interpretable atomic-orbital projections;
2. it is smooth and localized across the Brillouin zone; and
3. it represents the target bands and their orbital character well.

The method is intended for entangled band manifolds, where selecting a fixed
number of bands by energy alone can produce discontinuous or chemically
unhelpful subspaces. PPAOD instead selects a subspace inside the available DFT
eigenvectors at each k-point and optimizes that selection globally.

## Minimal pseudo-atomic basis

The target basis is fixed by the supplied Wannier90 projections. For a
carbon sp3 calculation this is one `s` and three `p` orbitals per atom. PPAOD
does not enlarge this basis with extra polarization functions, additional
radial zeta shells, or unrelated high-energy orbitals.

The radial part of each projected orbital is parameterized with a finite
spherical-Bessel basis inside a cutoff \(r_c\):

\[
 R_l(r) = \sum_q \theta_{lq} j_l(k_{lq} r),
\]

with the angular dependence represented by real spherical harmonics
\(Y_{lm}(\hat{\mathbf r})\). The coefficients \(\theta_{lq}\) are the
parameters optimized by PPAOD. This provides radial flexibility while keeping
the number and chemical meaning of the orbitals fixed.

For each k-point, projection onto the parameterized atomic orbitals gives a
matrix \(V(k)\) whose columns span the candidate target subspace:

\[
 H_{\mathrm{sub}}(k)
 = V^\dagger(k)\,E_{\mathrm{DFT}}(k)\,V(k).
\]

The columns are constrained to be orthonormal. The resulting \(J\)-dimensional
subspace is therefore represented using exactly the requested \(J\) orbitals,
even when the outer DFT window contains more than \(J\) bands.

## Band windows and initial fit

PPAOD starts from the Wannier90 outer disentanglement window. Bands outside
that window are excluded. A projectability filter can additionally remove
DFT states whose total overlap with the trial orbitals is below
`--proj-threshold`; at least \(J\) states must remain at every k-point.

Before nonlinear optimization, the radial coefficients are fit by ridge
least squares to the supplied `.amn` projections. This gives a physically
meaningful initial point and makes the optimization less sensitive to a
random starting basis. The initial AMN residual is reported as
`amn_fit_rms`.

This AMN fit is a projection fit, not an independent least-squares fit to
individual eigenvalues. The projected Hamiltonian inherits DFT eigenvalues
through \(H_{\mathrm{sub}}=V^\dagger E_{\mathrm{DFT}}V\).

## Smoothness term: \(\Omega_I\)

The gauge-invariant spread term \(\Omega_I\) measures how smoothly the chosen
subspace changes between neighboring k-points. It is evaluated from the
Wannier90 `.mmn` overlap matrices:

\[
 M^{(k,b)}_{\mathrm{sub}}
 = V^\dagger(k)\,M^{(k,b)}\,V(k+b).
\]

The loss penalizes loss of overlap between neighboring subspaces. A low
\(\Omega_I\) favors a smooth, localized real-space representation and avoids
arbitrary k-point-by-k-point changes in the selected DFT states.

PPAOD reports the spread per Wannier function in the combined objective,
\(\Omega_I/J\), so that this term remains comparable when the number of
orbitals changes.

## Orbital-character term: \(\Omega_D\)

The \(\Omega_D\) term compares local/orbital-projected densities of states at
each k-point. For DFT, the LPDOS for orbital \(i\) is

\[
 \rho^{\mathrm{DFT}}_{i k}(E)
 = \sum_m |V_{mi}(k)|^2\,
   \delta_\sigma(E-\epsilon^{\mathrm{DFT}}_{mk}).
\]

After diagonalizing the projected Hamiltonian,
\[
 H_{\mathrm{sub}}(k)C(k)
 = C(k)\,\mathrm{diag}(\epsilon^{\mathrm{sub}}_{nk}),
\]
the subspace LPDOS is

\[
 \rho^{\mathrm{sub}}_{i k}(E)
 = \sum_n |C_{in}(k)|^2\,
   \delta_\sigma(E-\epsilon^{\mathrm{sub}}_{nk}).
\]

PPAOD uses Gaussian broadening and normalizes each orbital LPDOS to a unit
probability density. It then evaluates

\[
 \Omega_D =
 \frac{1}{N_kJ}
 \sum_{k,i}
 KL\left(
 \rho^{\mathrm{sub}}_{ik}
 \,\middle\|\,
 \rho^{\mathrm{DFT}}_{ik}
 \right).
\]

This term is sensitive to both band-energy placement and orbital character.
It discourages a subspace that is smooth but represents the wrong DFT states.

## Combined optimization objective

The optimized objective is

\[
 \Omega_P =
 \alpha\frac{\Omega_I}{J}
 +(1-\alpha)\Omega_D,
\]

where `--alpha` controls the tradeoff. Larger \(\alpha\) emphasizes smooth,
localized orbitals; smaller \(\alpha\) emphasizes matching the DFT orbital
spectral character. The default is `--alpha 0.8`.

The `--l2` option adds a small regularization penalty on the radial
parameters. It controls numerical stability and coefficient size; it does
not add orbitals to the basis.

## What “band fitting” means here

PPAOD does not fit a separate free tight-binding eigenvalue model. The band
energies of the projected Hamiltonian are generated directly from the DFT
eigenvalues through the subspace projection. Band fidelity is encouraged
during optimization through the energy-resolved \(\Omega_D\) term and the
initial `.amn` projection fit.

After optimization, PPAOD Fourier transforms \(H_{\mathrm{sub}}(k)\) to
\(H(R)\). The output band-path plot compares the reconstructed
\(H(k)\rightarrow H(R)\rightarrow H(k)\) bands with `bands.out`. This is a
validation diagnostic rather than an additional optimization term.

## Minimal-basis interpretation

The goal is not to reproduce every DFT band using an increasingly large
basis. The goal is to identify the smallest chemically specified
pseudo-atomic subspace that captures the target manifold cleanly. If the
target cannot be represented accurately with the chosen minimal basis, the
result should expose that limitation through the AMN residual, \(\Omega_D\),
interlacing warnings, and band-path comparison rather than silently adding
polarization or extra-zeta functions.
