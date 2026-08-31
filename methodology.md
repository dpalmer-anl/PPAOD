# PPAOD methodology

PPAOD means **Parameterized Pseudo-Atomic Orbital Disentanglement**. Its
purpose is to construct a compact subspace from DFT Bloch eigenvectors that
has three properties:

1. it is made from physically interpretable atomic-orbital projections;
2. it is smooth across the Brillouin zone (in terms of wavefunction overlap) and disentangled from other bands; and
3. it represents the target DFT bands and their orbital character well.

Energies are expressed in eV and real-space lengths in Å (Angstroms);
reciprocal-space vectors use Å⁻¹.

The method is intended to be used for creating accurate tight binding models for entangled band manifolds, where selecting a fixed
number of bands by energy alone can produce discontinuous or chemically
unhelpful subspaces.

**Notation used throughout.** $J$ is the fixed number of target Wannier
functions / trial orbitals. $N(k)$ is the number of DFT bands retained at
$k$-point $k$ after the outer-window and projectability filters (see
below); it varies with $k$ and satisfies $N(k)\ge J$.

## Minimal pseudo-atomic orbital basis

The angular momentum for the basis is fixed by the supplied Wannier90 projections from the .win file. For a carbon sp3 calculation this is one `s` and three `p` orbitals per atom. PPAOD does not enlarge this basis with extra polarization functions, additional
radial zeta shells, or unrelated high-energy orbitals, and instead optimizes the radial part of each orbital.

The radial part of each projected orbital is parameterized with a finite
spherical-Bessel basis inside a cutoff $r_c$ (in Å):

$$
 R_l(r) = \sum_q \theta_{lq}\, j_l(k_{lq} r), \qquad r \le r_c,
$$

with $R_l(r)\equiv 0$ for $r>r_c$, and the radial basis nodes $k_{lq}$ fixed
in advance as the roots of $j_l(k_{lq} r_c)=0$. The angular dependence is
represented by real spherical harmonics $Y_{lm}(\hat{\mathbf r})$, so a
complete trial orbital $n$ (associated with a given atomic site
$\boldsymbol\tau_n$ and angular channel $l,m$) is

$$
 \phi_n(\theta,\mathbf r) = R_l(\theta, |\mathbf r - \boldsymbol\tau_n|)\, Y_{lm}(\widehat{\mathbf r - \boldsymbol\tau_n}).
$$

The coefficients $\theta_{lq}$ are the parameters optimized by PPAOD. This
provides radial flexibility while keeping the angular momentum of each
orbital fixed.

**Projection onto DFT bands.** Using the DFT plane-wave coefficients
$c_{mk}(\mathbf G)$ of band $m$ at $k$, and the Fourier transform
$\tilde\phi_n(\theta,\mathbf q)$ of the trial orbital, the projection of the
$N(k)$ retained DFT bands onto the $J$ trial orbitals is

$$
 A(\theta,k)_{mn} = \langle u_{mk}|\phi_n(\theta)\rangle
 = \sum_{\mathbf G} c^{*}_{mk}(\mathbf G)\,\tilde\phi_n(\theta,\,k+\mathbf G),
 \qquad A(\theta,k)\in\mathbb C^{N(k)\times J}.
$$

$A(\theta,k)$ is **not** itself orthonormal — its columns are the raw
overlaps of each DFT band with each trial orbital, and generally overlap
with one another. The candidate target subspace is instead spanned by the
Löwdin-symmetrically-orthonormalized matrix

$$
 V(k) = A(\theta,k)\,\big[A(\theta,k)^\dagger A(\theta,k)\big]^{-1/2},
 \qquad V(k)\in\mathbb C^{N(k)\times J}, \qquad V(k)^\dagger V(k) = I_J.
$$

The projected (disentangled) Hamiltonian is then

$$
 H_{\mathrm{sub}}(k) = V^\dagger(k)\,E_{\mathrm{DFT}}(k)\,V(k),
 \qquad H_{\mathrm{sub}}(k)\in\mathbb C^{J\times J},
$$

where $E_{\mathrm{DFT}}(k) = \mathrm{diag}(\epsilon^{\mathrm{DFT}}_{1k},\dots,\epsilon^{\mathrm{DFT}}_{N(k)k})$
is the $N(k)\times N(k)$ diagonal matrix of retained DFT eigenvalues at $k$.
Because $V(k)^\dagger V(k)=I_J$, the columns of $V(k)$ define an orthonormal
$J$-dimensional basis, so the resulting subspace is represented using
exactly the requested $J$ orbitals, even when the outer DFT window (after
filtering) contains more than $J$ bands.

## Band windows and initial fit

PPAOD starts from the Wannier90 outer disentanglement window. Bands outside
that window are excluded. In order to prevent DFT bands with low projectability delocalizing the atomic orbitals a projectability filter is applied. The projectability filter removes DFT states whose total overlap with the trial orbitals is below
`--proj-threshold`; at least $J$ states must remain at every $k$-point.
$N(k)$ in all equations above and below refers to the number of bands
remaining **after** both the window and the projectability filter are
applied.

Before nonlinear optimization, the radial coefficients are fit by ridge
least squares to the supplied `.amn` projections, $A_{w90}(k)$:

$$
 \theta_0 = \arg\min_\theta \; \sum_k \big\|A(\theta,k) - A_{w90}(k)\big\|_F^2 \;+\; \lambda_0\|\theta\|_2^2 ,
$$

where $\lambda_0$ is a ridge regularization strength. This gives a physically meaningful initial point and makes the subsequent nonlinear optimization less sensitive to a random starting basis. 

## Smoothness term: $\Omega_I$

The gauge-invariant disentanglement term $\Omega_I$ measures how smoothly the chosen
subspace changes between neighboring $k$-points. It is evaluated from the
Wannier90 `.mmn` overlap matrices between DFT bands at neighboring
$k$-points,

$$
 M^{(k,b)}_{mn} = \langle u_{mk}|u_{n,k+b}\rangle, \qquad M^{(k,b)}\in\mathbb C^{N(k)\times N(k+b)},
$$

projected into the disentangled subspace at each end,

$$
 M^{(k,b)}_{\mathrm{sub}} = V^\dagger(k)\,M^{(k,b)}\,V(k+b), \qquad M^{(k,b)}_{\mathrm{sub}} \in \mathbb C^{J\times J}.
$$

 The smoothness loss itself is

$$
\Omega_I = \tfrac{1}{N_k}\sum_{k,b}w_b\big[J-\mathrm{Tr}(P(k)M^{(k,b)}P(k+b)M^{(k,b)\dagger})\big],
$$

where $b$ runs over the finite-difference neighbor shell of $k$ on the
Monkhorst–Pack mesh and $w_b$ are the corresponding finite-difference
weights and $P(k)=V(k)V(k)^\dagger$. $\Omega_I=0$ exactly when the subspace is perfectly aligned between $k$ and every neighbor $k+b$, and $\Omega_I>0$ otherwise. 

A low $\Omega_I$ favors a smooth, localized real-space representation and
avoids arbitrary $k$-point-by-$k$-point changes in the selected DFT states.

PPAOD reports the spread per Wannier function in the combined objective,
$\Omega_I/J$, so that this term remains comparable when the number of
orbitals changes.

## Orbital-character term: $\Omega_D$

The $\Omega_D$ term compares local/orbital-projected densities of states at
each $k$-point. For DFT, the LPDOS for orbital $i$ is

$$
 \rho^{\mathrm{DFT}}_{ik}(E)
 = \sum_{m=1}^{N(k)} |V_{mi}(k)|^2\,
   \delta_\sigma(E-\epsilon^{\mathrm{DFT}}_{mk}).
$$

Because $V(k)^\dagger V(k)=I_J$, each column of $V(k)$ has unit norm,
$\sum_m |V_{mi}(k)|^2=1$, so $\rho^{\mathrm{DFT}}_{ik}(E)$ automatically
integrates to 1 over $E$.

After diagonalizing the projected Hamiltonian,
$$
 H_{\mathrm{sub}}(k)\,C(k)
 = C(k)\,\mathrm{diag}(\epsilon^{\mathrm{sub}}_{nk}),
 \qquad C(k)\in\mathbb C^{J\times J} \text{ unitary},
$$
the subspace LPDOS is

$$
 \rho^{\mathrm{sub}}_{ik}(E)
 = \sum_{n=1}^{J} |C_{in}(k)|^2\,
   \delta_\sigma(E-\epsilon^{\mathrm{sub}}_{nk}).
$$

Since $C(k)$ is unitary, its rows are also orthonormal,
$\sum_n |C_{in}(k)|^2=1$, so $\rho^{\mathrm{sub}}_{ik}(E)$ is likewise
automatically normalized to unit probability. The KL divergence is then used to measure the difference between the two LPDOS distributions. The KL divergence between the two
normalized LPDOS curves for orbital $i$ at $k$ is

$$
 KL\!\left(\rho^{\mathrm{sub}}_{ik}\,\middle\|\,\rho^{\mathrm{DFT}}_{ik}\right)
 = \int dE\; \rho^{\mathrm{sub}}_{ik}(E)\, \ln\!\left[\frac{\rho^{\mathrm{sub}}_{ik}(E)}{\rho^{\mathrm{DFT}}_{ik}(E)}\right],
$$

and PPAOD evaluates

$$
 \Omega_D(\theta) =
 \frac{1}{N_kJ}
 \sum_{k,i}
 KL\left(
 \rho^{\mathrm{sub}}_{ik}
 \,\middle\|\,
 \rho^{\mathrm{DFT}}_{ik}
 \right).
$$

This term measures the the difference between the DFT band and the projected subspace bands while simulataneously ensuring the orbital orderings of the eigenstates match DFT.

## Combined optimization objective

The optimized objective is

$$
 \Omega_P =
 \alpha\frac{\Omega_I}{J}
 +(1-\alpha)\Omega_D,
$$

where `--alpha` controls the tradeoff. Larger $\alpha$ emphasizes smooth,
localized orbitals; smaller $\alpha$ emphasizes matching the DFT orbital
spectral character. The default is `--alpha 0.8`. $\Omega_P$ is
differentiable in $\theta$ end-to-end
($\theta \to \phi_n(\theta) \to A(\theta,k) \to V(k) \to \{H_{\mathrm{sub}}(k), M^{(k,b)}_{\mathrm{sub}}\} \to \Omega_P$)
and is minimized with a gradient-based optimizer (Adam), starting from
$\theta_0$.

The `--l2` option adds a small regularization penalty $\lambda\|\theta\|_2^2$
on the radial parameters during this nonlinear optimization stage. It
controls numerical stability and coefficient size — in particular guarding
against near-linear-dependence in $A(\theta,k)^\dagger A(\theta,k)$, which
would make the Löwdin orthonormalization step ill-conditioned.
