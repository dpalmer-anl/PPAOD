# PPAOD

Parameterized Pseudo-Atomic Orbital Disentanglement (PPAOD) finds a compact
atomic-orbital subspace inside a set of DFT eigenvectors. The subspace is
optimized to remain smooth and localized across the Brillouin zone while
representing the target bands and their local orbital character. PPAOD uses a
minimal atomic-orbital basis: it does not add polarization functions or extra
zeta shells.

The optimization is described in [methodology.md](methodology.md).

## Installation

PPAOD requires Python 3.10 or newer. Install from a clone:

```bash
git clone https://github.com/dpalmer-anl/PPAOD.git
cd PPAOD
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
pytest
```

`torch` is installed through the normal Python package index by default.
For a CUDA installation, install the matching PyTorch build using the
instructions at https://pytorch.org/ before or after installing PPAOD.

## Required input files

PPAOD operates on a completed Quantum ESPRESSO plus Wannier90 calculation.
For a seed named `carbon_wannier`, place these files in one directory:

```text
qe/
├── carbon_calc.save/
│   ├── data-file-schema.xml
│   └── wfc*.dat
├── carbon_wannier.amn
├── carbon_wannier.eig
├── carbon_wannier.mmn
├── carbon_wannier.nnkp
└── carbon_wannier.win
```

The `.amn`, `.mmn`, `.eig`, `.nnkp`, and `.win` files must describe the same
mesh. The QE `wfc*.dat` files are needed when building a new projection
cache. A compatible `projection_cache.pt` can be supplied with
`--restart-file` to avoid rereading the wavefunctions. PPAOD does not bundle
QE, Wannier90, or machine-specific wavefunction data.

## Quick start: serial

Run on one CPU process:

```bash
ppaod \
  --qe-dir /path/to/qe \
  --seed carbon_wannier \
  --outdir /path/to/ppaod_out \
  --device cpu
```

Equivalent module invocation:

```bash
python -m ppaod.run_ppaod \
  --qe-dir /path/to/qe \
  --seed carbon_wannier \
  --outdir /path/to/ppaod_out \
  --device cpu
```

Useful controls include:

```text
--n-basis 10              Bessel functions per angular momentum channel
--r-c 5.0                 radial cutoff in Bohr
--max-steps 150           Adam optimization steps
--alpha 0.8               weight of ΩI/J versus ΩD
--proj-threshold 0.01     minimum trial-orbital projectability
--restart-file FILE       projection-cache file
--force-rebuild-cache     ignore an existing projection cache
```

The output directory contains `V_H_theta.pt`, `H_k.npy`, `H_R.npy`,
`R_frac.npy`, optimization histories, `meta.json`, and diagnostic plots.

## Quick start: parallel

PPAOD shards k-points across processes using `torch.distributed`. For a
four-process CPU run:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  --module ppaod.run_ppaod \
  --qe-dir /path/to/qe \
  --seed carbon_wannier \
  --outdir /path/to/ppaod_out \
  --device cpu
```

For a four-GPU node, use the CUDA PyTorch build and change the device:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  --module ppaod.run_ppaod \
  --qe-dir /path/to/qe \
  --seed carbon_wannier \
  --outdir /path/to/ppaod_out \
  --device cuda
```

Only rank zero writes shared output files. Each rank loads only its assigned
k-point wavefunctions. On a cluster, the same command can be placed in a
SLURM script after loading Python, PyTorch, and any required MPI/runtime
modules.

## Development

Run the package tests from the repository root:

```bash
pytest -q
python -m ppaod.run_ppaod --help
```

The package preserves the original result filenames and command-line
semantics so existing workflow jobs can be migrated without changing their
input data.
