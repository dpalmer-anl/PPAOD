#!/usr/bin/env python3
"""
Differentiable Wannier disentanglement via optimized spherical-Bessel radial
trial orbitals (SMV Ω_I + Adam).

Usage (from the PPAOD repository)::

    ppaod --qe-dir ./qe --seed carbon_wannier --outdir ./bessel_smv_out

    torchrun --nproc_per_node=4 --module ppaod.run_ppaod \\
        --qe-dir ./qe --device cuda \\
        --outdir ./bessel_smv_out

Reads QE ``prefix.save/wfc*.dat`` and Wannier90 ``.amn/.mmn/.eig/.nnkp/.win``.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

from .distributed_k import (
    all_gather_v_list,
    all_reduce_sum_,
    barrier,
    destroy_dist,
    init_dist,
    is_rank0,
    print0,
    shard_kpoints,
)
from .file_io.qe_wavefunctions import load_all_wavefunctions
from .file_io.wannier90_files import (
    filter_band_idx_by_orbital_weight,
    load_wannier_dataset,
    window_band_mask,
)
from .model.initial_fit import amn_match_report, fit_theta0_to_amn
from .model.omega_D import make_energy_grid, omega_P
from .model.omega_I import build_V_list, slice_mmn_to_window
from .model.projection import ProjectionCache, build_projection_cache
from .optimize import optimize_theta
from .outputs import (
    build_Hk_from_V,
    build_HR_from_Hk,
    check_interlacing,
    is_complete_sp3,
    plot_bond_integrals_vs_distance,
    plot_disentangled_vs_dft,
    plot_HR_vs_distance,
    plot_lpdos_gamma_K,
    projection_labels,
    save_outputs,
)

from .disentangle import Disentangler


PROJECTION_RESTART_VERSION = 2


def projection_restart_path(
    outdir: Path,
    dist_info,
    requested: Path | None,
) -> Path:
    """Return the cache path, using one rank-local file for distributed runs."""
    path = requested or (outdir / "projection_cache.pt")
    path = Path(path)
    if dist_info.enabled:
        path = path.with_name(
            f"{path.stem}.rank{dist_info.rank:05d}{path.suffix or '.pt'}"
        )
    return path


def projection_restart_signature(
    *,
    data,
    band_idx_global: list[np.ndarray],
    k_owners: list[int],
    args: argparse.Namespace,
    dist_info,
) -> dict:
    """Metadata used to reject a cache made for incompatible fit settings."""
    return {
        "version": PROJECTION_RESTART_VERSION,
        "num_bands": int(data.num_bands),
        "num_wann": int(data.num_wann),
        "nk": int(len(data.amn)),
        "n_basis": int(args.n_basis),
        "r_c": float(args.r_c),
        "proj_threshold": float(args.proj_threshold),
        "global_k_indices": [int(x) for x in k_owners],
        "band_idx_global": [
            np.asarray(x, dtype=np.int64).tolist() for x in band_idx_global
        ],
        "world_size": int(dist_info.world_size),
    }


def save_projection_restart(
    path: Path,
    cache: ProjectionCache,
    signature: dict,
) -> None:
    """Save the WFC-dependent projection cache on CPU for portable reloads."""
    payload = {
        "signature": signature,
        "T_l": [
            {int(l): tensor.detach().cpu() for l, tensor in per_k.items()}
            for per_k in cache.T_l
        ],
        "angular_phase": [
            [tensor.detach().cpu() for tensor in per_k]
            for per_k in cache.angular_phase
        ],
        "c_win": [tensor.detach().cpu() for tensor in cache.c_win],
        "band_idx": [
            np.asarray(idx, dtype=np.int64) for idx in cache.band_idx
        ],
        "bases": {
            int(l): {
                "l": int(basis.l),
                "r_c": float(basis.r_c),
                "k_nodes": np.asarray(basis.k_nodes, dtype=np.float64),
            }
            for l, basis in cache.bases.items()
        },
        "orbitals": cache.orbitals,
        "orb_l": [int(x) for x in cache.orb_l],
        "omega": float(cache.omega),
        "J": int(cache.J),
        "global_k_indices": (
            None
            if cache.global_k_indices is None
            else [int(x) for x in cache.global_k_indices]
        ),
        "band_idx_global": [
            np.asarray(idx, dtype=np.int64)
            for idx in (cache.band_idx_global or [])
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_projection_restart(
    path: Path,
    signature: dict,
    *,
    device: torch.device,
) -> ProjectionCache | None:
    """Load a validated projection cache and move tensors to ``device``."""
    if not path.is_file():
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before the weights_only argument
            payload = torch.load(path, map_location="cpu")
        if payload.get("signature") != signature:
            print0(f"  restart cache incompatible; rebuilding: {path}")
            return None

        from .basis.bessel_basis import BesselBasis

        bases = {
            int(l): BesselBasis(
                l=int(item["l"]),
                r_c=float(item["r_c"]),
                k_nodes=np.asarray(item["k_nodes"], dtype=np.float64),
            )
            for l, item in payload["bases"].items()
        }
        cache = ProjectionCache(
            T_l=[
                {
                    int(l): tensor.to(device=device)
                    for l, tensor in per_k.items()
                }
                for per_k in payload["T_l"]
            ],
            angular_phase=[
                [tensor.to(device=device) for tensor in per_k]
                for per_k in payload["angular_phase"]
            ],
            c_win=[
                tensor.to(device=device) for tensor in payload["c_win"]
            ],
            band_idx=[
                np.asarray(idx, dtype=np.int64) for idx in payload["band_idx"]
            ],
            bases=bases,
            orbitals=payload["orbitals"],
            orb_l=[int(x) for x in payload["orb_l"]],
            omega=float(payload["omega"]),
            J=int(payload["J"]),
            global_k_indices=(
                None
                if payload["global_k_indices"] is None
                else [int(x) for x in payload["global_k_indices"]]
            ),
            band_idx_global=[
                np.asarray(idx, dtype=np.int64)
                for idx in payload["band_idx_global"]
            ],
        )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print0(f"  could not load restart cache; rebuilding {path}: {exc}")
        return None
    print0(f"  loaded projection restart: {path}")
    return cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--qe-dir", type=Path, default=HERE / "qe")
    p.add_argument("--seed", type=str, default="carbon_wannier")
    p.add_argument(
        "--prefix-save",
        type=str,
        default="carbon_calc.save",
        help="QE .save directory name under qe-dir",
    )
    p.add_argument("--outdir", type=Path, default=HERE / "bessel_smv_out")
    p.add_argument(
        "--restart-file",
        type=Path,
        default=None,
        help=(
            "Projection-cache restart path (default: "
            "<outdir>/projection_cache.pt; distributed runs add .rankXXXXX)"
        ),
    )
    p.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Ignore an existing projection-cache restart and rebuild it",
    )
    p.add_argument("--n-basis", type=int, default=10, help="Bessel basis size per l")
    p.add_argument("--r-c", type=float, default=5.0, help="Radial cutoff r_c (Angstrom)")
    p.add_argument("--ridge", type=float, default=1e-2, help="Ridge for initial AMN fit")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--tol", type=float, default=1e-8, help="Ω_P convergence tolerance")
    p.add_argument("--l2", type=float, default=1e-6)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Ω_P = α·(Ω_I/J) + (1−α)·Ω_D (default 0.8); Ω_I/J = per-Wannier spread",
    )
    p.add_argument("--dos-sigma", type=float, default=0.5, help="Gaussian LPDOS smear (eV)")
    p.add_argument("--dos-nE", type=int, default=256, help="LPDOS energy grid points")
    p.add_argument("--skip-optimize", action="store_true", help="Only initial LS fit + Ω eval")
    p.add_argument(
        "--no-smooth-gauge",
        action="store_true",
        help="Skip Löwdin-AMN gauge smoothing before H(R)",
    )
    p.add_argument(
        "--proj-threshold",
        type=float,
        default=0.01,
        help=(
            "Drop DFT bands with total trial-orbital weight "
            "Σ_i |⟨ψ_mk|g_i⟩|² below this threshold. "
            "Must retain ≥ num_wann bands at every k."
        ),
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--compile",
        action="store_true",
        help="Reserved: enable torch.compile on projection kernels when available",
    )
    p.add_argument("--verbose", action="store_true", help="Print on all ranks")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dist_info = init_dist(args.device)
    device = dist_info.device
    qe_dir = args.qe_dir
    save_dir = qe_dir / args.prefix_save
    outdir = args.outdir
    if is_rank0(dist_info):
        outdir.mkdir(parents=True, exist_ok=True)
    barrier()

    print0("=== Load Wannier90 dataset ===", info=dist_info)
    data = load_wannier_dataset(qe_dir, args.seed)
    J = data.num_wann
    Nk = len(data.amn)
    print0(f"  Nk={Nk}  num_bands={data.num_bands}  J={J}", info=dist_info)
    print0(f"  dis_win_max={data.dis_win_max}  mp_grid={data.mp_grid}", info=dist_info)
    print0(
        "  orbitals: " + ", ".join(f"(l={o.l},mr={o.mr})" for o in data.orbitals),
        info=dist_info,
    )
    proj_names = projection_labels(data.orbitals)
    uniq = sorted(set(proj_names), key=proj_names.index)
    print0(
        f"  projection labels ({len(proj_names)}): "
        + ", ".join(f"{n}×{proj_names.count(n)}" for n in uniq),
        info=dist_info,
    )
    if dist_info.enabled:
        print0(
            f"  distributed: rank {dist_info.rank}/{dist_info.world_size} "
            f"backend={dist_info.backend} device={device}",
            info=dist_info,
        )

    # Outer-window band indices per k
    band_idx = []
    for ik in range(Nk):
        mask = window_band_mask(data.eig[ik], data.dis_win_min, data.dis_win_max)
        idx = np.flatnonzero(mask).astype(np.int64)
        if idx.size < J:
            raise RuntimeError(f"k={ik}: only {idx.size} bands in window, need ≥ J={J}")
        band_idx.append(idx)
    nwin = [len(i) for i in band_idx]
    print0(
        f"  outer window Nb[k]: min={min(nwin)} max={max(nwin)} mean={np.mean(nwin):.1f}",
        info=dist_info,
    )

    print0(
        f"=== Filter bands by orbital weight "
        f"Σ_i |⟨ψ|g_i⟩|² ≥ {args.proj_threshold} ===",
        info=dist_info,
    )
    band_idx, proj_stats = filter_band_idx_by_orbital_weight(
        data.amn,
        band_idx,
        threshold=args.proj_threshold,
        num_wann=J,
    )
    print0(
        f"  retained Nb[k]: min={proj_stats['n_after_min']} "
        f"max={proj_stats['n_after_max']} "
        f"mean={proj_stats['n_after_mean']:.1f}  "
        f"(removed {proj_stats['n_removed_total']} band·k total)",
        info=dist_info,
    )

    k_owners = shard_kpoints(Nk, dist_info.rank, dist_info.world_size)
    print0(
        f"=== Load QE wavefunctions (rank-local k={len(k_owners)}/{Nk}) ===",
        info=dist_info,
    )
    if args.verbose:
        print(f"  [rank {dist_info.rank}] k_owners={k_owners[:8]}{'...' if len(k_owners)>8 else ''}")

    cache_signature = projection_restart_signature(
        data=data,
        band_idx_global=band_idx,
        k_owners=k_owners,
        args=args,
        dist_info=dist_info,
    )
    restart_path = projection_restart_path(
        outdir,
        dist_info,
        args.restart_file,
    )
    cache = None
    if not args.force_rebuild_cache:
        cache = load_projection_restart(
            restart_path,
            cache_signature,
            device=device,
        )

    if cache is None:
        t0 = time.time()
        c_list, G_list, k_cart_list, alat, _b = load_all_wavefunctions(
            save_dir,
            k_indices=k_owners,
        )
        print0(
            f"  alat={alat:.6f} Angstrom  loaded local WFCs in {time.time()-t0:.1f}s "
            f"(rank0 nG={G_list[0].shape[0] if G_list else 0})",
            info=dist_info,
        )

        print0(
            "=== Precompute T_l / Y_lm / phase cache (local k) ===",
            info=dist_info,
        )
        t0 = time.time()
        cache = build_projection_cache(
            c_list=c_list,
            Gvecs_list=G_list,
            k_cart_list=k_cart_list,
            orbitals=data.orbitals,
            band_idx=band_idx,
            real_lattice_ang=data.real_lattice_ang,
            r_c=args.r_c,
            n_basis=args.n_basis,
            device=device,
            global_k_indices=k_owners,
            band_idx_global=band_idx,
        )
        # Free raw WFCs — c_win lives in the cache.
        del c_list, G_list, k_cart_list
        gc.collect()
        print0(
            f"  done in {time.time()-t0:.1f}s  Ω_cell={cache.omega:.4f} Angstrom³",
            info=dist_info,
        )
        save_projection_restart(restart_path, cache, cache_signature)
        print0(f"  saved projection restart: {restart_path}", info=dist_info)
    else:
        print0(
            f"=== Reusing projection cache (QE WFC read skipped) ===",
            info=dist_info,
        )

    assert cache is not None
    for l, basis in cache.bases.items():
        print0(
            f"  l={l}: r_c={basis.r_c}  "
            f"k_nodes={np.array2string(basis.k_nodes, precision=4)}",
            info=dist_info,
        )

    print0("=== Step 2: least-squares fit θ₀ to .amn (online XtX) ===", info=dist_info)
    theta0 = fit_theta0_to_amn(
        cache,
        data.amn,
        band_idx=band_idx,
        ridge=args.ridge,
        all_reduce=dist_info.enabled,
    )
    for l, c in theta0.items():
        print0(f"  c_ml[l={l}] = {c.detach().cpu().numpy()}", info=dist_info)
    match = amn_match_report(theta0, cache, data.amn, band_idx)
    print0(
        f"  |A(θ₀)−A_amn| (local k): rms={match['rms']:.4e}  max={match['max']:.4e}",
        info=dist_info,
    )
    if match["rms"] > 0.5:
        print0(
            "  WARNING: large AMN mismatch — check FT convention / r_c / Y_lm",
            info=dist_info,
        )

    mmn_win = slice_mmn_to_window(data, band_idx, k_owners=k_owners)

    # AMN Löwdin reference Ω_I (rank 0) while full MMN still in memory
    if is_rank0(dist_info):
        mmn_all = slice_mmn_to_window(data, band_idx)
        U_amn = []
        for ik in range(Nk):
            A = data.amn[ik][band_idx[ik]].to(dtype=torch.complex128)
            Z, _, Vh = torch.linalg.svd(A, full_matrices=False)
            U_amn.append(Z @ Vh)
        om_amn = Disentangler(
            U_amn,
            mmn_all,
            data.neighbors,
            data.weights,
            data.num_wann,
            verbose=False,
        ).omega_I(U_amn)
        print(
            f"  Ω_I(AMN Löwdin) = {om_amn:.8f}  (total; W90-comparable)  "
            f"Ω_I/J = {om_amn / J:.8f}"
        )
        del mmn_all, U_amn

    # Drop full MMN after window slice
    data.mmn.clear()
    gc.collect()

    grid = make_energy_grid(
        data.eig,
        band_idx,
        sigma=args.dos_sigma,
        nE=args.dos_nE,
        device=device,
    )
    print0(
        f"  LPDOS grid: E∈[{float(grid.E[0]):.2f},{float(grid.E[-1]):.2f}] eV  "
        f"nE={grid.E.numel()}  σ={grid.sigma}  α={args.alpha}",
        info=dist_info,
    )

    with torch.no_grad():
        V_local = build_V_list(theta0, cache)
        if dist_info.enabled:
            V0 = all_gather_v_list(V_local, k_owners, Nk, device=device)
            oP0_loc, oI0_loc, oD0_loc, _ = omega_P(
                theta0,
                cache,
                data,
                mmn_win=mmn_win,
                grid=grid,
                alpha=args.alpha,
                V_full=V0,
                k_owners=k_owners,
                Nk_global=Nk,
            )
            oP0 = all_reduce_sum_(oP0_loc.clone())
            oI0 = all_reduce_sum_(oI0_loc.clone())
            oD0 = all_reduce_sum_(oD0_loc.clone())
        else:
            oP0, oI0, oD0, _ = omega_P(
                theta0, cache, data, mmn_win=mmn_win, grid=grid, alpha=args.alpha
            )
    print0(
        f"  Ω_P(θ₀)={float(oP0):.8f}  Ω_I/J={float(oI0):.8f}  Ω_D={float(oD0):.8f}",
        info=dist_info,
    )

    history_I: list[float] | None = None
    history_D: list[float] | None = None
    if args.skip_optimize:
        theta = theta0
        history = [float(oP0)]
        history_I = [float(oI0)]
        history_D = [float(oD0)]
        with torch.no_grad():
            V_local = build_V_list(theta, cache, check_V=True)
            if dist_info.enabled:
                V_list = all_gather_v_list(V_local, k_owners, Nk, device=device)
                oP_loc, oI_loc, oD_loc, V_list = omega_P(
                    theta,
                    cache,
                    data,
                    mmn_win=mmn_win,
                    grid=grid,
                    alpha=args.alpha,
                    V_full=V_list,
                    k_owners=k_owners,
                    Nk_global=Nk,
                )
                omega_final = float(all_reduce_sum_(oP_loc.clone()))
                omega_I_final = float(all_reduce_sum_(oI_loc.clone()))
                omega_D_final = float(all_reduce_sum_(oD_loc.clone()))
            else:
                oP, oI, oD, V_list = omega_P(
                    theta,
                    cache,
                    data,
                    mmn_win=mmn_win,
                    grid=grid,
                    alpha=args.alpha,
                    check_V=True,
                )
                omega_final = float(oP)
                omega_I_final = float(oI)
                omega_D_final = float(oD)
        V_list = [V.detach().cpu() for V in V_list]
        steps = 0
    else:
        print0(
            f"=== Step 4: Adam optimize Ω_P = {args.alpha} (Ω_I/J) + {1 - args.alpha} Ω_D ===",
            info=dist_info,
        )
        result = optimize_theta(
            theta0,
            cache,
            data,
            mmn_win,
            grid=grid,
            alpha=args.alpha,
            lr=args.lr,
            max_steps=args.max_steps,
            tol=args.tol,
            l2=args.l2,
            dist_info=dist_info,
            Nk_global=Nk,
            k_owners=k_owners,
            compile_proj=args.compile,
        )
        theta = result.theta
        history = result.history
        history_I = result.history_I
        history_D = result.history_D
        V_list = result.V_list
        omega_final = result.omega_final
        omega_I_final = result.omega_I_final
        omega_D_final = result.omega_D_final
        steps = result.steps
        print0(
            f"  final Ω_P={omega_final:.8f}  Ω_I/J={omega_I_final:.8f}  "
            f"Ω_D={omega_D_final:.8f}  after {steps} steps",
            info=dist_info,
        )

    # Post-processing / plots on rank 0 only (full V_list gathered above)
    if not is_rank0(dist_info):
        barrier()
        destroy_dist()
        return 0

    if not args.no_smooth_gauge:
        print("=== Smooth within-subspace gauge (align to AMN Löwdin) ===")
        V_smooth = []
        for ik, V in enumerate(V_list):
            A = data.amn[ik][band_idx[ik]].to(dtype=torch.complex128)
            Z, _, Vh = torch.linalg.svd(A, full_matrices=False)
            U_ref = Z @ Vh
            V_smooth.append(Disentangler.align_to_reference(V, U_ref))
        V_list = V_smooth

    print("=== Build H(k) = V† E V ===")
    H_k = build_Hk_from_V(V_list, data.eig, band_idx)
    warns = check_interlacing(H_k, data.eig, band_idx)
    if warns:
        for w in warns:
            print("  INTERLACING WARNING:", w)
    else:
        print("  interlacing OK on sampled k-points")

    print("=== Fourier transform H(k) → H(R) ===")
    H_R, R_frac = build_HR_from_Hk(H_k, data.kpts_crystal, data.mp_grid)
    print(f"  H_R shape={H_R.shape}  nR={R_frac.shape[0]}  mp_grid={data.mp_grid}")

    meta = {
        "seed": args.seed,
        "qe_dir": str(qe_dir),
        "J": J,
        "n_basis": args.n_basis,
        "r_c": args.r_c,
        "r_c_units": "Angstrom",
        "units": {"energy": "eV", "length": "Angstrom"},
        "alpha": args.alpha,
        "dos_sigma": args.dos_sigma,
        "omega_P": omega_final,
        "omega_I": omega_I_final,
        "omega_D": omega_D_final,
        "steps": steps,
        "amn_fit_rms": match["rms"],
        "theta": {str(l): c.detach().cpu().tolist() for l, c in theta.items()},
        "hamiltonian_note": (
            "H(k)=V†EV is disentangled Bloch-gauge Hamiltonian before MLWF U(k); "
            "H_R is its MP Fourier transform"
        ),
        "omega_P_note": (
            "Ω_P = α (Ω_I_tot/J) + (1-α) Ω_D; "
            "Ω_I logged as per-Wannier spread; "
            "Ω_D = mean_{k,i} KL(LPDOS_TB || LPDOS_DFT) with unit-normalized LPDOS"
        ),
        "nR": int(R_frac.shape[0]),
        "proj_threshold": args.proj_threshold,
        "proj_filter": proj_stats,
        "projections": proj_names,
        "distributed": dist_info.enabled,
        "world_size": dist_info.world_size,
    }
    save_outputs(
        outdir,
        omega_history=history,
        omega_final=omega_final,
        theta=theta,
        V_list=V_list,
        H_k=H_k,
        kpts_crystal=data.kpts_crystal,
        meta=meta,
        history_I=history_I,
        history_D=history_D,
        H_R=H_R,
        R_frac=R_frac,
    )
    print(f"  wrote {outdir / 'V_H_theta.pt'}  (+ H_R.npy, R_frac.npy)")

    print("=== H(R) vs orbital distance (by channel) ===")
    hop_path = plot_HR_vs_distance(
        outdir,
        H_R=H_R,
        R_frac=R_frac,
        orbitals=data.orbitals,
        real_lattice_ang=data.real_lattice_ang,
    )
    print(f"  wrote {hop_path}")

    if is_complete_sp3(data.orbitals, data.real_lattice_ang):
        print("=== SK bond integrals from 4×4 atom blocks ===")
        bi_path = plot_bond_integrals_vs_distance(
            outdir,
            H_R=H_R,
            R_frac=R_frac,
            orbitals=data.orbitals,
            real_lattice_ang=data.real_lattice_ang,
        )
        print(f"  wrote {bi_path}  (+ bond_integrals.npz)")
    else:
        print(
            "=== SK bond integrals skipped "
            "(need full s,px,py,pz shells on every site) ==="
        )

    print("=== LPDOS DFT vs TB at Γ and K (available angular channels) ===")
    lpdos_path = plot_lpdos_gamma_K(
        outdir,
        V_list=V_list,
        eig=data.eig,
        band_idx=band_idx,
        kpts_crystal=data.kpts_crystal,
        orb_l=cache.orb_l,
        grid=grid,
    )
    if lpdos_path is None:
        print("  skip LPDOS plot: no s/p channels in projections")
    else:
        print(f"  wrote {lpdos_path}")

    bands_out = qe_dir / "bands.out"
    if bands_out.is_file():
        print("=== Band path H(k)→H(R)→path vs DFT ===")
        plot_path = plot_disentangled_vs_dft(
            outdir,
            H_k=H_k,
            kpts_mesh=data.kpts_crystal,
            mp_grid=data.mp_grid,
            bands_out=bands_out,
            H_R=H_R,
            R_frac=R_frac,
        )
        print(f"  wrote {plot_path}")
    else:
        print(f"  skip band plot: {bands_out} not found")

    print("Done.")
    barrier()
    destroy_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
