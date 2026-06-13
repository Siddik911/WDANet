#!/usr/bin/env python3
"""TRIAD — run a stage under strict LOSO on MODMA depression EEG.

Examples
--------
  # one-off: build the frozen preprocessing cache (also happens automatically)
  python run.py --build-cache

  # S1 shallow Riemannian baseline (fast, no GPU) — start here to validate the harness
  python run.py --stage s1

  # S2 correlation manifold / S3 + per-subject recentring
  python run.py --stage s2
  python run.py --stage s3

  # S0 Euclidean EEGNet floor (GPU)
  python run.py --stage s0 --epochs 30

  # quick smoke test on a class-balanced subset of subjects
  python run.py --stage s1 --limit-subjects 8
"""
import argparse
import sys

from triad import config as C
from triad import data as D
from triad.harness import run_loso


def main():
    # Flush every line immediately so progress streams to a log/pipe in real time
    # (Python block-buffers stdout when it is redirected to a file).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="modma",
                   choices=["modma", "mumtaz", "opennuero"],
                   help="which resting-state dataset to run on (default modma). "
                        "mumtaz=Mumtaz 2016 19-ch eyes-closed (.edf); "
                        "opennuero=Cavanagh ds003478 64-ch (.set). Each is resampled to "
                        "125 Hz and cached under cache/<dataset>/.")
    p.add_argument("--stage", default="s1",
                   choices=["s0", "s1", "s2", "s3", "s4", "s5", "n1", "cc", "cl",
                            "nr", "nc", "cr", "bd", "tnr", "ncv", "mb", "ds", "mbds",
                            "nrde", "de", "spec", "defu", "spfu", "mlp", "wattn"],
                   help="which stage model to run (default s1). n1=nuisance/detrend; "
                        "cc=class-conditional LDA; cl=AIRM channel clustering 125->K; "
                        "nr=identity-subspace removal (S2 + project out top-K identity PCs); "
                        "nc=identity removal on full covariance (= both cov & corr, no hack); "
                        "cr=nr + per-channel log-power, standardized (the info corr drops vs cov); "
                        "bd=block-diagonal cov⊕corr product manifold + identity removal; "
                        "tnr=transductive nr (fold held-out subject's unlabelled windows into "
                        "the class-orthogonalised nuisance subspace)")
    p.add_argument("--k-clusters", type=int, default=5,
                   help="number of channel clusters / region tokens for stage cl")
    p.add_argument("--n-nuisance", type=int, default=3,
                   help="number of between-subject identity PCs to project out (stage nr)")
    p.add_argument("--n-iter", type=int, default=0,
                   help="transductive (tnr) refinement iterations (0 = init-only base)")
    p.add_argument("--control", default="none",
                   choices=["none", "inductive", "random_inject"],
                   help="tnr ablation: 'inductive'=train-only V (= nr+class-protect); "
                        "'random_inject'=swap target mean for a random train subject "
                        "(if AUC is unchanged, the gain is NOT from target identity)")
    p.add_argument("--build-cache", action="store_true",
                   help="(re)build the preprocessing cache and exit")
    p.add_argument("--force-cache", action="store_true",
                   help="rebuild cache even if it exists")
    p.add_argument("--limit-subjects", type=int, default=None,
                   help="use only N (class-balanced) subjects — for quick smoke tests")
    p.add_argument("--cv-folds", type=int, default=None,
                   help="use K-fold stratified CV instead of LOSO (e.g. 10 to match WDANet). "
                        "Default: LOSO (one held-out subject per fold).")
    p.add_argument("--no-save", action="store_true", help="do not write results json")
    p.add_argument("--permute-labels", type=int, default=None, metavar="SEED",
                   help="leakage sanity test: shuffle labels across subjects "
                        "(AUC must collapse to ~0.5)")

    # shallow-stage (s1/s2/s3) options
    p.add_argument("--max-windows", type=int, default=120,
                   help="cap windows/subject for shallow stages (default 120)")
    p.add_argument("--C-reg", type=float, default=1.0,
                   help="logistic-regression inverse regularisation (shallow stages)")

    # deep-stage (s0/s4/s5) options
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.25, help="EEGNet dropout (s0)")
    p.add_argument("--subspace", type=int, default=20, help="SPDNet BiMap subspace dim (s4/s5/ds); also PCA dim for mlp/wattn")
    p.add_argument("--attn-dim", type=int, default=32,
                   help="wattn: additive attention hidden size (default 32)")
    p.add_argument("--hidden-dim", type=int, default=0,
                   help="wattn: classifier hidden layer size (0 = direct linear, default 0)")
    # dual-stream SPD (ds) options
    p.add_argument("--streams", default="cov,corr",
                   help="ds manifolds: 'cov' | 'corr' | 'cov,corr' (ablation ladder)")
    p.add_argument("--identity-removal", action="store_true",
                   help="ds Phase 2: freeze encoder, remove top-K identity PCs, refit head")
    p.add_argument("--ds-clusters", type=int, default=0,
                   help="ds: AIRM-cluster 125->K channels before the streams (0=off). "
                        "Compresses the input to KxK to regularise the deep encoder.")
    p.add_argument("--spatial-filters", type=int, default=40, help="SPDNet spatial filters (s4/s5)")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--verbose", action="store_true", help="print per-epoch loss")
    args = p.parse_args()

    C.set_active_dataset(args.dataset)
    print(f"[dataset] {args.dataset}  raw={C.DATA_DIR}  cache={C.CACHE_DIR}  "
          f"fs_orig={C.FS_ORIG}->{C.FS}Hz  ch={C.N_CHANNELS}")

    if args.build_cache or args.force_cache:
        print("Building preprocessing cache ...")
        D.build_cache(force=args.force_cache)
        print(f"Cache ready at {C.CACHE_DIR}")
        if args.build_cache:
            return

    if args.stage == "s0":
        mk = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  dropout=args.dropout, device=args.device, verbose=args.verbose)
    elif args.stage in ("s4", "s5"):
        mk = dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  subspace=args.subspace, spatial_filters=args.spatial_filters,
                  device=args.device, verbose=args.verbose)
    elif args.stage == "cl":
        mk = dict(n_clusters=args.k_clusters, max_windows=args.max_windows,
                  C_reg=args.C_reg)
    elif args.stage == "ds":
        mk = dict(streams=tuple(args.streams.split(",")),
                  identity_removal=args.identity_removal,
                  n_nuisance=args.n_nuisance, subspace=args.subspace,
                  n_clusters=args.ds_clusters,
                  epochs=args.epochs, max_windows=args.max_windows,
                  lr=args.lr, device=args.device, verbose=args.verbose)
    elif args.stage in ("mb", "de", "spec", "defu", "spfu"):  # band-power/spectral/fusion stages
        mk = dict(n_nuisance=args.n_nuisance, max_windows=args.max_windows, C_reg=args.C_reg)
    elif args.stage == "mlp":
        mk = dict(n_nuisance=args.n_nuisance, max_windows=args.max_windows,
                  epochs=args.epochs, pca_dim=args.subspace)
    elif args.stage == "wattn":
        mk = dict(n_nuisance=args.n_nuisance, max_windows=args.max_windows,
                  epochs=args.epochs, pca_dim=args.subspace,
                  attn_dim=args.attn_dim, hidden_dim=args.hidden_dim)
    elif args.stage == "mbds":                         # multiband dual-SPD ensemble
        mk = dict(streams=tuple(args.streams.split(",")),
                  n_nuisance=args.n_nuisance, subspace=args.subspace,
                  epochs=args.epochs, max_windows=args.max_windows,
                  lr=args.lr, device=args.device, verbose=args.verbose)
    elif args.stage in ("nr", "nc", "cr", "bd", "tnr", "ncv"):
        mk = dict(max_windows=args.max_windows, C_reg=args.C_reg)
        if args.stage != "ncv":                        # ncv picks K itself (inner-CV)
            mk["n_nuisance"] = args.n_nuisance
        if args.stage == "tnr":
            mk.update(n_iter=args.n_iter,
                      control=None if args.control == "none" else args.control)
    else:
        mk = dict(max_windows=args.max_windows, C_reg=args.C_reg)

    run_loso(stage=args.stage, model_kwargs=mk,
             limit_subjects=args.limit_subjects, save=not args.no_save,
             permute_labels=args.permute_labels,
             cv_folds=args.cv_folds)


if __name__ == "__main__":
    main()
