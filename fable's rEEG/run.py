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
                            "nrde", "de", "spec", "defu", "spfu", "mlp", "wattn",
                            "spddann"],
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
    p.add_argument("--harmonize", action="store_true",
                   help="project onto the shared 17-channel 10-20 montage (triad.harmonize) "
                        "so correlation matrices are cross-dataset compatible — required "
                        "for LODO; on a single dataset it measures the harmonisation cost.")
    p.add_argument("--test", default=None,
                   choices=["modma", "mumtaz", "opennuero"],
                   help="LODO mode: held-out target dataset. Setting this switches to "
                        "Leave-One-Dataset-Out — fit on --train datasets (pooled), evaluate "
                        "on this one. Implies --harmonize (montages differ).")
    p.add_argument("--train", default=None,
                   help="LODO source datasets, comma-separated (e.g. mumtaz,opennuero). "
                        "Default: the two datasets that are not --test.")
    p.add_argument("--recenter-datasets", action="store_true",
                   help="LODO L3 alignment: recentre each dataset's Riemannian (Frechet) "
                        "mean to identity before the shared tangent projection — sources at "
                        "fit, target transductively from its unlabelled windows. "
                        "Applies to nr/nrde/ncv.")
    p.add_argument("--de-standardize", action="store_true",
                   help="LODO spectral alignment: z-score the DE block per dataset (source "
                        "scalers at fit, target scaler transductively) to remove amplifier "
                        "band-power offsets. Applies to nrde.")
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
    p.add_argument("--seed", type=int, default=C.SEED,
                   help="random seed for deep stages (spddann) — vary for robustness")
    p.add_argument("--no-domain-adapt", action="store_true",
                   help="spddann: disable the domain-adversarial losses (GRL/L_M/L_P) and run "
                        "the supervised two-stream classifier only. Use WITHIN-dataset (no "
                        "domain gap); the DA losses destabilise same-distribution CV folds.")
    p.add_argument("--spd-id-removal", type=int, default=0,
                   help="spddann: strip the top-K between-subject identity PCs from the fused "
                        "tangent feature each epoch (nr's within-dataset mechanism). 0=off. "
                        "Try 3 to recover the shallow within-dataset performance.")
    p.add_argument("--feat-dropout", type=float, default=0.0,
                   help="spddann: dropout on the fused feature before the classifier (regulariser "
                        "for the small-N within-dataset regime). 0=off.")
    p.add_argument("--spd-bn", action="store_true",
                   help="spddann: insert Riemannian SPD batch-norm (Fréchet-mean recentre + "
                        "scalar dispersion, Kobler 2022) after BiMap. Regularises the encoder "
                        "and learns a mean-alignment; NeurIPS-2022's cross-dataset gain driver.")
    p.add_argument("--nr-stream", action="store_true",
                   help="spddann: add the fixed nrde connectivity stream (identity-removed "
                        "correlation tangent) to the fused vector. Within a single dataset this "
                        "lets the classifier match nrde (~0.71); for LODO the learned+adversarial "
                        "streams still carry transfer. The 'one model, both regimes' variant.")
    p.add_argument("--nr-k", type=int, default=3,
                   help="spddann --nr-stream: identity PCs removed from the fixed nr tangent "
                        "(shallow default 3 native; use 1 for harmonized/LODO).")
    p.add_argument("--nr-proj-dim", type=int, default=0,
                   help="spddann --nr-stream: route the fixed nr tangent through a learnable "
                        "Linear(nr_dim->K) that feeds the classifier AND the GRL discriminator, "
                        "so adversarial training can SUPPRESS the non-transferable connectivity "
                        "for LODO (fixes the nr-stream LODO overfit) while keeping it within-"
                        "dataset. 0 = raw concat (not GRL-suppressible). Try 64.")
    p.add_argument("--nr-domain-removal", type=int, default=0,
                   help="spddann --nr-stream: strip top-D between-DATASET directions from the nr "
                        "tangent (class-axis protected) — removes the source-specific connectivity "
                        "that causes the MODMA-LODO negative transfer. No-op within-dataset (one "
                        "domain). LODO only. Try 1-2.")
    p.add_argument("--wdann", action="store_true",
                   help="spddann (LODO): WDANet-style alignment — replace the CE domain "
                        "discriminator + prototype loss with global MMD + local class-conditional "
                        "MMD (LMMD), balanced by a dynamic factor ω. Aims to lift the transferable "
                        "LODO folds (MODMA/Mumtaz).")
    p.add_argument("--align-weight", type=float, default=1.0,
                   help="spddann --wdann: weight λ on the MMD alignment term (ramped by the DANN "
                        "schedule). Try 1.0-5.0.")
    p.add_argument("--spd-input", default="conv", choices=["conv", "corr"],
                   help="spddann SPD stream input: 'conv'=learned temporal+spatial conv then "
                        "covariance (paper backbone); 'corr'=raw channel correlation manifold "
                        "into BiMap (a learnable nr — beats the conv encoder within-dataset at "
                        "clinical N where conv overfits).")
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
    elif args.stage == "spddann":                      # domain-adversarial SPD (LODO, transductive)
        mk = dict(epochs=args.epochs, lr=args.lr, subspace=args.subspace,
                  max_windows=args.max_windows, device=args.device,
                  verbose=args.verbose, seed=args.seed,
                  domain_adapt=not args.no_domain_adapt,
                  identity_removal=args.spd_id_removal, feat_dropout=args.feat_dropout,
                  spd_input=args.spd_input, spd_bn=args.spd_bn,
                  nr_stream=args.nr_stream, nr_k=args.nr_k, nr_proj_dim=args.nr_proj_dim,
                  nr_domain_removal=args.nr_domain_removal,
                  wdann=args.wdann, align_weight=args.align_weight)
    elif args.stage == "mbds":                         # multiband dual-SPD ensemble
        mk = dict(streams=tuple(args.streams.split(",")),
                  n_nuisance=args.n_nuisance, subspace=args.subspace,
                  epochs=args.epochs, max_windows=args.max_windows,
                  lr=args.lr, device=args.device, verbose=args.verbose)
    elif args.stage in ("nr", "nc", "cr", "bd", "tnr", "ncv", "nrde"):
        mk = dict(max_windows=args.max_windows, C_reg=args.C_reg)
        if args.stage != "ncv":                        # ncv picks K itself (inner-CV)
            mk["n_nuisance"] = args.n_nuisance
        if args.recenter_datasets:                     # LODO L3 per-dataset recentring
            mk["dataset_recenter"] = True
        if args.de_standardize:                        # LODO per-dataset DE standardisation
            mk["de_dataset_standardize"] = True
        if args.stage == "tnr":
            mk.update(n_iter=args.n_iter,
                      control=None if args.control == "none" else args.control)
    else:
        mk = dict(max_windows=args.max_windows, C_reg=args.C_reg)

    if args.test is not None:
        # ---- LODO: train on the pooled source datasets, test on the held-out one ----
        from triad.harness import run_lodo
        all_ds = ["modma", "mumtaz", "opennuero"]
        train_ds = (args.train.split(",") if args.train
                    else [d for d in all_ds if d != args.test])
        bad = [d for d in train_ds if d not in all_ds] + (
            [args.test] if args.test in train_ds else [])
        if bad:
            p.error(f"invalid LODO datasets: train={train_ds} test={args.test}")
        run_lodo(stage=args.stage, train_datasets=train_ds, test_dataset=args.test,
                 model_kwargs=mk, save=not args.no_save, harmonize=True)
        return

    run_loso(stage=args.stage, model_kwargs=mk,
             limit_subjects=args.limit_subjects, save=not args.no_save,
             permute_labels=args.permute_labels,
             cv_folds=args.cv_folds, harmonize=args.harmonize)


if __name__ == "__main__":
    main()
