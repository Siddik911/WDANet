# TRIAD implementation — staged LOSO depression detection (MODMA)

Incremental, ablation-driven build-up from the roadmap
(`TRIAD_Progressive_Research_Roadmap.md`). The **data pipeline is frozen** and shared
by every stage; only the model changes between stages, so each stage-to-stage AUC delta
is interpretable.

## Layout
```
triad/
  config.py    frozen pipeline constants (fs, bands, window, channels, ...)
  data.py      load .mat -> bandpass 1-40Hz -> avg-ref -> decimate 250->125Hz -> cache -> epoch
  metrics.py   subject-level AUC / balanced-acc / macro-F1 / confusion + live per-fold reporter
  models.py    S0 EEGNet (torch) | S1 cov | S2 corr | S3 corr+recentring (pyriemann)
  harness.py   strict 53-fold Leave-One-Subject-Out loop
run.py         CLI
cache/         preprocessed per-subject signals (built once)
results/       per-stage per-fold JSON
```

## Dataset
MODMA 128-ch resting-state EEG, `EEG_128channels_resting_lanzhou_2015/`:
53 subjects = **24 MDD / 29 HC**, 250 Hz, ~5 min each. Channel 129 (zero reference)
dropped -> 128 channels. Positive class = MDD.

## Stages implemented
| Stage | Model | Mechanism added | Tests |
|------|-------|-----------------|-------|
| S0 | EEGNet (CNN) | Euclidean floor + frozen LOSO harness | leakage-free pipeline |
| S1 | LW-covariance + tangent-space LR | SPD geometry | geometry > Euclidean |
| S2 | correlation manifold + tangent LR | scale-invariance | corr > cov |
| S3 | S2 + per-subject Fréchet recentring | one-shot transduction | recentring jump |

(S4–S8 — deep encoder, SPDDSMBN, test-time transduction, ETPR, meta — come next.)

## Commands

```bash
# 0) one-off: build the preprocessing cache (~25 s; auto-runs on first stage too)
python run.py --build-cache

# quick smoke test on 8 class-balanced subjects (~20 s)
python run.py --stage s1 --limit-subjects 8 --max-windows 60

# full 53-fold LOSO (prints live per-fold metrics + final panel; saves results/<stage>_loso.json)
python run.py --stage s1        # ~11 min   covariance baseline
python run.py --stage s2        # ~11 min   correlation manifold
python run.py --stage s3        # ~12 min   + per-subject recentring
python run.py --stage s0 --epochs 30   # LONG (~2-3 h): trains EEGNet from scratch per fold (GPU)
```

Long jobs — run detached and watch the log:
```bash
nohup python run.py --stage s3 > results/s3.log 2>&1 &
tail -f results/s3.log
```

Useful flags: `--max-windows N` (cap windows/subject, shallow speed), `--C-reg`
(LR regularisation), `--epochs/--batch-size/--lr/--dropout/--device` (S0),
`--limit-subjects N` (smoke subset), `--no-save`.

## Live per-fold output
Each fold prints: held-out subject, true label, P(MDD) score, prediction, hit/miss,
#windows, **running cumulative AUC + balanced accuracy**, fold time. The final panel
reports subject-level ROC-AUC (primary), balanced accuracy and macro-F1 at both 0.50
and the Youden threshold, and the confusion matrix.

## Baseline result so far
- **S1 (covariance + tangent LR): subject-level ROC-AUC = 0.552** over 53 LOSO folds —
  an honest weak floor (no leakage; not implausibly high). Target to beat downstream:
  re-run EEG-RCformer 0.7154.

## Verification
`python verify_models.py` — 18/18 component checks pass against references:
`_cov_to_corr` matches `D^{-1/2}CD^{-1/2}` to 2e-16; `_recenter` Fréchet mean == I to
1e-14 (= pyriemann `TLCenter` whitening); tangent round-trip exact; EEGNet matches the
TSMNet `EEGNetv4` reference (3,602 params, max-norm constraints active); covariances SPD
& well-conditioned (Ledoit-Wolf, cond ~7e4).

Leakage gate (roadmap S0): `python run.py --stage s1 --permute-labels 0` — with labels
shuffled across subjects the AUC collapses to **0.417 (~chance)** vs **0.552** with real
labels, confirming the LOSO split has no information leakage.
