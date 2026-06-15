#!/usr/bin/env python3
"""Why did S4 (deep SPD net) fail? Five small tests on a 10-subject MODMA sample.

Each test isolates one hypothesis for the overfitting we saw (train loss -> ~0,
held-out predictions scattered):

  A. PROTOCOL   — window-mixed split vs subject-independent split.
                  If mixed >> subject-independent, the high numbers in the
                  literature come from leakage (same subject in train & test),
                  not better models. (Tests "is my evaluation just honest-hard?")
  B. IDENTITY   — can the learned features recover SUBJECT ID?
                  If yes, the net spent its capacity on individual identity, not
                  depression (consistent with the [B] finding). (Model objective.)
  C. OVERFIT    — train-acc vs held-out AUC across epochs.
                  Shows whether early stopping would have saved it, or the deep
                  net never generalises. (Capacity / training.)
  D. SHALLOW≷DEEP — same split, S2 shallow vs S4 deep.
                  If shallow > deep on identical data, the data HAS signal and the
                  deep model is the problem, not the dataset. (Data vs model.)
  E. SHUFFLE    — train S4 on RANDOM labels.
                  If train loss still -> 0, then "loss -> 0" is memorisation of
                  per-subject-constant labels, i.e. meaningless. (Smoking gun.)

Run:  python -u diagnose_s4.py 2>&1 | tee results/diagnose_s4.log     (~3-6 min)
"""
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

import os as _os, sys as _sys  # bootstrap: repo root on path (script moved to scripts/)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from triad import data as D
from triad.models import make_shallow
from triad.spdnet import SPDNet
from geoopt.optim import RiemannianAdam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAXW = 60          # windows per subject (small + fast)
EPOCHS = 20
SEED = 0


# --------------------------------------------------------------------------
def subj_windows(sub, max_w=MAXW):
    w = D.epoch(sub.data, max_windows=max_w)
    mu = w.mean((0, 2), keepdims=True)
    sd = w.std((0, 2), keepdims=True) + 1e-6
    return ((w - mu) / sd).astype(np.float32)


def train_net(X, y, epochs=EPOCHS, subspace=20, spatial=40, lr=1e-3, seed=SEED,
              record=None):
    """Train a deep SPD net (global SPD-BN = S4) on window-level (X, y)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SPDNet(n_ch=X.shape[1], domain_bn=False, subspace=subspace,
                 spatial_filters=spatial, device=DEVICE, spd_device="cpu")
    opt = RiemannianAdam(net.parameters(), lr=lr, weight_decay=1e-4, stabilize=10)
    cnt = np.bincount(y, minlength=2)
    weight = torch.tensor(cnt.sum() / (2 * np.clip(cnt, 1, None)), dtype=torch.double)
    lossfn = torch.nn.CrossEntropyLoss(weight=weight)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    history = []
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(y))
        tot = 0.0
        for i in range(0, len(y), 128):
            bi = order[i:i + 128]
            opt.zero_grad()
            loss = lossfn(net(Xt[bi]), yt[bi])
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        if record is not None and (ep + 1) in record:
            history.append((ep + 1, tot / len(y)))
    return net, tot / len(y), history


@torch.no_grad()
def net_proba(net, X):
    net.eval()
    return torch.softmax(net(torch.from_numpy(X)), dim=1)[:, 1].cpu().numpy()


@torch.no_grad()
def net_features(net, X):
    net.eval()
    l = net._pre_bn(torch.from_numpy(X))
    l = net.bn(l)
    return net.logeig(l).cpu().numpy()


def train_acc(net, X, y):
    return accuracy_score(y, (net_proba(net, X) >= 0.5).astype(int))


# --------------------------------------------------------------------------
def main():
    print(f"device={DEVICE}  windows/subj={MAXW}  epochs={EPOCHS}\n")
    subs = D.load_subjects(limit=10)               # 5 MDD + 5 HC
    wins = {s.sid: subj_windows(s) for s in subs}
    lab = {s.sid: s.label for s in subs}
    sids = [s.sid for s in subs]
    mdd = [s for s in sids if lab[s] == 1]
    hc = [s for s in sids if lab[s] == 0]
    train_sids = mdd[:3] + hc[:3]                  # 6 train subjects
    test_sids = mdd[3:] + hc[3:]                   # 4 test subjects
    print(f"subjects: {len(subs)} (train {train_sids}, test {test_sids})")

    def stack(sid_list):
        X = np.concatenate([wins[s] for s in sid_list])
        y = np.concatenate([np.full(len(wins[s]), lab[s]) for s in sid_list])
        g = np.concatenate([np.full(len(wins[s]), s) for s in sid_list])
        return X, y, g

    # ===== Test A: protocol =====
    print("\n" + "=" * 70 + "\n[A] PROTOCOL — window-mixed vs subject-independent\n" + "=" * 70)
    Xall, yall, gall = stack(sids)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(yall))
    cut = int(0.8 * len(yall))
    tr, te = perm[:cut], perm[cut:]
    netA, _, _ = train_net(Xall[tr], yall[tr])
    pmix = net_proba(netA, Xall[te])
    auc_mix = roc_auc_score(yall[te], pmix)
    acc_mix = accuracy_score(yall[te], (pmix >= 0.5).astype(int))

    Xtr, ytr, _ = stack(train_sids)
    netB, _, _ = train_net(Xtr, ytr)
    subj_scores, subj_lab = [], []
    for s in test_sids:
        subj_scores.append(net_proba(netB, wins[s]).mean())
        subj_lab.append(lab[s])
    auc_si = roc_auc_score(subj_lab, subj_scores)
    print(f"  window-mixed split  : window AUC={auc_mix:.3f}  acc={acc_mix:.3f}")
    print(f"  subject-independent : subject-level AUC={auc_si:.3f}  "
          f"(scores={[f'{v:.2f}' for v in subj_scores]} lab={subj_lab})")
    print(f"  >> if mixed >> subject-independent: literature 'success' is mostly "
          f"window-level leakage, not a better model.")

    # ===== Test B: identity =====
    print("\n" + "=" * 70 + "\n[B] IDENTITY — do features encode SUBJECT, not class?\n" + "=" * 70)
    feats, fy_sid, fy_cls = [], [], []
    for s in train_sids:
        f = net_features(netB, wins[s])
        feats.append(f)
        fy_sid.append(np.full(len(f), s))
        fy_cls.append(np.full(len(f), lab[s]))
    F = np.concatenate(feats); ysid = np.concatenate(fy_sid); ycls = np.concatenate(fy_cls)
    rp = rng.permutation(len(F)); c2 = int(0.7 * len(F)); a, b = rp[:c2], rp[c2:]
    id_acc = accuracy_score(ysid[b], LogisticRegression(max_iter=500)
                            .fit(F[a], ysid[a]).predict(F[b]))
    print(f"  subject-ID recoverable from features: acc={id_acc:.3f}  "
          f"(chance={1/len(train_sids):.3f})")
    print(f"  >> acc near 1.0 means capacity went to individual identity "
          f"(the dominant between-subject axis), not depression.")

    # ===== Test C: overfit curve =====
    print("\n" + "=" * 70 + "\n[C] OVERFIT — train-acc vs held-out AUC across epochs\n" + "=" * 70)
    print(f"  {'epoch':>5} {'train_loss':>11} {'train_acc':>10} {'test_subjAUC':>13}")
    for ep in (1, 3, 5, 10, 20):
        net_c, loss_c, _ = train_net(Xtr, ytr, epochs=ep)
        tacc = train_acc(net_c, Xtr, ytr)
        ss = [net_proba(net_c, wins[s]).mean() for s in test_sids]
        auc_c = roc_auc_score(subj_lab, ss)
        print(f"  {ep:>5} {loss_c:>11.4f} {tacc:>10.3f} {auc_c:>13.3f}")
    print(f"  >> train_acc -> 1 while test AUC stays low = pure overfit; if test AUC "
          f"never rises, early stopping won't save the deep model here.")

    # ===== Test D: shallow vs deep =====
    print("\n" + "=" * 70 + "\n[D] SHALLOW vs DEEP — same data, is the DATA the problem?\n" + "=" * 70)
    sh = make_shallow("s2", max_windows=MAXW)
    train_subj_objs = [s for s in subs if s.sid in train_sids]
    sh.fit(train_subj_objs)
    sh_scores = [sh.predict_subject(s)[0] for s in subs if s.sid in test_sids]
    auc_sh = roc_auc_score(subj_lab, sh_scores)
    print(f"  S2 shallow (corr+tangent LR) subject AUC = {auc_sh:.3f}")
    print(f"  S4 deep SPD net              subject AUC = {auc_si:.3f}")
    print(f"  >> shallow > deep on identical data ⇒ signal EXISTS (data is fine); "
          f"the deep model overfits it away.")

    # ===== Test E: label-shuffle memorisation =====
    print("\n" + "=" * 70 + "\n[E] SHUFFLE — can S4 fit RANDOM labels to ~0 loss?\n" + "=" * 70)
    yreal = ytr.copy()
    # shuffle labels at the SUBJECT level (each subject keeps one constant fake label)
    fake = {s: v for s, v in zip(train_sids, rng.permutation([lab[s] for s in train_sids]))}
    _, _, gtr = stack(train_sids)
    yfake = np.array([fake[s] for s in gtr])
    _, loss_real, _ = train_net(Xtr, yreal)
    _, loss_fake, _ = train_net(Xtr, yfake)
    print(f"  final train loss  real labels = {loss_real:.4f}")
    print(f"  final train loss  RANDOM labels = {loss_fake:.4f}")
    print(f"  >> if both ≈ 0, then 'loss -> 0' just means the net memorises whatever "
          f"per-subject label it's given — it is NOT evidence of learning depression.")

    # ===== verdict =====
    print("\n" + "#" * 70)
    print("# READ TOGETHER:")
    print(f"#  A: mixed AUC {auc_mix:.2f} vs subject-indep {auc_si:.2f}")
    print(f"#  B: subject-ID acc {id_acc:.2f} (chance {1/len(train_sids):.2f})")
    print(f"#  D: shallow {auc_sh:.2f} vs deep {auc_si:.2f}")
    print(f"#  E: train loss real {loss_real:.3f} vs random {loss_fake:.3f}")
    print("#" * 70)


if __name__ == "__main__":
    main()
