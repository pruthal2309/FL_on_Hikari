"""
federated_hikari_v4_fixed.py

Purpose: reproduce the DOCUMENTED v4 methodology (paper Alg. 2 + GitHub README)
before layering any new FL algorithm on top. This is a diagnostic/reproduction
run, not yet the "80%+" experiment.

Three confirmed bugs vs. the documented v4 pipeline are fixed here, each
tagged [FIX n] so you can find and revert individually:

  [FIX 1] Stratified class capping (Table XI) was missing entirely.
          Benign/Background/Probing are now capped before client partitioning;
          Bruteforce/Bruteforce-XML/XMRIGCC keep 100% of their samples.
  [FIX 2] WeightedRandomSampler was missing from local training; only the
          loss was reweighted. Both are now used together, as in Algorithm 2.
  [FIX 3] Centralized warm-start pretraining is NOT part of the documented
          v4 method (it appears nowhere in Alg. 2 or the README). It is now
          OFF by default (WARMUP_EPOCHS = 0) and clearly labeled as a
          diagnostic-only path if you re-enable it.

Also restored to match the documented v4 config exactly:
  FL_ROUNDS=8, LOCAL_EPOCHS=6, DROPOUT_RATE=0.30 (0.30->0.25->0.20->0.10),
  FEDPROX_MU=0 by default (v4 did not use FedProx; the paper says so
  explicitly in Sec. IV-E). Turn it on later as a SEPARATE, isolated
  experiment once this baseline is confirmed to reproduce.

Added: per-round client-update L2 norms + pairwise cosine similarity, so
client drift is something you can see in numbers instead of assume.

Everything else (model architecture, LayerNorm, weighted CE, OneCycleLR,
AdamW, gradient clipping, size-aware capped Dirichlet partitioning,
best-checkpoint-by-macroF1 selection) is unchanged from your script.

NOTE: this still trains 6 clients (one per class), matching your actual
GitHub repo. The paper's claim of "5 clients, Benign+Background merged"
does not match the repo's own README or your script -- flagging that
mismatch is Section 4/6 of the audit; it is NOT fixed here because
changing client count is a partitioning-design decision, not a bug fix,
and should be tested as its own controlled experiment (Section 6 of your
brief), not silently folded into a bug-fix pass.
"""

import os, warnings, random, copy, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

DATA_PATH  = "/kaggle/input/datasets/kk0105/allflowmeter-hikari2021/ALLFLOWMETER_HIKARI2021.csv"
OUTPUT_DIR = Path("/kaggle/working/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL = "traffic_category"

# ---------------- [FIX 1] Stratified class caps (paper Table XI) ----------------
# Applied to TRAINING data only, after the train/test split, before client
# partitioning -- test set is never touched.
CLASS_CAPS = {
    "Benign": 30000,
    "Background": 20000,
    "Probing": 10000,
    # Bruteforce, Bruteforce-XML, XMRIGCC CryptoMiner: no cap (keep all)
}

USE_ALL_FEATURES = True
VARIANCE_THRESH  = 1e-8
CORR_THRESH      = 0.985

RUN_CENTRALIZED_BASELINE = True     # MLP centralized ceiling, diagnostic only
PARTITION_MODE = "dirichlet"        # "dirichlet" (non-IID) or "iid" (diagnostic ceiling)

DOMINANT_FRAC   = 0.75
RESIDUAL_CAP    = 0.30
DIRICHLET_ALPHA = 0.5

# ---------------- restored to documented v4 config ----------------
FL_ROUNDS       = 8
LOCAL_EPOCHS    = 6
BATCH_SIZE      = 256
LR              = 2e-3
LR_ROUND_DECAY  = 0.97
WEIGHT_DECAY    = 1e-4
TEST_SPLIT      = 0.20
HIDDEN_DIMS     = [512, 256, 128, 64]
DROPOUT_RATE    = 0.30

# [FIX 3] FedProx OFF by default -- v4's documented 95% number did not use it.
# Re-enable only as an isolated, separately-reported experiment.
FEDPROX_MU               = 0.0

# [FIX 3] Centralized warm-start OFF by default -- not part of Algorithm 2.
# Set > 0 only if you explicitly want to re-run it as a labeled diagnostic.
WARMUP_EPOCHS            = 0
WARMUP_SAMPLES_PER_CLASS = 4000

BASELINE_EPOCHS = 25

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("=" * 72)
print("  PHASE 1 -- FEATURE PREP")
print("=" * 72)

print("\n[1/4] Loading dataset...")
df = pd.read_csv(DATA_PATH)
drop_fl = ["Unnamed: 0.1", "Unnamed: 0", "uid", "originh", "responh", "Label"]
df.drop(columns=[c for c in drop_fl if c in df.columns], inplace=True)
df = df.reset_index(drop=True)
print(f"      Shape : {df.shape}")

le        = LabelEncoder()
df["label_enc"] = le.fit_transform(df[TARGET_COL])
classes   = le.classes_
n_classes = len(classes)
n_clients = n_classes

for c in df.columns:
    if df[c].dtype == object and c != TARGET_COL:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

all_feature_cols = [c for c in df.columns if c not in [TARGET_COL, "label_enc"]]
print(f"      Total numeric feature columns available: {len(all_feature_cols)}")

print("\n[2/4] Selecting feature set...")
if USE_ALL_FEATURES:
    feature_cols = list(all_feature_cols)
    print(f"      Using ALL {len(feature_cols)} features directly.")
else:
    X_raw = np.nan_to_num(df[all_feature_cols].values.astype(np.float32))
    variances = X_raw.var(axis=0)
    keep_var  = variances > VARIANCE_THRESH
    kept_after_var = [f for f, k in zip(all_feature_cols, keep_var) if k]
    Xv = X_raw[:, keep_var]
    corr = np.corrcoef(Xv, rowvar=False)
    to_drop = set()
    n = corr.shape[0]
    for i in range(n):
        if i in to_drop: continue
        for j in range(i + 1, n):
            if j in to_drop: continue
            if abs(corr[i, j]) > CORR_THRESH:
                to_drop.add(j)
    feature_cols = [f for idx, f in enumerate(kept_after_var) if idx not in to_drop]
    print(f"      Remaining features: {len(feature_cols)} / {len(all_feature_cols)}")

input_dim = len(feature_cols)

print("\n[3/4] Train/test split (test set untouched by everything downstream)...")
X = np.nan_to_num(df[feature_cols].values.astype(np.float32))
y = df["label_enc"].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y)

# [FIX 1] Apply stratified class caps to the TRAINING split only.
print("\n[4/4] Applying stratified class caps (train split only)...")
rng_cap = np.random.default_rng(SEED)
keep_idx = []
for k, cname in enumerate(classes):
    cls_idx = np.where(y_train == k)[0]
    cap = CLASS_CAPS.get(cname, None)
    if cap is not None and len(cls_idx) > cap:
        cls_idx = rng_cap.choice(cls_idx, size=cap, replace=False)
        print(f"      {cname:<20} capped {len(np.where(y_train == k)[0]):>7,} -> {cap:>7,}")
    else:
        print(f"      {cname:<20} kept full: {len(cls_idx):>7,}")
    keep_idx.append(cls_idx)
keep_idx = np.concatenate(keep_idx)
rng_cap.shuffle(keep_idx)
X_train, y_train = X_train[keep_idx], y_train[keep_idx]

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

X_test_t  = torch.from_numpy(X_test).to(DEVICE)
y_test_np = y_test.copy()

print(f"\n  Train (post-cap) : {len(X_train):,}  |  Test : {len(X_test):,}  |  Device : {DEVICE}")
print(f"  Feature dim : {input_dim}")


# =====================================================================
#  MODEL / SHARED UTILITIES
# =====================================================================

class TrafficMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, dropout=0.30):
        super().__init__()
        drops  = [dropout, dropout*0.833, dropout*0.667, dropout*0.333]  # ~0.30/0.25/0.20/0.10
        blocks = []
        in_d   = input_dim
        for h_d, dr in zip(hidden_dims, drops):
            blocks += [nn.Linear(in_d, h_d), nn.LayerNorm(h_d), nn.GELU(), nn.Dropout(dr)]
            in_d = h_d
        self.backbone = nn.Sequential(*blocks)
        self.head     = nn.Linear(in_d, n_classes)
        self.residual = nn.Linear(input_dim, in_d)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.backbone(x) + self.residual(x))


def get_model():
    return TrafficMLP(input_dim, HIDDEN_DIMS, n_classes, DROPOUT_RATE).to(DEVICE)

def get_weights(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}

def set_weights(m, w):
    m.load_state_dict(w, strict=True)
    return m

def flat_weights(w):
    """Flatten a state_dict into one vector, for norm/cosine diagnostics."""
    return torch.cat([v.float().flatten() for v in w.values()])

def make_weighted_ce(y_local):
    cnt    = np.bincount(y_local, minlength=n_classes).astype(float)
    cnt    = np.where(cnt == 0, 1e-9, cnt)
    weight = torch.tensor(cnt.sum() / cnt / n_classes, dtype=torch.float32).to(DEVICE)
    return nn.CrossEntropyLoss(weight=weight)

def make_weighted_sampler(y_local):
    """[FIX 2] Balances mini-batches, on top of (not instead of) weighted CE."""
    cnt = np.bincount(y_local, minlength=n_classes).astype(float)
    cnt = np.where(cnt == 0, 1e-9, cnt)
    sample_w = 1.0 / cnt[y_local]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(y_local), replacement=True)

def local_train(model, X_local, y_local, n_epochs, lr, global_params=None, mu=0.0):
    model.train()
    sampler   = make_weighted_sampler(y_local)   # [FIX 2]
    loader    = DataLoader(
        TensorDataset(torch.from_numpy(X_local).float(),
                      torch.from_numpy(y_local).long()),
        batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)
    criterion = make_weighted_ce(y_local)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(loader), epochs=n_epochs,
        pct_start=0.3, anneal_strategy="cos")
    epoch_losses = []
    for _ in range(n_epochs):
        run, nb = 0.0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            if global_params is not None and mu > 0:
                prox = sum(((p - gp) ** 2).sum()
                           for p, gp in zip(model.parameters(), global_params))
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            run += loss.item(); nb += 1
        epoch_losses.append(run / max(nb, 1))
    return model, epoch_losses

@torch.no_grad()
def evaluate_model(model, X_t, y_np):
    model.eval()
    preds = []
    for (Xb,) in DataLoader(TensorDataset(X_t), batch_size=2048):
        preds.append(model(Xb.to(DEVICE)).argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds)
    acc  = accuracy_score(y_np, y_pred)
    f1m  = f1_score(y_np, y_pred, average="macro", zero_division=0)
    bacc = balanced_accuracy_score(y_np, y_pred)
    return acc, f1m, bacc, y_pred


# =====================================================================
#  DIAGNOSTIC -- CENTRALIZED BASELINE (labeled diagnostic, not the FL result)
# =====================================================================
if RUN_CENTRALIZED_BASELINE:
    print("\n" + "=" * 72)
    print(f"  DIAGNOSTIC ONLY -- CENTRALIZED CEILING ({BASELINE_EPOCHS} epochs, NOT federated)")
    print("=" * 72)
    t0 = time.time()
    baseline_model = get_model()
    baseline_model, base_losses = local_train(
        baseline_model, X_train, y_train, BASELINE_EPOCHS, lr=LR)
    base_acc, base_f1, base_bacc, base_pred = evaluate_model(baseline_model, X_test_t, y_test_np)
    print(f"  Centralized ceiling -- Acc={base_acc*100:.2f}%  MacroF1={base_f1*100:.2f}%  "
          f"BalAcc={base_bacc*100:.2f}%  [{time.time()-t0:.1f}s]")
    print("  --> This is NOT the FL result. It exists only to bound how much of the")
    print("      gap to 95% is 'features/model capacity' vs. 'the federation itself'.")


# =====================================================================
#  PHASE 2 -- FEDERATED LEARNING
# =====================================================================

print("\n" + "=" * 72)
print(f"  PHASE 2 -- FEDERATED LEARNING (partition_mode={PARTITION_MODE}, "
      f"clients={n_clients}, warmup_epochs={WARMUP_EPOCHS}, fedprox_mu={FEDPROX_MU})")
print("=" * 72)

def size_aware_partition(y, n_clients, dominant_frac=0.75,
                         residual_cap=0.30, alpha=0.5, seed=42):
    rng      = np.random.default_rng(seed)
    dom_pool, res_pool = {}, {}
    for k in range(n_clients):
        idx = np.where(y == k)[0].copy()
        rng.shuffle(idx)
        nd = int(len(idx) * dominant_frac)
        dom_pool[k] = idx[:nd]
        res_pool[k] = idx[nd:]
    dom_sizes  = {k: len(v) for k, v in dom_pool.items()}
    client_idx = [list(dom_pool[k]) for k in range(n_clients)]
    for cls in range(n_clients):
        res = res_pool[cls].copy()
        if len(res) == 0:
            continue
        order = np.argsort(-rng.dirichlet(alpha * np.ones(n_clients)))
        ptr   = 0
        for k in order:
            if ptr >= len(res):
                break
            give = min(int(residual_cap * dom_sizes[k]), len(res) - ptr)
            if give > 0:
                client_idx[k].extend(res[ptr:ptr+give].tolist())
                ptr += give
    for k in range(n_clients):
        arr = np.array(client_idx[k])
        rng.shuffle(arr)
        client_idx[k] = arr.tolist()
    return client_idx

def iid_partition(y, n_clients, seed=42):
    rng = np.random.default_rng(seed)
    idx_by_class = {k: np.where(y == k)[0].copy() for k in range(n_clients)}
    for k in idx_by_class:
        rng.shuffle(idx_by_class[k])
    client_idx = [[] for _ in range(n_clients)]
    for k, idx in idx_by_class.items():
        splits = np.array_split(idx, n_clients)
        for c in range(n_clients):
            client_idx[c].extend(splits[c].tolist())
    for c in range(n_clients):
        arr = np.array(client_idx[c])
        rng.shuffle(arr)
        client_idx[c] = arr.tolist()
    return client_idx

print("\n[1/3] Partitioning data...")
if PARTITION_MODE == "iid":
    client_indices = iid_partition(y_train, n_clients, seed=SEED)
else:
    client_indices = size_aware_partition(
        y_train, n_clients, DOMINANT_FRAC, RESIDUAL_CAP, DIRICHLET_ALPHA, seed=SEED)

client_names = [f"Client_{k}_{classes[k]}" for k in range(n_clients)]
COLORS       = plt.cm.tab10(np.linspace(0, 1, n_classes))

for k, (name, idx) in enumerate(zip(client_names, client_indices)):
    ly  = y_train[idx]
    cc  = Counter(ly)
    tot = len(ly)
    dp  = 100 * cc.get(k, 0) / tot
    print(f"  [{k}] {name:<40} n={tot:>6,}  dom={dp:.1f}%")

def fedavg_aggregate(weights_list, sizes):
    N   = sum(sizes)
    nrm = [s / N for s in sizes]
    agg = {}
    for key in weights_list[0]:
        agg[key] = sum(nrm[i] * weights_list[i][key].float()
                       for i in range(len(weights_list)))
    return agg

# [FIX 3] Warm-start is now opt-in and clearly labeled as a diagnostic if used.
global_model = get_model()
if WARMUP_EPOCHS > 0:
    print(f"\n[2/3] DIAGNOSTIC-ONLY centralized warm-start ({WARMUP_EPOCHS} epochs)...")
    print("      NOTE: this is centralized pretraining -- not part of the")
    print("      documented v4 methodology. Do not report the post-warmup")
    print("      number as an FL result.")
    warm_idx = []
    rng_warm = np.random.default_rng(SEED)
    for k in range(n_classes):
        cls_idx = np.where(y_train == k)[0]
        take = min(WARMUP_SAMPLES_PER_CLASS, len(cls_idx))
        warm_idx.extend(rng_warm.choice(cls_idx, size=take, replace=False).tolist())
    warm_idx = np.array(warm_idx)
    rng_warm.shuffle(warm_idx)
    X_warm, y_warm = X_train[warm_idx], y_train[warm_idx]
    global_model, warm_losses = local_train(global_model, X_warm, y_warm, WARMUP_EPOCHS, lr=LR)
    warm_acc, warm_f1, warm_bacc, _ = evaluate_model(global_model, X_test_t, y_test_np)
    print(f"  Post-warm-start (diagnostic) Acc={warm_acc*100:.2f}%  MacroF1={warm_f1*100:.2f}%")
else:
    print("\n[2/3] No warm-start (matches Algorithm 2: global model starts from random init).")
    warm_f1 = -1.0

global_weights = get_weights(global_model)
n_params = sum(p.numel() for p in global_model.parameters())
print(f"  Architecture : {input_dim}->512->256->128->64->{n_classes}  |  Params: {n_params:,}")

client_data       = [(X_train[np.array(client_indices[k])],
                      y_train[np.array(client_indices[k])])
                     for k in range(n_clients)]
round_global_acc  = []
round_global_f1   = []
round_global_bacc = []
client_round_acc  = defaultdict(list)
client_all_losses = defaultdict(list)
round_update_norms = []      # per round: dict client -> ||delta_k||
round_cosine_sims   = []     # per round: dict (i,j) -> cosine similarity

best_f1        = warm_f1
best_weights   = copy.deepcopy(global_weights)
best_round_idx = 0

print(f"\n[3/3] FL Training -- {FL_ROUNDS} rounds x {LOCAL_EPOCHS} local epochs "
      f"(FedProx mu={FEDPROX_MU})...")
t_total = time.time()

for fl_round in range(1, FL_ROUNDS + 1):
    t_rnd = time.time()
    round_lr = LR * (LR_ROUND_DECAY ** (fl_round - 1))
    print(f"\n  ROUND {fl_round}/{FL_ROUNDS}  (lr={round_lr:.2e})")
    rnd_weights, rnd_sizes = [], []
    global_params_ref = [p.detach().clone() for p in global_model.parameters()]
    global_flat = flat_weights(global_weights)

    deltas = {}
    for k in range(n_clients):
        X_local, y_local = client_data[k]
        local_m = get_model()
        set_weights(local_m, copy.deepcopy(global_weights))
        local_m, ep_losses = local_train(
            local_m, X_local, y_local, LOCAL_EPOCHS, lr=round_lr,
            global_params=global_params_ref, mu=FEDPROX_MU)
        X_lt        = torch.from_numpy(X_local).float().to(DEVICE)
        l_acc, _, _, _ = evaluate_model(local_m, X_lt, y_local)
        client_round_acc[k].append(l_acc)
        client_all_losses[k].extend(ep_losses)
        dp = 100 * Counter(y_local)[k] / len(y_local)
        print(f"    [{k}] {client_names[k]:<42} n={len(X_local):>6,}  "
              f"dom={dp:.0f}%  loss={ep_losses[-1]:.4f}  acc={l_acc:.4f}")
        local_w = get_weights(local_m)
        rnd_weights.append(local_w)
        rnd_sizes.append(len(X_local))
        deltas[k] = flat_weights(local_w) - global_flat   # for drift diagnostics

    # -------- client drift diagnostics --------
    norms = {k: float(d.norm().item()) for k, d in deltas.items()}
    round_update_norms.append(norms)
    cos = {}
    for i, j in combinations(range(n_clients), 2):
        num = torch.dot(deltas[i], deltas[j]).item()
        den = (deltas[i].norm().item() * deltas[j].norm().item() + 1e-12)
        cos[(i, j)] = num / den
    round_cosine_sims.append(cos)
    print(f"    [drift] update norms: " +
          ", ".join(f"C{k}={norms[k]:.2f}" for k in range(n_clients)))
    mean_cos = np.mean(list(cos.values()))
    print(f"    [drift] mean pairwise cosine similarity: {mean_cos:.3f} "
          f"(near 0 or negative => clients pulling in different directions)")

    global_weights = fedavg_aggregate(rnd_weights, rnd_sizes)
    set_weights(global_model, global_weights)
    g_acc, g_f1, g_bacc, _ = evaluate_model(global_model, X_test_t, y_test_np)
    round_global_acc.append(g_acc)
    round_global_f1.append(g_f1)
    round_global_bacc.append(g_bacc)
    if g_f1 > best_f1:
        best_f1 = g_f1
        best_weights = copy.deepcopy(global_weights)
        best_round_idx = fl_round
    print(f"  Global Round {fl_round}  Acc={g_acc*100:.2f}%  MacroF1={g_f1*100:.2f}%  "
          f"BalAcc={g_bacc*100:.2f}%  [{time.time()-t_rnd:.1f}s]")

total_time = time.time() - t_total
best_round = int(np.argmax(round_global_acc)) + 1 if round_global_acc else 0
print(f"\n  FL done in {total_time:.1f}s")
if round_global_acc:
    print(f"  Best acc: {max(round_global_acc)*100:.2f}% (R{best_round})")


print("\n[Final] Evaluation using BEST checkpoint (by macro-F1)...")
set_weights(global_model, best_weights)
final_acc, final_f1, final_bacc, y_pred = evaluate_model(global_model, X_test_t, y_test_np)
print(f"\n  Final Global Accuracy (best ckpt)   : {final_acc*100:.2f}%")
print(f"  Final Global Macro-F1 (best ckpt)   : {final_f1*100:.2f}%")
print(f"  Final Balanced Accuracy (best ckpt) : {final_bacc*100:.2f}%")
print(f"  Features used                       : {input_dim} / {len(all_feature_cols)}")
if RUN_CENTRALIZED_BASELINE:
    gap = (base_acc - final_acc) * 100
    print(f"  Centralized ceiling was             : {base_acc*100:.2f}%  (gap: {gap:+.2f} pts)")
print(classification_report(y_test_np, y_pred,
                             target_names=classes, digits=4, zero_division=0))

# -------- plots --------
LABEL = f"{input_dim} features | {n_clients} clients | partition={PARTITION_MODE} | fixed v4 repro"

fig, ax = plt.subplots(figsize=(12, 5))
rounds = list(range(1, FL_ROUNDS + 1))
ax.plot(rounds, [a*100 for a in round_global_acc], "o-", color="crimson", lw=2.5, ms=8,
        label=f"Global Acc (best={max(round_global_acc)*100:.2f}%)")
ax.plot(rounds, [a*100 for a in round_global_f1], "^-", color="navy", lw=2, ms=6,
        label=f"Global MacroF1 (best={max(round_global_f1)*100:.2f}%)")
if RUN_CENTRALIZED_BASELINE:
    ax.axhline(base_acc*100, color="green", ls="--", lw=2,
               label=f"Centralized ceiling ({base_acc*100:.2f}%)")
for k in range(n_clients):
    ax.plot(rounds, [a*100 for a in client_round_acc[k]], "s--", alpha=0.35,
            color=COLORS[k], ms=4, label=f"C{k}: {classes[k][:10]}")
ax.set_xlabel("FL Round"); ax.set_ylabel("Score (%)")
ax.set_title(f"FL vs Centralized Ceiling\n{LABEL}", fontsize=11, fontweight="bold")
ax.legend(fontsize=7, loc="lower right", ncol=2)
ax.grid(alpha=0.3); ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_round_accuracy_fixed.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 7))
cm = confusion_matrix(y_test_np, y_pred)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(
    ax=ax, colorbar=True, xticks_rotation=45, cmap="Blues")
ax.set_title(f"Federated NN -- Confusion Matrix (best ckpt)\n{LABEL} | "
             f"Acc={final_acc*100:.2f}% | MacroF1={final_f1*100:.2f}%",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_confusion_matrix_fixed.png", dpi=150); plt.close()

# drift diagnostic plot
fig, ax = plt.subplots(figsize=(10, 4))
for k in range(n_clients):
    ax.plot(rounds, [round_update_norms[r][k] for r in range(len(rounds))],
            "o-", color=COLORS[k], label=f"C{k}: {classes[k][:10]}")
ax.set_xlabel("FL Round"); ax.set_ylabel("||delta_k|| (local update L2 norm)")
ax.set_title("Client update norms per round (drift indicator)")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/client_drift_norms.png", dpi=150); plt.close()

print("\n  Plots saved.")
print(f"\n{'='*72}")
print(f"  PIPELINE COMPLETE (v4 reproduction w/ 3 confirmed bugs fixed)")
print(f"  FL Final Acc     : {final_acc*100:.2f}%  (checkpoint round {best_round_idx})")
print(f"  FL Final MacroF1 : {final_f1*100:.2f}%")
print(f"{'='*72}")
print("\nNEXT STEPS (do these one at a time, not all at once -- Section 36):")
print("  1. Run this script as-is. This is your new honest baseline.")
print("  2. Compare it to the ORIGINAL script's ~70% and to v0's 68% centralized ceiling.")
print("  3. Only after this baseline is understood, test ONE of: FedProx (tune mu),")
print("     n_clients=5 with Benign+Background merged, or a server optimizer (FedAdam).")
print("     Report each in isolation before stacking them.")
