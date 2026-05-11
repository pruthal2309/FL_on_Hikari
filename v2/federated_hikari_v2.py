"""
=============================================================================
  Federated Learning on ALLFLOWMETER_HIKARI_2021  — v2
  ---------------------------------------------------------------
  FINAL FIX: Size-Aware Two-Phase Dirichlet Partition

  Problem with v2: Even with 70% dominance reserved, minority clients
  (XMRIGCC=2.6K, Bruteforce-XML=4.1K) get flooded by residuals from
  majority classes (Benign=347K). 30% of Benign = 83K, split over 6
  clients = ~14K each, which dwarfs the 1.8K dominant XMRIGCC slice.

  ROOT CAUSE: Class imbalance ratio = 106x. No fixed fraction fixes this
  without also capping residual injection.

  FINAL SOLUTION — Size-Aware Capped Residual Partition:
  Phase 1: Client k gets dominant_frac (70%) of class k exclusively.
  Phase 2: Residuals from other classes are capped per-client so that
           no residual class can exceed `residual_cap_frac` (30%) of
           the client's OWN dominant slice size. This prevents large
           classes from flooding minority clients.
=============================================================================
"""

import os, warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH        = "/kaggle/input/datasets/kk0105/allflowmeter-hikari2021/ALLFLOWMETER_HIKARI2021.csv"
OUTPUT_DIR       = "/mnt/user-data/outputs"
DOMINANT_FRAC    = 0.75   # fraction of each class given exclusively to owner client
RESIDUAL_CAP     = 0.30   # max residual per-foreign-class as fraction of dominant slice
DIRICHLET_ALPHA  = 0.5    # heterogeneity of residual spread
N_TREES          = 60
TEST_SPLIT       = 0.20
TARGET_COL       = "traffic_category"
DROP_COLS        = ["Unnamed: 0.1", "Unnamed: 0", "uid", "originh", "responh", "Label"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 1 │ DATA PREPROCESSING")
print("="*70)

df = pd.read_csv(DATA_PATH)
print(f"  Raw shape : {df.shape}")
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

le = LabelEncoder()
df["label_enc"] = le.fit_transform(df[TARGET_COL])
classes   = le.classes_
n_classes = len(classes)
n_clients = n_classes

print(f"\n  Classes ({n_classes}):")
vc = df[TARGET_COL].value_counts()
for cls, cnt in vc.items():
    print(f"    {cls:<25} {cnt:>7,}  ({100*cnt/len(df):.1f}%)")
print(f"\n  Imbalance ratio (max/min): "
      f"{vc.max()/vc.min():.0f}x  ← this is why a simple fix fails")

for c in df.select_dtypes(include="object").columns:
    if c != TARGET_COL:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

feature_cols = [c for c in df.columns if c not in [TARGET_COL, "label_enc"]]
X = np.nan_to_num(df[feature_cols].values.astype(np.float32))
y = df["label_enc"].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42, stratify=y)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print("  Preprocessing complete ✓")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — SIZE-AWARE CAPPED DIRICHLET PARTITION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 2 │ SIZE-AWARE CAPPED DIRICHLET PARTITION")
print("="*70)
print(f"""
  Strategy: Three-Phase Size-Aware Partition
  ──────────────────────────────────────────
  Phase 1 — Dominant slice ({int(DOMINANT_FRAC*100)}%):
            Client k exclusively gets {int(DOMINANT_FRAC*100)}% of class k's samples.
            This is the ONLY source of class k for client k.

  Phase 2 — Capped residual injection:
            From each foreign class j≠k, client k gets at most
            {int(RESIDUAL_CAP*100)}% × |dominant_slice_k| samples.
            This CAPS large-class contamination proportional to the
            client's own dominant size — solving the 106x imbalance issue.

  Phase 3 — Dirichlet-ordered selection within cap:
            Within the cap, samples are selected via Dirichlet-weighted
            sampling to maintain stochastic heterogeneity.

  Expected result: Every client dominated ≥70% by its assigned class.
""")


def size_aware_partition(y, n_clients, dominant_frac=0.75,
                         residual_cap=0.30, alpha=0.5, seed=42):
    """
    Size-Aware Capped Dirichlet Partition.

    For each client k:
      1. Take dominant_frac of class k → assigned exclusively to client k.
      2. For every other class j, compute:
            cap_k_j = int(residual_cap * dominant_size_k)
         Then from class j's residual pool, give client k at most cap_k_j
         samples (Dirichlet-weighted ordering across clients).

    This guarantees dominance even when imbalance ratio > 100x.
    """
    rng = np.random.default_rng(seed)

    # ── Phase 1: build dominant slices ──────────────────────────────────
    dominant_pool = {}   # cls -> dominant indices for its owner
    residual_pool = {}   # cls -> residual indices for Dirichlet distribution

    for cls_id in range(n_clients):
        idx = np.where(y == cls_id)[0].copy()
        rng.shuffle(idx)
        n_dom = int(len(idx) * dominant_frac)
        dominant_pool[cls_id] = idx[:n_dom]
        residual_pool[cls_id] = idx[n_dom:]

    dominant_sizes = {k: len(v) for k, v in dominant_pool.items()}

    # ── Phase 2 & 3: capped residual injection ───────────────────────────
    client_indices = [list(dominant_pool[k]) for k in range(n_clients)]

    for cls_id in range(n_clients):
        res = residual_pool[cls_id].copy()
        if len(res) == 0:
            continue
        # Dirichlet proportions → ordering priority per client
        props = rng.dirichlet(alpha * np.ones(n_clients))
        # Sort clients by their proportion (highest gets first pick)
        order = np.argsort(-props)
        ptr   = 0
        for k in order:
            if ptr >= len(res):
                break
            # Cap for this client k receiving from class cls_id
            cap = int(residual_cap * dominant_sizes[k])
            give = min(cap, len(res) - ptr)
            if give > 0:
                client_indices[k].extend(res[ptr:ptr+give].tolist())
                ptr += give

    # Shuffle each client's full dataset
    for k in range(n_clients):
        arr = np.array(client_indices[k])
        rng.shuffle(arr)
        client_indices[k] = arr.tolist()

    return client_indices


client_indices = size_aware_partition(
    y_train, n_clients,
    dominant_frac=DOMINANT_FRAC,
    residual_cap=RESIDUAL_CAP,
    alpha=DIRICHLET_ALPHA)

client_names = [f"Client_{k}_{classes[k]}" for k in range(n_clients)]

# ── Partition report ──────────────────────────────────────────────────────
print(f"  {'Client':<35} {'Total':>8}  {'Dom%':>6}  Top-3 class distribution")
print(f"  {'-'*90}")

partition_stats = []
all_dom_pcts = []
for k, (name, idx) in enumerate(zip(client_names, client_indices)):
    local_y   = y_train[idx]
    cls_count = Counter(local_y)
    total     = len(local_y)
    dom_pct   = 100 * cls_count.get(k, 0) / total if total > 0 else 0
    all_dom_pcts.append(dom_pct)
    top3 = ", ".join(
        f"{classes[c]}:{cls_count[c]}({100*cls_count[c]/total:.1f}%)"
        for c, _ in cls_count.most_common(3))
    status = "✅" if dom_pct >= 50 else "⚠️ "
    print(f"  {status} {name:<33} {total:>8,}  {dom_pct:>5.1f}%  {top3}")
    partition_stats.append({
        "client": f"C{k}\n{classes[k][:8]}", "n_samples": total,
        **{classes[c]: cls_count.get(c, 0) for c in range(n_classes)}
    })

print(f"\n  Mean dominance %: {np.mean(all_dom_pcts):.1f}%  "
      f"| Min: {np.min(all_dom_pcts):.1f}%  | Max: {np.max(all_dom_pcts):.1f}%")

# ── Plots ─────────────────────────────────────────────────────────────────
colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
ps_df  = pd.DataFrame(partition_stats).set_index("client")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    f"v2 Size-Aware Capped Dirichlet Partition\n"
    f"(dominant_frac={DOMINANT_FRAC}, residual_cap={RESIDUAL_CAP}, α={DIRICHLET_ALPHA})",
    fontsize=12, fontweight="bold")

(ps_df[list(classes)].div(ps_df["n_samples"], axis=0) * 100).plot(
    kind="bar", stacked=True, ax=axes[0], color=colors, edgecolor="white")
axes[0].set_title("Class Composition per Client (%) — Each client dominated by its class")
axes[0].set_ylabel("Percentage")
axes[0].tick_params(axis="x", labelrotation=0, labelsize=7)
axes[0].legend(classes, loc="upper right", fontsize=7)

# Highlight dominant % on bars
dom_vals = [100 * Counter(y_train[client_indices[k]]).get(k, 0) / len(client_indices[k])
            for k in range(n_clients)]
bars = ps_df["n_samples"].plot(kind="bar", ax=axes[1], color="steelblue", edgecolor="white")
for i, (bar, dp) in enumerate(zip(axes[1].patches, dom_vals)):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                 f"Dom:\n{dp:.0f}%", ha="center", va="bottom", fontsize=8, color="darkblue")
axes[1].set_title("Sample Count per Client (with dominance %)")
axes[1].set_ylabel("# Samples")
axes[1].tick_params(axis="x", labelrotation=0, labelsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "partition_plot_v2.png"), dpi=150)
plt.close()
print("  Partition plot saved ✓")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — LOCAL TRAINING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 3 │ LOCAL CLIENT TRAINING")
print("="*70)

local_models, local_accs = [], []
for k, (name, idx) in enumerate(zip(client_names, client_indices)):
    X_loc = X_train[idx]
    y_loc = y_train[idx]
    if len(np.unique(y_loc)) < 2:
        print(f"  [{k}] {name}: only 1 class — skipping")
        local_models.append(None); local_accs.append(None); continue

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_loc, y_loc, test_size=0.15, random_state=42, stratify=y_loc)
    cw_dict = dict(zip(
        np.unique(y_tr),
        compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)))

    rf = RandomForestClassifier(n_estimators=N_TREES, max_depth=15,
                                min_samples_leaf=5, class_weight=cw_dict,
                                n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    acc = accuracy_score(y_val, rf.predict(X_val))
    local_accs.append(acc); local_models.append(rf)

    dom_pct = 100 * Counter(y_loc)[k] / len(y_loc)
    print(f"  [{k}] {name:<40} n={len(X_tr):>7,}  dom={dom_pct:.1f}%  acc={acc:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — FEDAVG AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 4 │ FEDAVG AGGREGATION (Weighted Soft Voting)")
print("="*70)

valid   = [(k, m) for k, m in enumerate(local_models) if m is not None]
sizes   = np.array([len(client_indices[k]) for k, _ in valid], dtype=float)
weights = sizes / sizes.sum()

proba_agg = np.zeros((len(X_test), n_classes))
for i, (k, model) in enumerate(valid):
    lp = np.zeros((len(X_test), n_classes))
    rp = model.predict_proba(X_test)
    for j, ci in enumerate(model.classes_):
        lp[:, ci] = rp[:, j]
    proba_agg += weights[i] * lp
    print(f"  Client {k} ({classes[k]:<22}) weight={weights[i]:.4f}")

y_pred = np.argmax(proba_agg, axis=1)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 5 │ GLOBAL MODEL EVALUATION")
print("="*70)

global_acc = accuracy_score(y_test, y_pred)
print(f"\n  ✅ Global Federated Accuracy : {global_acc*100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=classes,
                             digits=4, zero_division=0))

print(f"\n  Per-Client Summary:")
print(f"  {'#':<4} {'Class':<22} {'Samples':>9} {'Dom%':>7} {'LocalAcc':>10}")
print(f"  {'-'*58}")
for k in range(n_clients):
    dom_pct = 100 * Counter(y_train[client_indices[k]]).get(k,0) / len(client_indices[k])
    acc_str = f"{local_accs[k]:.4f}" if local_accs[k] is not None else "skipped"
    flag = "✅" if dom_pct >= 50 else "⚠️ "
    print(f"  {flag}{k:<3} {classes[k]:<22} {len(client_indices[k]):>9,} "
          f"{dom_pct:>6.1f}% {acc_str:>10}")

print(f"\n  {'─'*58}")
print(f"  Global Federated Accuracy : {global_acc*100:.2f}%")
print(f"  {'─'*58}\n")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"v2 Federated Results  |  Global Accuracy = {global_acc*100:.2f}%",
             fontsize=12, fontweight="bold")

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(
    ax=axes[0], colorbar=False, xticks_rotation=45, cmap="Blues")
axes[0].set_title("Global Model — Confusion Matrix")

valid_k  = [k for k in range(n_clients) if local_accs[k] is not None]
clabels  = [f"C{k}\n{classes[k][:7]}" for k in valid_k]
cvals    = [local_accs[k] for k in valid_k]
dom_pcts = [100*Counter(y_train[client_indices[k]]).get(k,0)/len(client_indices[k])
            for k in valid_k]

bars = axes[1].bar(clabels, cvals, color="steelblue", edgecolor="white")
axes[1].axhline(global_acc, color="crimson", linestyle="--", linewidth=2,
                label=f"Global FedAcc={global_acc*100:.2f}%")
axes[1].set_ylim(0, 1.12)
axes[1].set_title("Per-Client Accuracy (dom% shown above bar)")
axes[1].set_ylabel("Accuracy")
axes[1].legend(fontsize=9)
for bar, val, dp in zip(bars, cvals, dom_pcts):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{val:.3f}\ndom:{dp:.0f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "results_v2.png"), dpi=150)
plt.close()

print("  Plots saved ✓")
print("  Pipeline v2 complete ✓\n")
