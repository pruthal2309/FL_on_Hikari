"""
=============================================================================
  Federated Learning on ALLFLOWMETER_HIKARI_2021
  -----------------------------------------------
  Strategy : Label-Skewed Partitioning via Dirichlet Distribution
  Clients  : One per unique traffic_category label  (6 clients)
  Model    : RandomForest trained locally, aggregated via FedAvg (tree voting)
  Metrics  : Per-client accuracy + Global aggregated accuracy
=============================================================================
"""

# ── Imports ─────────────────────────────────────────────────────────────────
import os, warnings, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH    = "/mnt/user-data/uploads/ALLFLOWMETER_HIKARI2021.csv"
OUTPUT_DIR   = "/mnt/user-data/outputs"
DIRICHLET_ALPHA = 0.5          # lower → more label skew; try 0.1 for extreme
N_TREES_PER_CLIENT = 50        # trees per local RF model
TEST_SPLIT   = 0.20            # held-out global test ratio
TARGET_COL   = "traffic_category"  # multi-class label (6 classes)
DROP_COLS    = ["Unnamed: 0.1", "Unnamed: 0", "uid",
                "originh", "responh", "Label"]  # leaky / non-numeric

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 1 │ DATA PREPROCESSING")
print("="*70)

df = pd.read_csv(DATA_PATH)
print(f"  Raw dataset shape : {df.shape}")

# Drop identifier / redundant columns
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Encode target
le = LabelEncoder()
df["label_enc"] = le.fit_transform(df[TARGET_COL])
classes         = le.classes_          # ['Background','Benign','Bruteforce',...]
n_classes       = len(classes)
print(f"  Classes ({n_classes}): {list(classes)}")
print(f"  Class distribution:\n{df[TARGET_COL].value_counts().to_string()}\n")

# Encode remaining categoricals
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET_COL]
for c in cat_cols:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# Feature / label split
feature_cols = [c for c in df.columns if c not in [TARGET_COL, "label_enc"]]
X = df[feature_cols].values.astype(np.float32)
y = df["label_enc"].values.astype(int)

# Replace inf / nan
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Global train / test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42, stratify=y)

print(f"  Global train size : {len(X_train):,}  |  test size : {len(X_test):,}")

# Feature scaling (fit only on train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("  Preprocessing complete ✓")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — DIRICHLET LABEL-SKEWED PARTITIONING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 2 │ DIRICHLET LABEL-SKEWED PARTITIONING")
print("="*70)

# Number of clients = number of unique classes
n_clients = n_classes
client_names = [f"Client_{i}_{classes[i]}" for i in range(n_clients)]
print(f"  Clients : {n_clients}  (one per class)")
print(f"  Dirichlet α = {DIRICHLET_ALPHA}  "
      f"({'extreme skew' if DIRICHLET_ALPHA<0.3 else 'moderate skew'})\n")

def dirichlet_partition(X, y, n_clients, alpha, seed=42):
    """
    Dirichlet label-skew partition:
    For each class c, draw proportions p ~ Dir(alpha) over clients
    and assign a fraction p[k] of class-c samples to client k.
    """
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(n_clients)]
    unique_classes = np.unique(y)

    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        # Draw proportions from Dirichlet
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        # Convert to cumulative split points
        splits = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        chunks  = np.split(cls_idx, splits)
        for k, chunk in enumerate(chunks):
            client_indices[k].extend(chunk.tolist())

    # Shuffle each client's data
    for k in range(n_clients):
        rng.shuffle(client_indices[k])

    return client_indices

client_indices = dirichlet_partition(X_train, y_train, n_clients,
                                     alpha=DIRICHLET_ALPHA)

# ── Report partition stats ────────────────────────────────────────────────
partition_stats = []
print(f"  {'Client':<35} {'Samples':>8}  Class distribution (top-3)")
print(f"  {'-'*70}")
for k, (name, idx) in enumerate(zip(client_names, client_indices)):
    local_y   = y_train[idx]
    cls_count = Counter(local_y)
    top3      = ", ".join(
        f"{classes[c]}:{cls_count[c]}" for c, _ in cls_count.most_common(3))
    print(f"  {name:<35} {len(idx):>8}  {top3}")
    partition_stats.append({
        "client": name, "n_samples": len(idx),
        **{classes[c]: cls_count.get(c, 0) for c in range(n_classes)}
    })

# ── Visualise partition ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Dirichlet Label-Skewed Partition  (α={})".format(DIRICHLET_ALPHA),
             fontsize=14, fontweight="bold")

colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
ps_df  = pd.DataFrame(partition_stats).set_index("client")

# Stacked bar — class composition per client
(ps_df[list(classes)].div(ps_df["n_samples"], axis=0) * 100).plot(
    kind="bar", stacked=True, ax=axes[0], color=colors, edgecolor="white")
axes[0].set_title("Class Distribution per Client (%)")
axes[0].set_ylabel("Percentage")
axes[0].set_xticklabels([f"C{k}" for k in range(n_clients)], rotation=0)
axes[0].legend(classes, loc="upper right", fontsize=7)

# Sample size per client
ps_df["n_samples"].plot(kind="bar", ax=axes[1], color="steelblue",
                        edgecolor="white")
axes[1].set_title("Sample Count per Client")
axes[1].set_ylabel("# Samples")
axes[1].set_xticklabels([f"C{k}" for k in range(n_clients)], rotation=0)
for bar in axes[1].patches:
    axes[1].text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+200, f"{int(bar.get_height()):,}",
                 ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "partition_plot.png"), dpi=150)
plt.close()
print("\n  Partition plot saved ✓")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — LOCAL TRAINING (each client trains its own RandomForest)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 3 │ LOCAL CLIENT TRAINING")
print("="*70)

local_models   = []   # list of trained RF models
local_accs     = []   # per-client local accuracy

for k, (name, idx) in enumerate(zip(client_names, client_indices)):
    X_local = X_train[idx]
    y_local = y_train[idx]

    # Local train / val split
    if len(np.unique(y_local)) < 2:
        print(f"  [{k}] {name}: only 1 class — skipping")
        local_models.append(None)
        local_accs.append(None)
        continue

    X_loc_tr, X_loc_val, y_loc_tr, y_loc_val = train_test_split(
        X_local, y_local, test_size=0.15, random_state=42,
        stratify=y_local if len(np.unique(y_local))>1 else None)

    # Compute class weights to handle within-client imbalance
    cw = compute_class_weight("balanced",
                               classes=np.unique(y_loc_tr), y=y_loc_tr)
    cw_dict = dict(zip(np.unique(y_loc_tr), cw))

    # Train local RF
    rf = RandomForestClassifier(
        n_estimators=N_TREES_PER_CLIENT,
        max_depth=15,
        min_samples_leaf=5,
        class_weight=cw_dict,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_loc_tr, y_loc_tr)

    # Local validation accuracy
    y_pred_val = rf.predict(X_loc_val)
    acc = accuracy_score(y_loc_val, y_pred_val)
    local_accs.append(acc)
    local_models.append(rf)

    print(f"  [{k}] {name[:40]:<40}  n={len(X_loc_tr):>7}  "
          f"local_val_acc={acc:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — FEDERATED AGGREGATION (FedAvg via Soft-Voting Ensemble)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 4 │ GLOBAL MODEL AGGREGATION (FedAvg — Soft Voting)")
print("="*70)

"""
FedAvg for tree-based models:
We aggregate by weighted soft-voting: each client model contributes
predicted class probabilities, weighted proportionally to its local
dataset size (the standard FedAvg weight = n_k / N).
"""

valid_clients  = [(k, m) for k, m in enumerate(local_models) if m is not None]
n_valid        = len(valid_clients)
sizes          = np.array([len(client_indices[k]) for k, _ in valid_clients],
                           dtype=float)
fed_weights    = sizes / sizes.sum()   # FedAvg weights

print(f"  Aggregating {n_valid} client models with FedAvg weights:")
for i, (k, _) in enumerate(valid_clients):
    print(f"    Client {k} ({client_names[k][:30]}) → weight={fed_weights[i]:.4f}")

# ── Global inference: weighted average of probability predictions ──────────
proba_sum = np.zeros((len(X_test), n_classes))

for i, (k, model) in enumerate(valid_clients):
    # predict_proba may not cover all classes if local data is skewed
    local_proba = np.zeros((len(X_test), n_classes))
    raw_proba   = model.predict_proba(X_test)
    for j, cls_idx in enumerate(model.classes_):
        local_proba[:, cls_idx] = raw_proba[:, j]
    proba_sum += fed_weights[i] * local_proba

# Final global predictions
y_global_pred = np.argmax(proba_sum, axis=1)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — GLOBAL ACCURACY & REPORTING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  STEP 5 │ GLOBAL MODEL EVALUATION")
print("="*70)

global_acc = accuracy_score(y_test, y_global_pred)
print(f"\n  ✅ Global Federated Model Accuracy : {global_acc*100:.2f}%\n")

report = classification_report(
    y_test, y_global_pred,
    target_names=classes, digits=4, zero_division=0)
print(report)

# ── Per-client summary table ──────────────────────────────────────────────
print("\n  Per-Client Local Validation Accuracy:")
print(f"  {'#':<4} {'Client':<40} {'n_samples':>10} {'local_acc':>10}")
print(f"  {'-'*65}")
for k, (name, idx) in enumerate(zip(client_names, client_indices)):
    acc_str = f"{local_accs[k]:.4f}" if local_accs[k] is not None else "skipped"
    print(f"  {k:<4} {name:<40} {len(idx):>10} {acc_str:>10}")

print(f"\n  {'─'*65}")
print(f"  Global Federated Accuracy (test set)        : {global_acc*100:.2f}%")
print(f"  {'─'*65}\n")

# ── Confusion matrix plot ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
cm  = confusion_matrix(y_test, y_global_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=classes)
cmd.plot(ax=ax, colorbar=True, xticks_rotation=45, cmap="Blues")
ax.set_title(f"Global Federated Model — Confusion Matrix\n"
             f"(Dirichlet α={DIRICHLET_ALPHA}, Accuracy={global_acc*100:.2f}%)",
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# ── Per-client accuracy bar chart ─────────────────────────────────────────
valid_accs   = [(k, local_accs[k]) for k in range(n_clients)
                if local_accs[k] is not None]
ck_labels    = [f"C{k}" for k, _ in valid_accs]
ck_vals      = [a for _, a in valid_accs]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(ck_labels, ck_vals, color="steelblue", edgecolor="white")
ax.axhline(global_acc, color="crimson", linestyle="--", linewidth=1.8,
           label=f"Global Federated Acc={global_acc*100:.2f}%")
ax.set_ylim(0, 1.05)
ax.set_title("Per-Client Local Val Accuracy vs. Global Federated Accuracy",
             fontsize=12)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Client")
ax.legend()
for bar, val in zip(bars, ck_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "client_accuracy.png"), dpi=150)
plt.close()

print("  All plots saved to outputs/ ✓")
print("\n  Pipeline complete. ✓\n")
