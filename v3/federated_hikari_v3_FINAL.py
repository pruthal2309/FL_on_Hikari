"""
=============================================================================
  FEDERATED LEARNING — NEURAL NETWORK + TRUE FedAvg WEIGHT AGGREGATION
  Dataset : ALLFLOWMETER_HIKARI_2021
  Author  : FL Engineer
  Version : v4 — Neural Network Edition
=============================================================================

  ARCHITECTURE OVERVIEW
  ─────────────────────
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    FEDERATED LEARNING SYSTEM                        │
  │                                                                     │
  │   Client 0          Client 1    ...    Client 5                    │
  │  [LocalNN]          [LocalNN]          [LocalNN]                   │
  │   Trains on         Trains on          Trains on                   │
  │   local data        local data         local data                  │
  │       │                 │                  │                        │
  │       └────────── ▼ Send Weights ──────────┘                       │
  │                  [GLOBAL SERVER]                                    │
  │              FedAvg: W_global = Σ (n_k/N) × W_k                   │
  │                  ↓ Broadcast ↓                                      │
  │       ┌─────────────────────────────────┐                          │
  │  Next Round: clients load global weights → fine-tune locally       │
  │       └─────────────────────────────────┘                          │
  └─────────────────────────────────────────────────────────────────────┘

  KEY DESIGN DECISIONS
  ────────────────────
  1. Local Model   : Deep MLP with BatchNorm + Dropout + GELU activations
                     Input(81) → 512 → 256 → 128 → 64 → Output(6)
  2. Loss Function : Focal Loss (handles 106x class imbalance far better
                     than CrossEntropy — penalises easy examples less)
  3. Optimizer     : AdamW with CosineAnnealingLR per client
  4. FedAvg        : TRUE weight averaging — Σ(n_k/N)×θ_k per layer tensor
                     (not soft-voting — actual model parameter aggregation)
  5. FL Rounds     : Multiple communication rounds; each round clients
                     start from global weights → local fine-tune → aggregate
  6. Partition     : Size-Aware Capped Dirichlet (v3 fix retained)
  7. Evaluation    : Per-round global accuracy tracked + full final report
=============================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os, warnings, random, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hyperparameters & Config ─────────────────────────────────────────────────
DATA_PATH       = "/kaggle/input/datasets/kk0105/allflowmeter-hikari2021/ALLFLOWMETER_HIKARI2021.csv"
OUTPUT_DIR      = "/mnt/user-data/outputs"
TARGET_COL      = "traffic_category"
DROP_COLS       = ["Unnamed: 0.1", "Unnamed: 0", "uid", "originh", "responh", "Label"]

# Partition
DOMINANT_FRAC   = 0.75
RESIDUAL_CAP    = 0.30
DIRICHLET_ALPHA = 0.5

# FL Training
FL_ROUNDS       = 5       # number of global communication rounds
LOCAL_EPOCHS    = 5       # local epochs per round per client
BATCH_SIZE      = 512
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
TEST_SPLIT      = 0.20

# NN Architecture
HIDDEN_DIMS     = [512, 256, 128, 64]
DROPOUT_RATE    = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# MODULE 1 — NEURAL NETWORK ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════

class TrafficClassifierNN(nn.Module):
    """
    Deep MLP for network traffic classification.

    Architecture:
        Input(81) → [Linear→BN→GELU→Dropout] × 4 layers → Output(6)

    Design choices:
    • BatchNorm   : stabilises training on heterogeneous FL client data
    • GELU        : smoother gradient flow vs ReLU for tabular data
    • Dropout     : regularisation — critical since clients have skewed data
    • Residual    : skip connection between layer pairs to ease optimisation
    """

    def __init__(self, input_dim: int, hidden_dims: list, n_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.input_dim   = input_dim
        self.n_classes   = n_classes

        layers = []
        in_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout if i < len(hidden_dims)-1 else dropout*0.5)
            ]
            in_dim = h_dim

        self.backbone   = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, n_classes)

        # Residual projection for skip connection (input → last hidden)
        self.residual_proj = nn.Linear(input_dim, hidden_dims[-1])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)          # skip connection
        out      = self.backbone(x)
        out      = out + residual                  # add residual
        return self.classifier(out)               # raw logits


# ════════════════════════════════════════════════════════════════════════════
# MODULE 2 — FOCAL LOSS
# ════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class imbalanced classification.
    FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    γ (gamma) : focusing parameter. γ=0 → standard CE. γ=2 (default) puts
                more weight on hard misclassified examples, ignoring easy ones.
    α (alpha) : per-class weight tensor to handle class frequency imbalance.

    This is CRITICAL here: imbalance ratio = 106x. Standard CrossEntropy
    would collapse to predicting only Benign (62.6% of data).
    """

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # shape: (n_classes,)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss  = F.cross_entropy(logits, targets, reduction="none",
                                   weight=self.alpha)
        pt       = torch.exp(-ce_loss)             # probability of correct class
        focal_w  = (1.0 - pt) ** self.gamma
        loss     = (focal_w * ce_loss).mean()
        return loss


# ════════════════════════════════════════════════════════════════════════════
# MODULE 3 — DATA PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*72)
print("  STEP 1 │ DATA PREPROCESSING")
print("="*72)

df = pd.read_csv(DATA_PATH)
print(f"  Raw shape : {df.shape}")
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

le        = LabelEncoder()
df["label_enc"] = le.fit_transform(df[TARGET_COL])
classes   = le.classes_
n_classes = len(classes)
n_clients = n_classes

print(f"\n  Target classes ({n_classes}):")
vc = df[TARGET_COL].value_counts()
for cls, cnt in vc.items():
    bar = "█" * int(30 * cnt / vc.max())
    print(f"    {cls:<25} {cnt:>7,}  {bar}")
print(f"\n  Imbalance ratio : {vc.max()//vc.min()}x  → Focal Loss will handle this")

# Encode remaining object columns
for c in df.select_dtypes(include="object").columns:
    if c != TARGET_COL:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

feature_cols = [c for c in df.columns if c not in [TARGET_COL, "label_enc"]]
input_dim    = len(feature_cols)
X = np.nan_to_num(df[feature_cols].values.astype(np.float32))
y = df["label_enc"].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# Global test tensors
X_test_t = torch.from_numpy(X_test).to(DEVICE)
y_test_t  = torch.from_numpy(y_test).to(DEVICE)

print(f"\n  Input features : {input_dim}")
print(f"  Train samples  : {len(X_train):,}")
print(f"  Test  samples  : {len(X_test):,}")
print(f"  Device         : {DEVICE}")
print("  Preprocessing complete ✓")

# ════════════════════════════════════════════════════════════════════════════
# MODULE 4 — DIRICHLET PARTITION (Size-Aware Capped — v3 logic retained)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*72)
print("  STEP 2 │ SIZE-AWARE CAPPED DIRICHLET PARTITION")
print("="*72)


def size_aware_partition(y, n_clients, dominant_frac=0.75,
                         residual_cap=0.30, alpha=0.5, seed=42):
    rng            = np.random.default_rng(seed)
    dominant_pool  = {}
    residual_pool  = {}

    for cls_id in range(n_clients):
        idx   = np.where(y == cls_id)[0].copy()
        rng.shuffle(idx)
        n_dom = int(len(idx) * dominant_frac)
        dominant_pool[cls_id] = idx[:n_dom]
        residual_pool[cls_id] = idx[n_dom:]

    dominant_sizes = {k: len(v) for k, v in dominant_pool.items()}
    client_indices = [list(dominant_pool[k]) for k in range(n_clients)]

    for cls_id in range(n_clients):
        res = residual_pool[cls_id].copy()
        if len(res) == 0:
            continue
        props = rng.dirichlet(alpha * np.ones(n_clients))
        order = np.argsort(-props)
        ptr   = 0
        for k in order:
            if ptr >= len(res):
                break
            cap  = int(residual_cap * dominant_sizes[k])
            give = min(cap, len(res) - ptr)
            if give > 0:
                client_indices[k].extend(res[ptr:ptr+give].tolist())
                ptr += give

    for k in range(n_clients):
        arr = np.array(client_indices[k])
        rng.shuffle(arr)
        client_indices[k] = arr.tolist()

    return client_indices


client_indices = size_aware_partition(
    y_train, n_clients, DOMINANT_FRAC, RESIDUAL_CAP, DIRICHLET_ALPHA)

client_names   = [f"Client_{k}_{classes[k]}" for k in range(n_clients)]

print(f"\n  {'Client':<38} {'N':>8}  {'Dom%':>6}  Top-2 distribution")
print(f"  {'-'*80}")
partition_stats = []
for k, (name, idx) in enumerate(zip(client_names, client_indices)):
    local_y   = y_train[idx]
    cls_count = Counter(local_y)
    total     = len(local_y)
    dom_pct   = 100 * cls_count.get(k, 0) / total
    top2      = " | ".join(
        f"{classes[c]}:{100*cls_count[c]/total:.1f}%"
        for c, _ in cls_count.most_common(2))
    flag = "✅" if dom_pct >= 50 else "⚠️ "
    print(f"  {flag} {name:<36} {total:>8,}  {dom_pct:>5.1f}%  {top2}")
    partition_stats.append({
        "client": f"C{k}\n{classes[k][:7]}", "n_samples": total,
        **{classes[c]: cls_count.get(c, 0) for c in range(n_classes)}
    })

# Partition visualisation
colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
ps_df  = pd.DataFrame(partition_stats).set_index("client")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle(f"Dirichlet Partition  (dom_frac={DOMINANT_FRAC}, "
             f"res_cap={RESIDUAL_CAP}, α={DIRICHLET_ALPHA})",
             fontsize=12, fontweight="bold")
(ps_df[list(classes)].div(ps_df["n_samples"], axis=0) * 100).plot(
    kind="bar", stacked=True, ax=axes[0], color=colors, edgecolor="white")
axes[0].set_title("Class Composition per Client (%)")
axes[0].set_ylabel("Percentage")
axes[0].tick_params(axis="x", labelrotation=0, labelsize=8)
axes[0].legend(classes, fontsize=7, loc="upper right")

dom_vals = [100*Counter(y_train[client_indices[k]]).get(k,0)/len(client_indices[k])
            for k in range(n_clients)]
ax1_bars = axes[1].bar(
    [f"C{k}" for k in range(n_clients)],
    [len(client_indices[k]) for k in range(n_clients)],
    color=colors, edgecolor="white")
for bar, dp in zip(ax1_bars, dom_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
                 f"dom\n{dp:.0f}%", ha="center", va="bottom", fontsize=8)
axes[1].set_title("Samples per Client")
axes[1].set_ylabel("# Samples")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_nn_partition.png", dpi=150)
plt.close()
print("\n  Partition plot saved ✓")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 5 — FEDERATED UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def get_model() -> TrafficClassifierNN:
    """Instantiate a fresh global-architecture model."""
    return TrafficClassifierNN(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        n_classes=n_classes,
        dropout=DROPOUT_RATE
    ).to(DEVICE)


def get_model_weights(model: nn.Module) -> dict:
    """Extract a deep copy of model state_dict (weights only, no grad)."""
    return copy.deepcopy(model.state_dict())


def set_model_weights(model: nn.Module, weights: dict) -> nn.Module:
    """Load a state_dict into a model."""
    model.load_state_dict(weights)
    return model


def fedavg_aggregate(client_weights: list, client_sizes: list) -> dict:
    """
    True FedAvg Aggregation.
    Computes: W_global = Σ_k (n_k / N) × W_k
    for every parameter tensor in the model.

    Parameters
    ----------
    client_weights : list of state_dicts from each participating client
    client_sizes   : list of local dataset sizes (n_k)

    Returns
    -------
    aggregated state_dict (global model weights)
    """
    total_samples = sum(client_sizes)
    fed_weights   = [n / total_samples for n in client_sizes]

    # Start with a zero-valued copy of the first client's state_dict
    agg_weights = {}
    for key in client_weights[0].keys():
        # Weighted sum across all clients for this parameter tensor
        agg_weights[key] = sum(
            fed_weights[i] * client_weights[i][key].float()
            for i in range(len(client_weights))
        )

    return agg_weights


def make_focal_loss(y_local: np.ndarray) -> FocalLoss:
    """
    Build FocalLoss with per-class alpha weights derived from
    inverse class frequency on the local client data.
    """
    counts    = np.bincount(y_local, minlength=n_classes).astype(float)
    counts    = np.where(counts == 0, 1e-6, counts)   # avoid div/0
    freq      = counts / counts.sum()
    alpha     = torch.tensor(1.0 / freq, dtype=torch.float32).to(DEVICE)
    alpha     = alpha / alpha.sum()                    # normalise
    return FocalLoss(alpha=alpha, gamma=2.0)


def make_dataloader(X_local: np.ndarray, y_local: np.ndarray,
                    batch_size: int) -> DataLoader:
    """
    Build a DataLoader with WeightedRandomSampler to further counter
    within-client class imbalance during mini-batch sampling.
    """
    dataset   = TensorDataset(
        torch.from_numpy(X_local).float(),
        torch.from_numpy(y_local).long()
    )
    counts    = np.bincount(y_local, minlength=n_classes).astype(float)
    counts    = np.where(counts == 0, 1.0, counts)
    class_wt  = 1.0 / counts
    sample_wt = torch.tensor([class_wt[y] for y in y_local], dtype=torch.float)
    sampler   = WeightedRandomSampler(sample_wt, num_samples=len(sample_wt),
                                      replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      drop_last=False)


def local_train(model: nn.Module, X_local: np.ndarray, y_local: np.ndarray,
                epochs: int, lr: float, wd: float) -> tuple:
    """
    Train model locally for `epochs` epochs.
    Returns (trained model, list of epoch losses).
    """
    model.train()
    loader    = make_dataloader(X_local, y_local, BATCH_SIZE)
    criterion = make_focal_loss(y_local)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    epoch_losses = []
    for ep in range(epochs):
        total_loss, n_batches = 0.0, 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        scheduler.step()
        epoch_losses.append(total_loss / max(n_batches, 1))

    return model, epoch_losses


@torch.no_grad()
def evaluate_global(model: nn.Module, X_t: torch.Tensor,
                    y_t: torch.Tensor) -> tuple:
    """
    Evaluate global model on the held-out test set.
    Returns (accuracy, predicted labels numpy array).
    """
    model.eval()
    logits = model(X_t)
    preds  = logits.argmax(dim=1).cpu().numpy()
    labels = y_t.cpu().numpy()
    acc    = accuracy_score(labels, preds)
    return acc, preds


# ════════════════════════════════════════════════════════════════════════════
# MODULE 6 — FEDERATED LEARNING ROUNDS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*72)
print(f"  STEP 3 │ FEDERATED TRAINING  ({FL_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs)")
print("="*72)
print(f"""
  FL Protocol per Round:
  ─────────────────────
  1. Server broadcasts global model weights to all clients
  2. Each client:
       a. Loads global weights into its local model
       b. Trains for {LOCAL_EPOCHS} epochs using local data
          (Focal Loss + AdamW + CosineAnnealingLR + WeightedSampler)
       c. Sends updated weight tensors back to server
  3. Server aggregates: W_global = Σ (n_k / N) × W_k   [True FedAvg]
  4. Evaluate global model on held-out test set
  5. Repeat for {FL_ROUNDS} rounds
""")

# ── Initialise global model ───────────────────────────────────────────────
global_model   = get_model()
global_weights = get_model_weights(global_model)

# ── Prepare local datasets ────────────────────────────────────────────────
client_data = []
for k in range(n_clients):
    idx      = np.array(client_indices[k])
    X_local  = X_train[idx]
    y_local  = y_train[idx]
    client_data.append((X_local, y_local))

# ── Tracking ──────────────────────────────────────────────────────────────
round_global_accs     = []
round_client_accs     = defaultdict(list)
all_client_losses     = defaultdict(list)   # k -> list of per-round epoch losses

# ── FL Rounds ─────────────────────────────────────────────────────────────
for fl_round in range(1, FL_ROUNDS + 1):
    print(f"\n  {'─'*65}")
    print(f"  ▶  ROUND {fl_round}/{FL_ROUNDS}")
    print(f"  {'─'*65}")

    participating_weights = []
    participating_sizes   = []

    for k in range(n_clients):
        X_local, y_local = client_data[k]
        n_local          = len(X_local)

        # Step 1: Load global weights into local model
        local_model = get_model()
        set_model_weights(local_model, copy.deepcopy(global_weights))

        # Step 2: Local training
        local_model, ep_losses = local_train(
            local_model, X_local, y_local,
            epochs=LOCAL_EPOCHS, lr=LR, wd=WEIGHT_DECAY)

        # Step 3: Local evaluation
        local_model.eval()
        with torch.no_grad():
            X_lt = torch.from_numpy(X_local).float().to(DEVICE)
            y_lt = torch.from_numpy(y_local).long().to(DEVICE)
            local_preds = local_model(X_lt).argmax(1).cpu().numpy()
        local_acc = accuracy_score(y_local, local_preds)
        round_client_accs[k].append(local_acc)
        all_client_losses[k].extend(ep_losses)

        dom_pct = 100 * Counter(y_local)[k] / len(y_local)
        print(f"    [{k}] {client_names[k]:<40} "
              f"n={n_local:>7,}  dom={dom_pct:.0f}%  "
              f"loss={ep_losses[-1]:.4f}  local_acc={local_acc:.4f}")

        # Step 4: Collect weights for aggregation
        participating_weights.append(get_model_weights(local_model))
        participating_sizes.append(n_local)

    # Step 5: FedAvg aggregation
    global_weights = fedavg_aggregate(participating_weights, participating_sizes)
    set_model_weights(global_model, global_weights)

    # Step 6: Global evaluation
    round_acc, _ = evaluate_global(global_model, X_test_t, y_test_t)
    round_global_accs.append(round_acc)
    print(f"\n  🌐 Round {fl_round} Global Accuracy : {round_acc*100:.2f}%")

print(f"\n  {'='*65}")
print(f"  FL Training Complete ✓")
print(f"  Best Round Accuracy  : {max(round_global_accs)*100:.2f}%  "
      f"(Round {np.argmax(round_global_accs)+1})")
print(f"  Final Round Accuracy : {round_global_accs[-1]*100:.2f}%")
print(f"  {'='*65}")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 7 — FINAL EVALUATION & REPORTING
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*72)
print("  STEP 4 │ FINAL GLOBAL MODEL EVALUATION")
print("="*72)

final_acc, y_pred = evaluate_global(global_model, X_test_t, y_test_t)
print(f"\n  ✅ Final Global Federated Accuracy : {final_acc*100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=classes,
                             digits=4, zero_division=0))

print(f"\n  Per-Client Summary (final round):")
print(f"  {'#':<4} {'Class':<24} {'Samples':>8} {'Dom%':>6} {'FinalLocalAcc':>14}")
print(f"  {'-'*62}")
for k in range(n_clients):
    dom_pct   = 100 * Counter(y_train[client_indices[k]]).get(k, 0) / len(client_indices[k])
    final_loc = round_client_accs[k][-1] if round_client_accs[k] else 0
    flag      = "✅" if dom_pct >= 50 else "⚠️ "
    print(f"  {flag}{k:<3} {classes[k]:<24} {len(client_indices[k]):>8,} "
          f"{dom_pct:>5.1f}% {final_loc:>14.4f}")

print(f"\n  {'─'*62}")
print(f"  Final Global Federated Accuracy : {final_acc*100:.2f}%")
print(f"  {'─'*62}")


# ════════════════════════════════════════════════════════════════════════════
# MODULE 8 — VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

print("\n  Generating visualisations...")

colors6 = plt.cm.tab10(np.linspace(0, 1, n_clients))

# ── Fig 1: FL Round Accuracy Progression ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, FL_ROUNDS+1), [a*100 for a in round_global_accs],
        "o-", color="crimson", linewidth=2.5, markersize=8,
        label="Global Model Accuracy")
for r, acc in enumerate(round_global_accs, 1):
    ax.annotate(f"{acc*100:.2f}%", (r, acc*100),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color="crimson")

# Per-client accuracy across rounds
for k in range(n_clients):
    ax.plot(range(1, FL_ROUNDS+1),
            [a*100 for a in round_client_accs[k]],
            "--", alpha=0.5, color=colors6[k],
            label=f"C{k}:{classes[k][:10]}")

ax.set_xlabel("FL Round")
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Federated Learning — Accuracy per Round\n"
             f"(Neural Network + FedAvg, {FL_ROUNDS} Rounds × {LOCAL_EPOCHS} Local Epochs)",
             fontsize=11)
ax.set_xticks(range(1, FL_ROUNDS+1))
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_nn_round_accuracy.png", dpi=150)
plt.close()

# ── Fig 2: Local Training Loss per Client ─────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for k in range(n_clients):
    losses = all_client_losses[k]
    axes[k].plot(losses, color=colors6[k], linewidth=1.5)
    axes[k].set_title(f"C{k}: {classes[k]}", fontsize=10)
    axes[k].set_xlabel("Epoch (across all rounds)")
    axes[k].set_ylabel("Focal Loss")
    axes[k].grid(alpha=0.3)
    # Mark round boundaries
    for r in range(1, FL_ROUNDS):
        axes[k].axvline(r * LOCAL_EPOCHS - 0.5, color="gray",
                        linestyle=":", alpha=0.6)
fig.suptitle("Local Client Training Loss  (vertical lines = FL round boundaries)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_nn_client_losses.png", dpi=150)
plt.close()

# ── Fig 3: Confusion Matrix ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(
    ax=ax, colorbar=True, xticks_rotation=45, cmap="Blues")
ax.set_title(f"Global Federated NN — Confusion Matrix\n"
             f"(Accuracy = {final_acc*100:.2f}%)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_nn_confusion_matrix.png", dpi=150)
plt.close()

# ── Fig 4: Final Per-Client vs Global Accuracy ────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
final_local = [round_client_accs[k][-1] for k in range(n_clients)]
dom_pcts_all = [100*Counter(y_train[client_indices[k]]).get(k,0)/len(client_indices[k])
                for k in range(n_clients)]
bars = ax.bar(
    [f"C{k}\n{classes[k][:9]}" for k in range(n_clients)],
    final_local, color=colors6, edgecolor="white")
ax.axhline(final_acc, color="crimson", linestyle="--", linewidth=2.2,
           label=f"Global FedAvg Acc = {final_acc*100:.2f}%")
ax.set_ylim(0, 1.12)
ax.set_title("Final Local Accuracy per Client vs Global Federated Accuracy",
             fontsize=11)
ax.set_ylabel("Accuracy")
ax.legend(fontsize=10)
for bar, val, dp in zip(bars, final_local, dom_pcts_all):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{val:.3f}\ndom:{dp:.0f}%", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_nn_client_accuracy.png", dpi=150)
plt.close()

# ── Fig 5: FedAvg Weight Aggregation Diagram ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
total = sum(len(client_indices[k]) for k in range(n_clients))
fed_w = [len(client_indices[k]) / total for k in range(n_clients)]
table_data = [["Client", "Dominant Class", "Samples", "FedAvg Weight", "Final Local Acc"]]
for k in range(n_clients):
    table_data.append([
        f"C{k}", classes[k],
        f"{len(client_indices[k]):,}",
        f"{fed_w[k]:.4f}",
        f"{round_client_accs[k][-1]:.4f}"
    ])
table_data.append(["─"*6, "─"*20, "─"*10, "─"*14, "─"*16])
table_data.append(["GLOBAL", "FedAvg Aggregated", f"{total:,}", "1.0000",
                   f"{final_acc:.4f}"])

t = ax.table(cellText=table_data[1:], colLabels=table_data[0],
             loc="center", cellLoc="center")
t.auto_set_font_size(False)
t.set_fontsize(10)
t.scale(1.2, 2.0)
for j in range(len(table_data[0])):
    t[0, j].set_facecolor("#2c3e50")
    t[0, j].set_text_props(color="white", fontweight="bold")
for j in range(len(table_data[0])):
    t[len(table_data)-1, j].set_facecolor("#27ae60")
    t[len(table_data)-1, j].set_text_props(color="white", fontweight="bold")
for i in range(1, n_clients+1):
    for j in range(len(table_data[0])):
        t[i, j].set_facecolor((*colors6[i-1][:3], 0.15))

ax.set_title("FedAvg Weight Summary  —  W_global = Σ (n_k / N) × W_k",
             fontsize=12, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_nn_fedavg_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("  All 5 plots saved ✓")
print(f"\n{'='*72}")
print(f"  PIPELINE COMPLETE")
print(f"  ✅ Final Global Federated Accuracy : {final_acc*100:.2f}%")
print(f"  Output files saved to : {OUTPUT_DIR}")
print(f"{'='*72}\n")