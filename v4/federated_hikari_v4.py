"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   FEDERATED LEARNING  —  NEURAL NETWORK + TRUE FedAvg  [v4 PRODUCTION]    ║
║   Dataset : ALLFLOWMETER_HIKARI_2021                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FL SYSTEM OVERVIEW                                                         ║
║  ──────────────────                                                         ║
║  Clients : 6  (one per traffic class — Background, Benign, Bruteforce,     ║
║                Bruteforce-XML, Probing, XMRIGCC CryptoMiner)               ║
║                                                                              ║
║  Each round:                                                                 ║
║  ┌─────────────────────────────────────────────────────┐                   ║
║  │  GLOBAL SERVER holds W_global                       │                   ║
║  │        │ broadcast                                   │                   ║
║  │        ▼                                             │                   ║
║  │  CLIENT k  loads W_global → trains locally          │                   ║
║  │            → sends W_k back                         │                   ║
║  │        │ aggregate                                   │                   ║
║  │        ▼                                             │                   ║
║  │  W_global ← Σ_k (n_k/N) · W_k   ← True FedAvg     │                      ║
║  └─────────────────────────────────────────────────────┘                    ║
║                                                                             ║
║  MODEL: LayerNorm MLP  Input(81)→512→256→128→64→Output(6)                   ║
║         + Residual skip connection (Input→64)                               ║
║                                                                             ║
║  KEY ENGINEERING DECISIONS (lessons from v3/v4 debugging)                   ║
║  ──────────────────────────────────────────────────────                     ║
║  ✅ LayerNorm (not BatchNorm)  — BN running stats break FedAvg              ║
║  ✅ Weighted CrossEntropy      — Focal Loss α→0 for dominant class          ║
║                                   causing loss=0 collapse                   ║
║  ✅ OneCycleLR per client      — CosineAnnealing resets LR to near-0       ║
║                                   at round start, killing early learning    ║
║  ✅ Inverse-freq class weights — Counters within-client imbalance           ║
║  ✅ Gradient clipping (1.0)    — Stabilises FL non-IID gradient variance    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, warnings, random, copy, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ────────────────────────────────────────────────────────────────
DATA_PATH       = "/kaggle/input/datasets/kk0105/allflowmeter-hikari2021/ALLFLOWMETER_HIKARI2021.csv"
OUTPUT_DIR      = "/mnt/user-data/outputs"
TARGET_COL      = "traffic_category"
DROP_COLS       = ["Unnamed: 0.1","Unnamed: 0","uid","originh","responh","Label"]

# Cap majority classes; keep ALL minority rows
CLASS_CAPS = {
    "Benign"              : 30000,
    "Background"          : 20000,
    "Probing"             : 10000,
    "Bruteforce"          : 5884,
    "Bruteforce-XML"      : 5145,
    "XMRIGCC CryptoMiner" : 3279,
}

# Partition
DOMINANT_FRAC   = 0.75
RESIDUAL_CAP    = 0.30
DIRICHLET_ALPHA = 0.5

# FL Training
FL_ROUNDS       = 8
LOCAL_EPOCHS    = 6
BATCH_SIZE      = 256
LR              = 2e-3
WEIGHT_DECAY    = 1e-4
TEST_SPLIT      = 0.20

# Model
HIDDEN_DIMS     = [512, 256, 128, 64]
DROPOUT_RATE    = 0.30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# MODULE 1 — MODEL
# ════════════════════════════════════════════════════════════════════════════

class TrafficMLP(nn.Module):
    """
    FL-Safe Deep MLP with LayerNorm and Residual Skip.

    Architecture:
        Input(81)
         ├──→ Linear(81→64)  ← residual skip
         └──→ Linear(81→512) → LayerNorm → GELU → Dropout(0.30)
               → Linear(512→256) → LayerNorm → GELU → Dropout(0.25)
               → Linear(256→128) → LayerNorm → GELU → Dropout(0.20)
               → Linear(128→64)  → LayerNorm → GELU → Dropout(0.10)
               → + residual
               → Linear(64→6)  [logits]

    LayerNorm vs BatchNorm in FL:
        BatchNorm stores running_mean/running_var per layer.
        FedAvg averaging these across non-IID clients produces meaningless
        statistics → activations collapse → loss/gradient vanishes.
        LayerNorm normalises per sample with no running state.
        All its parameters (γ, β) are safely averaged by FedAvg.
    """
    def __init__(self, input_dim: int, hidden_dims: list,
                 n_classes: int, dropout: float = 0.30):
        super().__init__()
        drops  = [dropout, dropout*0.83, dropout*0.67, dropout*0.33]
        blocks = []
        in_d   = input_dim
        for h_d, dr in zip(hidden_dims, drops):
            blocks += [nn.Linear(in_d, h_d),
                       nn.LayerNorm(h_d),
                       nn.GELU(),
                       nn.Dropout(dr)]
            in_d = h_d

        self.backbone     = nn.Sequential(*blocks)
        self.head         = nn.Linear(in_d, n_classes)
        self.residual     = nn.Linear(input_dim, in_d)  # skip: input → last hidden

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x) + self.residual(x))


# ════════════════════════════════════════════════════════════════════════════
# MODULE 2 — DATA PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*72)
print("  STEP 1 │ DATA PREPROCESSING")
print("═"*72)

t0 = time.time()
df = pd.read_csv(DATA_PATH)
print(f"  Raw dataset : {df.shape}")
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Stratified cap: keep all minority, cap majority
sampled = [
    df[df[TARGET_COL] == cls].sample(
        n=min(cap, int((df[TARGET_COL] == cls).sum())), random_state=SEED)
    for cls, cap in CLASS_CAPS.items()
]
df = pd.concat(sampled).reset_index(drop=True)

le        = LabelEncoder()
df["label_enc"] = le.fit_transform(df[TARGET_COL])
classes   = le.classes_
n_classes = len(classes)
n_clients = n_classes

print(f"\n  Classes ({n_classes}) after capping:")
vc = df[TARGET_COL].value_counts()
for cls, cnt in vc.items():
    bar = "█" * int(40 * cnt / vc.max())
    print(f"    {cls:<25} {cnt:>6,}  {bar}")

# Encode remaining object columns
for c in df.columns:
    if df[c].dtype == object and c != TARGET_COL:
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

X_test_t  = torch.from_numpy(X_test).to(DEVICE)
y_test_np = y_test.copy()

print(f"\n  Input features  : {input_dim}")
print(f"  Train samples   : {len(X_train):,}")
print(f"  Test  samples   : {len(X_test):,}")
print(f"  Device          : {DEVICE}")
print(f"  Preprocessing   : {time.time()-t0:.1f}s  ✓")

# ════════════════════════════════════════════════════════════════════════════
# MODULE 3 — SIZE-AWARE CAPPED DIRICHLET PARTITION
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*72)
print("  STEP 2 │ SIZE-AWARE CAPPED DIRICHLET PARTITION")
print("═"*72)


def size_aware_partition(y: np.ndarray, n_clients: int,
                         dominant_frac: float = 0.75,
                         residual_cap:  float = 0.30,
                         alpha:         float = 0.5,
                         seed:          int   = 42) -> list:
    """
    Two-Phase Size-Aware Capped Dirichlet Partition (v3 logic).

    Phase 1 — Dominant:  client k gets dominant_frac of class k exclusively.
    Phase 2 — Residual:  remaining samples distributed via Dirichlet(alpha)
                         but capped per-client at residual_cap × |dominant_k|
                         to prevent large-class flooding of minority clients.
    """
    rng = np.random.default_rng(seed)
    dom_pool, res_pool = {}, {}

    for k in range(n_clients):
        idx   = np.where(y == k)[0].copy()
        rng.shuffle(idx)
        nd    = int(len(idx) * dominant_frac)
        dom_pool[k] = idx[:nd]
        res_pool[k] = idx[nd:]

    dom_sizes    = {k: len(v) for k, v in dom_pool.items()}
    client_idx   = [list(dom_pool[k]) for k in range(n_clients)]

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


client_indices = size_aware_partition(
    y_train, n_clients, DOMINANT_FRAC, RESIDUAL_CAP, DIRICHLET_ALPHA)
client_names   = [f"Client_{k}_{classes[k]}" for k in range(n_clients)]
COLORS         = plt.cm.tab10(np.linspace(0, 1, n_classes))

print(f"\n  {'Client':<38} {'N':>7}  {'Dom%':>6}  Distribution (top-3)")
print(f"  {'─'*82}")
partition_stats = []
all_dom_pcts   = []
for k, (name, idx) in enumerate(zip(client_names, client_indices)):
    ly  = y_train[idx]; cc = Counter(ly); tot = len(ly)
    dp  = 100 * cc.get(k, 0) / tot
    all_dom_pcts.append(dp)
    top3 = " | ".join(
        f"{classes[c]}:{100*cc[c]/tot:.1f}%" for c, _ in cc.most_common(3))
    flag = "✅" if dp >= 50 else "⚠️ "
    print(f"  {flag} {name:<36} {tot:>7,}  {dp:>5.1f}%  {top3}")
    partition_stats.append({
        "client"   : f"C{k}\n{classes[k][:7]}",
        "n_samples": tot,
        **{classes[c]: cc.get(c, 0) for c in range(n_classes)}
    })

print(f"\n  Dominance → Mean:{np.mean(all_dom_pcts):.1f}%  "
      f"Min:{np.min(all_dom_pcts):.1f}%  Max:{np.max(all_dom_pcts):.1f}%")

# Partition plot
ps_df = pd.DataFrame(partition_stats).set_index("client")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle(
    f"Dirichlet Partition  "
    f"(dom_frac={DOMINANT_FRAC}, res_cap={RESIDUAL_CAP}, α={DIRICHLET_ALPHA})",
    fontsize=12, fontweight="bold")
(ps_df[list(classes)].div(ps_df["n_samples"], axis=0) * 100).plot(
    kind="bar", stacked=True, ax=axes[0], color=COLORS, edgecolor="white")
axes[0].set_title("Class Composition per Client (%)")
axes[0].set_ylabel("Percentage")
axes[0].tick_params(axis="x", labelrotation=0, labelsize=8)
axes[0].legend(classes, fontsize=7, loc="upper right")

dom_vals = [100*Counter(y_train[client_indices[k]]).get(k,0)/len(client_indices[k])
            for k in range(n_clients)]
bars = axes[1].bar(
    [f"C{k}" for k in range(n_clients)],
    [len(client_indices[k]) for k in range(n_clients)],
    color=COLORS, edgecolor="white")
for bar, dp in zip(bars, dom_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                 f"dom\n{dp:.0f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
axes[1].set_title("Samples per Client"); axes[1].set_ylabel("# Samples")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_nn_partition.png", dpi=150); plt.close()
print("  Partition plot saved ✓")

# ════════════════════════════════════════════════════════════════════════════
# MODULE 4 — FL UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def get_model() -> TrafficMLP:
    return TrafficMLP(input_dim, HIDDEN_DIMS, n_classes, DROPOUT_RATE).to(DEVICE)

def get_weights(m: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in m.state_dict().items()}

def set_weights(m: nn.Module, w: dict) -> nn.Module:
    m.load_state_dict(w, strict=True)
    return m

def fedavg_aggregate(weights_list: list, sizes: list) -> dict:
    """
    ════════════════════════════════════════════════════════
    True FedAvg Weight Aggregation  (McMahan et al. 2017)
    ════════════════════════════════════════════════════════
    W_global = Σ_k  (n_k / N)  ×  W_k

    Applied independently to every parameter tensor:
      • Linear weights & biases
      • LayerNorm γ (weight) and β (bias) — safely averaged since
        LayerNorm has no running statistics unlike BatchNorm

    Parameters
    ----------
    weights_list : list of state_dicts, one per client
    sizes        : list of local dataset sizes n_k

    Returns
    -------
    Aggregated global state_dict
    """
    N   = sum(sizes)
    nrm = [s / N for s in sizes]

    agg = {}
    for key in weights_list[0]:
        agg[key] = sum(
            nrm[i] * weights_list[i][key].float()
            for i in range(len(weights_list))
        )
    return agg


def make_weighted_ce(y_local: np.ndarray) -> nn.CrossEntropyLoss:
    """
    Inverse-frequency class weights for CrossEntropyLoss.

    For each class c: weight_c = (N / count_c) / n_classes
    This up-weights rare classes and down-weights majority.

    Why NOT Focal Loss here:
    In a skewed partition, the dominant class (e.g. Background=67%)
    gets α_dominant ≈ 0 in Focal Loss. With α→0, the loss contribution
    of all dominant-class samples vanishes → loss=0 → no gradient.
    Weighted CE avoids this by scaling, not zeroing.
    """
    cnt    = np.bincount(y_local, minlength=n_classes).astype(float)
    cnt    = np.where(cnt == 0, 1e-9, cnt)
    weight = torch.tensor(cnt.sum() / cnt / n_classes, dtype=torch.float32).to(DEVICE)
    return nn.CrossEntropyLoss(weight=weight)


def local_train(model: nn.Module, X_local: np.ndarray,
                y_local: np.ndarray, n_epochs: int) -> tuple:
    """
    One FL round of local training.

    Optimiser: AdamW
    Scheduler: OneCycleLR — ramps up then anneals within each round.
               (NOT CosineAnnealing — that resets LR to near-0 at
               the START of each round, killing early-epoch learning.)
    Loss:      Weighted CrossEntropyLoss (inverse frequency weights)
    Clipping:  Gradient norm clipped at 1.0 for FL stability

    Returns: (trained_model, list_of_per_epoch_losses)
    """
    model.train()
    loader    = DataLoader(
        TensorDataset(torch.from_numpy(X_local).float(),
                      torch.from_numpy(y_local).long()),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    criterion = make_weighted_ce(y_local)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(loader),
        epochs=n_epochs,
        pct_start=0.3,
        anneal_strategy="cos")

    epoch_losses = []
    for _ in range(n_epochs):
        run, nb = 0.0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            run += loss.item(); nb += 1
        epoch_losses.append(run / max(nb, 1))

    return model, epoch_losses


@torch.no_grad()
def evaluate(model: nn.Module, X_t: torch.Tensor,
             y_np: np.ndarray) -> tuple:
    """Evaluate model accuracy; batched to avoid OOM."""
    model.eval()
    preds = []
    for (Xb,) in DataLoader(TensorDataset(X_t), batch_size=2048):
        preds.append(model(Xb.to(DEVICE)).argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds)
    return accuracy_score(y_np, y_pred), y_pred


# ════════════════════════════════════════════════════════════════════════════
# MODULE 5 — FEDERATED LEARNING ROUNDS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*72)
print(f"  STEP 3 │ FL TRAINING  —  {FL_ROUNDS} Rounds × {LOCAL_EPOCHS} Local Epochs")
print("═"*72)

global_model   = get_model()
global_weights = get_weights(global_model)
n_params       = sum(p.numel() for p in global_model.parameters())
n_keys         = len(global_weights)

print(f"""
  Model        : LayerNorm MLP   {input_dim}→512→256→128→64→{n_classes}
  Parameters   : {n_params:,}
  FedAvg keys  : {n_keys} tensors aggregated per round
  Loss         : Weighted CrossEntropyLoss (inverse-frequency per client)
  Optimizer    : AdamW (lr={LR}) + OneCycleLR (pct_start=0.3)
  Grad clip    : max_norm=1.0
  Rounds       : {FL_ROUNDS} × {LOCAL_EPOCHS} local epochs
""")

client_data      = [(X_train[np.array(client_indices[k])],
                     y_train[np.array(client_indices[k])])
                    for k in range(n_clients)]
round_global_acc  = []
client_round_acc  = defaultdict(list)
client_all_losses = defaultdict(list)

t_total = time.time()

for fl_round in range(1, FL_ROUNDS + 1):
    t_rnd = time.time()
    print(f"  {'─'*68}")
    print(f"  ▶  ROUND {fl_round}/{FL_ROUNDS}")
    print(f"  {'─'*68}")

    rnd_weights = []
    rnd_sizes   = []

    for k in range(n_clients):
        X_local, y_local = client_data[k]

        # [1] Broadcast: load global weights
        local_m = get_model()
        set_weights(local_m, copy.deepcopy(global_weights))

        # [2] Local training (LOCAL_EPOCHS epochs)
        local_m, ep_losses = local_train(local_m, X_local, y_local, LOCAL_EPOCHS)

        # [3] Local evaluation
        X_lt  = torch.from_numpy(X_local).float().to(DEVICE)
        l_acc, _ = evaluate(local_m, X_lt, y_local)
        client_round_acc[k].append(l_acc)
        client_all_losses[k].extend(ep_losses)

        dp = 100 * Counter(y_local)[k] / len(y_local)
        print(f"    [{k}] {client_names[k]:<42} "
              f"n={len(X_local):>6,}  dom={dp:.0f}%  "
              f"loss={ep_losses[-1]:.4f}  acc={l_acc:.4f}")

        # [4] Collect weights for aggregation
        rnd_weights.append(get_weights(local_m))
        rnd_sizes.append(len(X_local))

    # [5] FedAvg: W_global = Σ (n_k/N) · W_k
    global_weights = fedavg_aggregate(rnd_weights, rnd_sizes)
    set_weights(global_model, global_weights)

    # [6] Global evaluation on held-out test set
    g_acc, _ = evaluate(global_model, X_test_t, y_test_np)
    round_global_acc.append(g_acc)
    print(f"\n  🌐 Round {fl_round} Global Accuracy : {g_acc*100:.2f}%  "
          f"[round={time.time()-t_rnd:.1f}s | "
          f"total={time.time()-t_total:.0f}s]\n")

total_time = time.time() - t_total
best_round = int(np.argmax(round_global_acc)) + 1
print(f"  {'═'*68}")
print(f"  FL Training Complete  |  Total: {total_time:.1f}s")
print(f"  Best  : {max(round_global_acc)*100:.2f}%  (Round {best_round})")
print(f"  Final : {round_global_acc[-1]*100:.2f}%")
print(f"  {'═'*68}")

# ════════════════════════════════════════════════════════════════════════════
# MODULE 6 — FINAL EVALUATION
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*72)
print("  STEP 4 │ FINAL GLOBAL MODEL EVALUATION")
print("═"*72)

final_acc, y_pred = evaluate(global_model, X_test_t, y_test_np)

print(f"\n  ✅  Final Global Federated Accuracy : {final_acc*100:.2f}%\n")
print(classification_report(y_test_np, y_pred,
                             target_names=classes, digits=4, zero_division=0))

total_n = sum(len(client_indices[k]) for k in range(n_clients))
fed_w   = [len(client_indices[k]) / total_n for k in range(n_clients)]

print(f"\n  FedAvg Summary — W_global = Σ (n_k/N) · W_k  [{n_keys} tensors]")
print(f"  {'─'*68}")
print(f"  {'#':<3} {'Class':<24} {'n_k':>7} {'n_k/N':>7} "
      f"{'Dom%':>6} {'FinalLocalAcc':>14}")
print(f"  {'─'*68}")
for k in range(n_clients):
    dp   = 100*Counter(y_train[client_indices[k]]).get(k,0)/len(client_indices[k])
    flag = "✅" if dp >= 50 else "⚠️ "
    print(f"  {flag}{k:<2} {classes[k]:<24} "
          f"{len(client_indices[k]):>7,} {fed_w[k]:>7.4f} "
          f"{dp:>5.1f}% {client_round_acc[k][-1]:>14.4f}")
print(f"  {'─'*68}")
print(f"  {'GLOBAL':<28} {total_n:>7,} {'1.0000':>7} "
      f"{'':>6} {final_acc:>14.4f}")

# ════════════════════════════════════════════════════════════════════════════
# MODULE 7 — VISUALISATIONS
# ════════════════════════════════════════════════════════════════════════════

print("\n  Generating 5 visualisation plots...")

# ── Plot 1: Round Accuracy Progression ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
rounds = list(range(1, FL_ROUNDS + 1))
ax.plot(rounds, [a*100 for a in round_global_acc],
        "o-", color="crimson", lw=2.5, ms=9, zorder=5,
        label=f"🌐 Global Model (best={max(round_global_acc)*100:.2f}%)")
for r, a in zip(rounds, round_global_acc):
    ax.annotate(f"{a*100:.1f}%", (r, a*100),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color="crimson", fontweight="bold")
for k in range(n_clients):
    ax.plot(rounds, [a*100 for a in client_round_acc[k]],
            "s--", alpha=0.6, color=COLORS[k], ms=5,
            label=f"C{k}: {classes[k][:10]}")
ax.set_xlabel("FL Round", fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title(
    f"Federated Learning — Accuracy per Round\n"
    f"LayerNorm MLP + Weighted CE + FedAvg  "
    f"({FL_ROUNDS} Rounds × {LOCAL_EPOCHS} Local Epochs)",
    fontsize=11, fontweight="bold")
ax.set_xticks(rounds)
ax.legend(fontsize=8, loc="lower right", ncol=2)
ax.grid(alpha=0.3)
ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_nn_round_accuracy.png", dpi=150); plt.close()

# ── Plot 2: Per-Client Loss Curves ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for k in range(n_clients):
    losses = client_all_losses[k]
    axes[k].plot(losses, color=COLORS[k], lw=2)
    for r in range(1, FL_ROUNDS):
        axes[k].axvline(r * LOCAL_EPOCHS + 0.5, color="gray", ls=":", alpha=0.7)
    dp = 100*Counter(y_train[client_indices[k]]).get(k,0)/len(client_indices[k])
    axes[k].set_title(f"C{k}: {classes[k]}", fontsize=10, fontweight="bold")
    axes[k].set_xlabel("Local Epoch (all rounds)")
    axes[k].set_ylabel("Weighted CE Loss")
    axes[k].grid(alpha=0.3)
    axes[k].set_facecolor((*COLORS[k][:3], 0.07))
    axes[k].annotate(f"dom={dp:.0f}% | n={len(client_indices[k]):,}",
                     xy=(0.03, 0.92), xycoords="axes fraction",
                     fontsize=8.5, color=COLORS[k])
fig.suptitle("Per-Client Training Loss  |  Vertical lines = FL Round Boundaries",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_nn_client_losses.png", dpi=150); plt.close()

# ── Plot 3: Confusion Matrix ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
cm = confusion_matrix(y_test_np, y_pred)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(
    ax=ax, colorbar=True, xticks_rotation=45, cmap="Blues")
ax.set_title(
    f"Global Federated NN — Confusion Matrix\n"
    f"FedAvg | LayerNorm MLP | Accuracy = {final_acc*100:.2f}%",
    fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_nn_confusion_matrix.png", dpi=150); plt.close()

# ── Plot 4: Per-Client Final Accuracy vs Global ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fin_loc  = [client_round_acc[k][-1] for k in range(n_clients)]
dom_pcts = [100*Counter(y_train[client_indices[k]]).get(k,0)/len(client_indices[k])
            for k in range(n_clients)]
bars = ax.bar(
    [f"C{k}\n{classes[k][:9]}" for k in range(n_clients)],
    [v*100 for v in fin_loc],
    color=COLORS[:n_clients], edgecolor="white", width=0.6)
ax.axhline(final_acc*100, color="crimson", ls="--", lw=2.2,
           label=f"Global FedAvg = {final_acc*100:.2f}%")
ax.set_ylim(0, 118)
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Per-Client Final Local Accuracy vs Global Federated Accuracy",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
for bar, val, dp in zip(bars, fin_loc, dom_pcts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{val*100:.1f}%\ndom:{dp:.0f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_nn_client_accuracy.png", dpi=150); plt.close()

# ── Plot 5: FedAvg Analysis — Pie + Accuracy Heatmap ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

wedges, texts, autotexts = axes[0].pie(
    fed_w,
    labels=[f"C{k}\n{classes[k][:10]}" for k in range(n_clients)],
    autopct="%1.1f%%",
    colors=COLORS[:n_clients],
    startangle=140, pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=1.5))
for at in autotexts: at.set_fontsize(9)
axes[0].set_title(
    f"FedAvg Contribution Weights  n_k/N\n"
    f"W_global = Σ (n_k/N) · W_k  [{n_keys} tensors]",
    fontsize=11, fontweight="bold")

acc_mat = np.array([[client_round_acc[k][r] * 100
                     for r in range(FL_ROUNDS)]
                    for k in range(n_clients)])
im = axes[1].imshow(acc_mat, cmap="YlGn", aspect="auto", vmin=0, vmax=100)
axes[1].set_xticks(range(FL_ROUNDS))
axes[1].set_xticklabels([f"R{r+1}" for r in range(FL_ROUNDS)])
axes[1].set_yticks(range(n_clients))
axes[1].set_yticklabels([f"C{k}: {classes[k][:10]}" for k in range(n_clients)])
axes[1].set_title("Client Accuracy Heatmap per FL Round (%)",
                  fontsize=11, fontweight="bold")
for i in range(n_clients):
    for j in range(FL_ROUNDS):
        axes[1].text(j, i, f"{acc_mat[i,j]:.1f}",
                     ha="center", va="center", fontsize=8.5,
                     color="black" if acc_mat[i,j] > 50 else "white")
plt.colorbar(im, ax=axes[1], label="Accuracy (%)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_nn_fedavg_analysis.png", dpi=150); plt.close()

print("  5 plots saved ✓")

# ── Final Summary ─────────────────────────────────────────────────────────
print(f"\n{'═'*72}")
print(f"  PIPELINE COMPLETE  (v4 — LayerNorm MLP + True FedAvg)")
print(f"  {'─'*68}")
print(f"  Architecture  : LayerNorm MLP ({input_dim}→512→256→128→64→{n_classes})")
print(f"  Parameters    : {n_params:,}  |  FedAvg tensors: {n_keys}")
print(f"  Loss          : Weighted CrossEntropyLoss (inverse-freq per client)")
print(f"  Optimizer     : AdamW + OneCycleLR")
print(f"  Aggregation   : True FedAvg — W_global=Σ(n_k/N)·W_k [{n_keys} tensors]")
print(f"  FL Protocol   : {FL_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs")
print(f"  Training Time : {total_time:.1f}s")
print(f"  {'─'*68}")
print(f"  ✅  Final Global Federated Accuracy : {final_acc*100:.2f}%")
print(f"  🏆  Best  Round Accuracy            : {max(round_global_acc)*100:.2f}%  (Round {best_round})")
print(f"{'═'*72}\n")