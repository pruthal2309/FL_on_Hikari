import os, warnings, random, copy, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

from deap import base, creator, tools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

DATA_PATH    = "/kaggle/input/datasets/kk0105/allflowmeter-hikari2021/ALLFLOWMETER_HIKARI2021.csv"
OUTPUT_DIR   = Path("/kaggle/working/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL   = "traffic_category"
DROP_COLS    = ["Unnamed: 0.1", "Unnamed: 0", "uid", "originh", "responh", "traffic_category", "Label"]

POP_SIZE     = 80
N_GEN        = 50
CXPB         = 0.70
MUTPB        = 0.02
TOURNAMENT_K = 5
MIN_FEATURES = 10
ALPHA        = 0.05
EVAL_N_TREES = 30
EVAL_SAMPLE  = 20_000
RANDOM_SEED  = 42


DOMINANT_FRAC   = 0.75
RESIDUAL_CAP    = 0.30
DIRICHLET_ALPHA = 0.5

FL_ROUNDS    = 10
LOCAL_EPOCHS = 6
BATCH_SIZE   = 256
LR           = 2e-3
WEIGHT_DECAY = 1e-4
TEST_SPLIT   = 0.20
HIDDEN_DIMS  = [512, 256, 128, 64]
DROPOUT_RATE = 0.30

SEED = RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("=" * 72)
print("  PHASE 1 — GA FEATURE SELECTION")
print("=" * 72)

print("\n[1/5] Loading dataset for GA...")
df_ga = pd.read_csv(DATA_PATH)
print(f"      Shape : {df_ga.shape}")

le_ga   = LabelEncoder()
y_full  = le_ga.fit_transform(df_ga[TARGET_COL])
classes = le_ga.classes_
n_cls   = len(classes)

feature_cols_all = [
    c for c in df_ga.columns
    if c not in DROP_COLS and df_ga[c].dtype != "object"
]
X_full = df_ga[feature_cols_all].values.astype(np.float32)
X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
N_FEATURES = X_full.shape[1]

print(f"      Features : {N_FEATURES}  |  Classes : {list(classes)}")

print("\n[2/5] Computing RF importance for warm-start...")
np.random.seed(RANDOM_SEED)
idx_bias = np.random.choice(len(X_full), min(40_000, len(X_full)), replace=False)
bias_rf  = RandomForestClassifier(
    n_estimators=50, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
bias_rf.fit(X_full[idx_bias], y_full[idx_bias])
importances = bias_rf.feature_importances_
prob_on = 0.15 + 0.70 * (
    (importances - importances.min()) /
    (importances.max() - importances.min() + 1e-9)
)

print("\n[3/5] Building GA eval split...")
np.random.seed(RANDOM_SEED)
idx_eval = np.random.choice(len(X_full), min(EVAL_SAMPLE, len(X_full)), replace=False)
X_ev, y_ev = X_full[idx_eval], y_full[idx_eval]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
tr_idx, te_idx = next(sss.split(X_ev, y_ev))
X_tr, X_te = X_ev[tr_idx], X_ev[te_idx]
y_tr, y_te = y_ev[tr_idx], y_ev[te_idx]

_cache: dict = {}

def evaluate_ga(individual):
    key = tuple(individual)
    if key in _cache:
        return _cache[key]
    selected = [i for i, b in enumerate(individual) if b == 1]
    if len(selected) < MIN_FEATURES:
        _cache[key] = (0.0,)
        return (0.0,)
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_tr[:, selected])
    Xte_s  = scaler.transform(X_te[:, selected])
    clf = RandomForestClassifier(
        n_estimators=EVAL_N_TREES, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_SEED)
    clf.fit(Xtr_s, y_tr)
    preds   = clf.predict(Xte_s)
    ba      = balanced_accuracy_score(y_te, preds)
    penalty = ALPHA * (len(selected) / N_FEATURES)
    fit     = (float(ba - penalty),)
    _cache[key] = fit
    return fit

print("\n[4/5] Setting up DEAP...")
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

def biased_bit(i):
    return 1 if random.random() < prob_on[i] else 0

def make_individual():
    ind = creator.Individual(biased_bit(i) for i in range(N_FEATURES))
    on  = [i for i, b in enumerate(ind) if b == 1]
    if len(on) < MIN_FEATURES:
        for i in random.sample(range(N_FEATURES), MIN_FEATURES - len(on)):
            ind[i] = 1
    return ind

toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",   evaluate_ga)
toolbox.register("mate",       tools.cxUniform, indpb=0.5)
toolbox.register("mutate",     tools.mutFlipBit, indpb=MUTPB)
toolbox.register("select",     tools.selTournament, tournsize=TOURNAMENT_K)

print(f"\n[5/5] Running GA (pop={POP_SIZE}, gen={N_GEN})...")
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

population = toolbox.population(n=POP_SIZE)
hof        = tools.HallOfFame(5)
stats      = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("max", np.max)
stats.register("avg", np.mean)
stats.register("min", np.min)
logbook        = tools.Logbook()
logbook.header = ["gen", "nevals", "max", "avg", "min"]

t0   = time.time()
fits = list(map(toolbox.evaluate, population))
for ind, f in zip(population, fits):
    ind.fitness.values = f
hof.update(population)
logbook.record(gen=0, nevals=len(population), **stats.compile(population))
print(logbook.stream)

for gen in range(1, N_GEN + 1):
    offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(c1, c2)
            del c1.fitness.values, c2.fitness.values
    for mut in offspring:
        if random.random() < MUTPB * N_FEATURES:
            toolbox.mutate(mut)
            del mut.fitness.values
    invalid = [ind for ind in offspring if not ind.fitness.valid]
    for ind, f in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = f
    offspring  = tools.selBest(offspring, len(offspring) - 1) + tools.selBest(population, 1)
    population[:] = offspring
    hof.update(population)
    logbook.record(gen=gen, nevals=len(invalid), **stats.compile(population))
    print(logbook.stream)

print(f"\n  GA done in {(time.time()-t0)/60:.1f} min | cache hits: {len(_cache)}")

best          = hof[0]
sel_idx       = [i for i, b in enumerate(best) if b == 1]
GA_SEL_FEATS  = [feature_cols_all[i] for i in sel_idx]

print(f"\n  Best fitness : {best.fitness.values[0]:.4f}")
print(f"  Features selected : {len(GA_SEL_FEATS)} / {N_FEATURES}")
for k, f in enumerate(GA_SEL_FEATS, 1):
    print(f"    {k:2d}. {f}")


print("\n" + "=" * 72)
print("  PHASE 2 — FEDERATED LEARNING")
print("=" * 72)


class TrafficMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, dropout=0.30):
        super().__init__()
        drops  = [dropout, dropout*0.83, dropout*0.67, dropout*0.33]
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


print("\n[1/4] Data preprocessing...")
df = pd.read_csv(DATA_PATH)
drop_fl = ["Unnamed: 0.1", "Unnamed: 0", "uid", "originh", "responh", "Label"]
df.drop(columns=[c for c in drop_fl if c in df.columns], inplace=True)

df = df.reset_index(drop=True)

le        = LabelEncoder()
df["label_enc"] = le.fit_transform(df[TARGET_COL])
classes   = le.classes_
n_classes = len(classes)
n_clients = n_classes

for c in df.columns:
    if df[c].dtype == object and c != TARGET_COL:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

all_feature_cols = [c for c in df.columns if c not in [TARGET_COL, "label_enc"]]
feature_cols     = [f for f in GA_SEL_FEATS if f in df.columns]
input_dim        = len(feature_cols)

print(f"  GA features used : {input_dim} / {len(all_feature_cols)}")

X = np.nan_to_num(df[feature_cols].values.astype(np.float32))
y = df["label_enc"].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

X_test_t  = torch.from_numpy(X_test).to(DEVICE)
y_test_np = y_test.copy()

print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}  |  Device : {DEVICE}")

print("\n[2/4] Partitioning data (Capped Dirichlet)...")

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

client_indices = size_aware_partition(
    y_train, n_clients, DOMINANT_FRAC, RESIDUAL_CAP, DIRICHLET_ALPHA)
client_names   = [f"Client_{k}_{classes[k]}" for k in range(n_clients)]
COLORS         = plt.cm.tab10(np.linspace(0, 1, n_classes))

for k, (name, idx) in enumerate(zip(client_names, client_indices)):
    ly  = y_train[idx]
    cc  = Counter(ly)
    tot = len(ly)
    dp  = 100 * cc.get(k, 0) / tot
    print(f"  [{k}] {name:<40} n={tot:>6,}  dom={dp:.1f}%")


def get_model():
    return TrafficMLP(input_dim, HIDDEN_DIMS, n_classes, DROPOUT_RATE).to(DEVICE)

def get_weights(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}

def set_weights(m, w):
    m.load_state_dict(w, strict=True)
    return m

def fedavg_aggregate(weights_list, sizes):
    N   = sum(sizes)
    nrm = [s / N for s in sizes]
    agg = {}
    for key in weights_list[0]:
        agg[key] = sum(nrm[i] * weights_list[i][key].float()
                       for i in range(len(weights_list)))
    return agg

def make_weighted_ce(y_local):
    cnt    = np.bincount(y_local, minlength=n_classes).astype(float)
    cnt    = np.where(cnt == 0, 1e-9, cnt)
    weight = torch.tensor(cnt.sum() / cnt / n_classes,
                          dtype=torch.float32).to(DEVICE)
    return nn.CrossEntropyLoss(weight=weight)

def local_train(model, X_local, y_local, n_epochs):
    model.train()
    loader    = DataLoader(
        TensorDataset(torch.from_numpy(X_local).float(),
                      torch.from_numpy(y_local).long()),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    criterion = make_weighted_ce(y_local)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(loader), epochs=n_epochs,
        pct_start=0.3, anneal_strategy="cos")
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
def evaluate_fl(model, X_t, y_np):
    model.eval()
    preds = []
    for (Xb,) in DataLoader(TensorDataset(X_t), batch_size=2048):
        preds.append(model(Xb.to(DEVICE)).argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds)
    return accuracy_score(y_np, y_pred), y_pred


print(f"\n[3/4] FL Training — {FL_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs...")

global_model   = get_model()
global_weights = get_weights(global_model)
n_params       = sum(p.numel() for p in global_model.parameters())
n_keys         = len(global_weights)

print(f"  Architecture : {input_dim}→512→256→128→64→{n_classes}  |  Params: {n_params:,}")

client_data       = [(X_train[np.array(client_indices[k])],
                      y_train[np.array(client_indices[k])])
                     for k in range(n_clients)]
round_global_acc  = []
client_round_acc  = defaultdict(list)
client_all_losses = defaultdict(list)

t_total = time.time()

for fl_round in range(1, FL_ROUNDS + 1):
    t_rnd = time.time()
    print(f"\n  ROUND {fl_round}/{FL_ROUNDS}")
    rnd_weights, rnd_sizes = [], []

    for k in range(n_clients):
        X_local, y_local = client_data[k]
        local_m = get_model()
        set_weights(local_m, copy.deepcopy(global_weights))
        local_m, ep_losses = local_train(local_m, X_local, y_local, LOCAL_EPOCHS)
        X_lt     = torch.from_numpy(X_local).float().to(DEVICE)
        l_acc, _ = evaluate_fl(local_m, X_lt, y_local)
        client_round_acc[k].append(l_acc)
        client_all_losses[k].extend(ep_losses)
        dp = 100 * Counter(y_local)[k] / len(y_local)
        print(f"    [{k}] {client_names[k]:<42} n={len(X_local):>6,}  "
              f"dom={dp:.0f}%  loss={ep_losses[-1]:.4f}  acc={l_acc:.4f}")
        rnd_weights.append(get_weights(local_m))
        rnd_sizes.append(len(X_local))

    global_weights = fedavg_aggregate(rnd_weights, rnd_sizes)
    set_weights(global_model, global_weights)
    g_acc, _ = evaluate_fl(global_model, X_test_t, y_test_np)
    round_global_acc.append(g_acc)
    print(f"  Global Round {fl_round} Accuracy : {g_acc*100:.2f}%  "
          f"[{time.time()-t_rnd:.1f}s]")

total_time = time.time() - t_total
best_round = int(np.argmax(round_global_acc)) + 1
print(f"\n  FL done in {total_time:.1f}s  |  Best: {max(round_global_acc)*100:.2f}% (R{best_round})")


print("\n[4/4] Final evaluation & plots...")

final_acc, y_pred = evaluate_fl(global_model, X_test_t, y_test_np)
print(f"\n  Final Global Accuracy : {final_acc*100:.2f}%")
print(f"  GA features used      : {input_dim} / {len(all_feature_cols)}")
print(classification_report(y_test_np, y_pred,
                             target_names=classes, digits=4, zero_division=0))

GA_LABEL = f"GA-selected {input_dim} features"

fig, ax = plt.subplots(figsize=(12, 5))
rounds  = list(range(1, FL_ROUNDS + 1))
ax.plot(rounds, [a*100 for a in round_global_acc],
        "o-", color="crimson", lw=2.5, ms=9, zorder=5,
        label=f"Global Model (best={max(round_global_acc)*100:.2f}%)")
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
    f"LayerNorm MLP + Weighted CE + FedAvg | {GA_LABEL} "
    f"({FL_ROUNDS}R × {LOCAL_EPOCHS}E)", fontsize=11, fontweight="bold")
ax.set_xticks(rounds)
ax.legend(fontsize=8, loc="lower right", ncol=2)
ax.grid(alpha=0.3)
ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_round_accuracy.png", dpi=150); plt.close()

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for k in range(n_clients):
    losses = client_all_losses[k]
    axes[k].plot(losses, color=COLORS[k], lw=2)
    for r in range(1, FL_ROUNDS):
        axes[k].axvline(r * LOCAL_EPOCHS + 0.5, color="gray", ls=":", alpha=0.7)
    dp = 100*Counter(y_train[client_indices[k]]).get(k,0)/len(client_indices[k])
    axes[k].set_title(f"C{k}: {classes[k]}", fontsize=10, fontweight="bold")
    axes[k].set_xlabel("Local Epoch")
    axes[k].set_ylabel("Loss")
    axes[k].grid(alpha=0.3)
    axes[k].set_facecolor((*COLORS[k][:3], 0.07))
    axes[k].annotate(f"dom={dp:.0f}% | n={len(client_indices[k]):,} | feats={input_dim}",
                     xy=(0.03, 0.92), xycoords="axes fraction",
                     fontsize=8.5, color=COLORS[k])
fig.suptitle(f"Per-Client Training Loss | {GA_LABEL}", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_client_losses.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(9, 7))
cm = confusion_matrix(y_test_np, y_pred)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(
    ax=ax, colorbar=True, xticks_rotation=45, cmap="Blues")
ax.set_title(
    f"Federated NN — Confusion Matrix\n{GA_LABEL} | Accuracy={final_acc*100:.2f}%",
    fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_confusion_matrix.png", dpi=150); plt.close()

total_n = sum(len(client_indices[k]) for k in range(n_clients))
fed_w   = [len(client_indices[k]) / total_n for k in range(n_clients)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].pie(
    fed_w,
    labels=[f"C{k}\n{classes[k][:10]}" for k in range(n_clients)],
    autopct="%1.1f%%", colors=COLORS[:n_clients],
    startangle=140, pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=1.5))
axes[0].set_title(
    f"FedAvg Weights (n_k/N)\nW_global = Σ(n_k/N)·W_k [{n_keys} tensors]",
    fontsize=11, fontweight="bold")

acc_mat = np.array([[client_round_acc[k][r]*100 for r in range(FL_ROUNDS)]
                    for k in range(n_clients)])
im = axes[1].imshow(acc_mat, cmap="YlGn", aspect="auto", vmin=0, vmax=100)
axes[1].set_xticks(range(FL_ROUNDS))
axes[1].set_xticklabels([f"R{r+1}" for r in range(FL_ROUNDS)])
axes[1].set_yticks(range(n_clients))
axes[1].set_yticklabels([f"C{k}: {classes[k][:10]}" for k in range(n_clients)])
axes[1].set_title(f"Client Accuracy Heatmap per Round\n{GA_LABEL}",
                  fontsize=11, fontweight="bold")
for i in range(n_clients):
    for j in range(FL_ROUNDS):
        axes[1].text(j, i, f"{acc_mat[i,j]:.1f}", ha="center", va="center",
                     fontsize=8.5, color="black" if acc_mat[i,j] > 50 else "white")
plt.colorbar(im, ax=axes[1], label="Accuracy (%)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_fedavg_analysis.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(14, max(5, len(feature_cols) * 0.28)))
ax.barh(range(len(feature_cols)), [1]*len(feature_cols), color="steelblue", alpha=0.7)
ax.set_yticks(range(len(feature_cols)))
ax.set_yticklabels(feature_cols, fontsize=8)
ax.set_title(
    f"GA-Selected Features ({len(feature_cols)} / {len(all_feature_cols)}) "
    f"— {100*(1-len(feature_cols)/len(all_feature_cols)):.1f}% reduction",
    fontsize=11, fontweight="bold")
ax.xaxis.set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fl_ga_features.png", dpi=150); plt.close()

print("  All plots saved.")

print(f"\n{'='*72}")
print(f"  PIPELINE COMPLETE")
print(f"  Architecture  : LayerNorm MLP ({input_dim}→512→256→128→64→{n_classes})")
print(f"  GA Features   : {input_dim} selected / {len(all_feature_cols)} total "
      f"({100*(1-input_dim/len(all_feature_cols)):.1f}% reduction)")
print(f"  Parameters    : {n_params:,}  |  FedAvg tensors: {n_keys}")
print(f"  FL Protocol   : {FL_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs")
print(f"  Training Time : {total_time:.1f}s")
print(f"  Final Accuracy: {final_acc*100:.2f}%")
print(f"  Best Round    : {max(round_global_acc)*100:.2f}% (Round {best_round})")
print(f"{'='*72}")