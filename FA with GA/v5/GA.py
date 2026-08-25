"""
GA-Based Feature Selection for Federated Learning
Dataset  : ALLFLOWMETER_HIKARI2021.csv
Target   : traffic_category  (6-class multiclass)
           Background | Benign | Bruteforce | Bruteforce-XML | Probing    | XMRIGCC CryptoMiner

Chromosome encoding
───────────────────
Each individual is a binary vector, length = n_features.
  1 → feature selected   0 → feature dropped

Fitness = macro_balanced_accuracy  –  alpha * (n_selected / n_total)
          ↑ rewards all 6 classes equally    ↑ penalises too many features

Why macro balanced accuracy?
  - Benign  = 63 %, Background = 31 %  →  plain accuracy is misleading
  - Minority classes (Bruteforce 1%, XMRig 0.6%) must not be ignored
  - Macro-averaged balanced_accuracy_score weights every class equally

GA hyper-params (tune at top of file)
──────────────────────────────────────
POP_SIZE     : population size          (50-200)
N_GEN        : number of generations   (30-100)
CXPB         : crossover probability
MUTPB        : per-bit mutation probability
TOURNAMENT_K : tournament selection size
MIN_FEATURES : hard floor on selected features
ALPHA        : feature-count penalty   (0.0 – 0.1)
"""

import numpy  as np
import pandas as pd
import random, json, time
from pathlib import Path

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (balanced_accuracy_score,
                                     classification_report,
                                     confusion_matrix)
from sklearn.pipeline        import Pipeline

from deap import base, creator, tools

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIG — edit these to tune the GA
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "/kaggle/input/datasets/kk0105/allflowmeter-hikari2021/ALLFLOWMETER_HIKARI2021.csv"   # ← update if needed
# OUTPUT_DIR   = Path("ga_results")
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COL   = "traffic_category"              # ← multiclass label

# GA
POP_SIZE     = 100
N_GEN        = 1
CXPB         = 0.70   # crossover probability
MUTPB        = 0.02   # per-bit mutation probability
TOURNAMENT_K = 5
MIN_FEATURES = 10     # never allow fewer than this
ALPHA        = 0.05   # fitness penalty weight for feature count

# Eval (small & fast during GA)
EVAL_N_TREES    = 30
EVAL_SAMPLE     = 20_000   # rows sampled per fitness call
RANDOM_SEED     = 42

# Columns that are NOT features
DROP_COLS = [
    "Unnamed: 0.1", "Unnamed: 0", "uid",
    "originh", "responh", "traffic_category", "Label",
]

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD & PRE-PROCESS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  GA Feature Selection — HIKARI 2021  (multiclass)")
print(f"  Target : {TARGET_COL}")
print("=" * 65)

print("\n[1/5] Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"      Shape           : {df.shape}")
print(f"      Class distribution:\n{df[TARGET_COL].value_counts().to_string()}")

# Encode target
le      = LabelEncoder()
y_full  = le.fit_transform(df[TARGET_COL])          # int 0-5
classes = le.classes_
n_cls   = len(classes)
print(f"\n      Encoded classes ({n_cls}): {list(classes)}")

# Feature matrix — numeric only, drop identifiers
feature_cols = [
    c for c in df.columns
    if c not in DROP_COLS and df[c].dtype != "object"
]
X_full = df[feature_cols].values.astype(np.float32)
X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

N_FEATURES = X_full.shape[1]
print(f"      Feature count   : {N_FEATURES}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  WARM-START BIAS via RF Importance
#     Bias initial population toward high-importance features so GA
#     converges faster without losing random exploration.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Computing RF importance for warm-start …")
np.random.seed(RANDOM_SEED)
idx_bias = np.random.choice(len(X_full), min(40_000, len(X_full)), replace=False)

bias_rf = RandomForestClassifier(
    n_estimators = 50,
    class_weight = "balanced",   # important: multiclass & imbalanced
    n_jobs       = -1,
    random_state = RANDOM_SEED,
)


bias_rf.fit(X_full[idx_bias], y_full[idx_bias])
importances = bias_rf.feature_importances_

# Map importance → init probability in [0.15, 0.85]
prob_on = 0.15 + 0.70 * (
    (importances - importances.min()) /
    (importances.max() - importances.min() + 1e-9)
)
print(f"      Importance range: [{importances.min():.4f}, {importances.max():.4f}]")
print(f"      Top-5 warm-start features:")
top5 = np.argsort(importances)[::-1][:5]
for i in top5:
    print(f"        {importances[i]:.4f}  {feature_cols[i]}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FIXED EVAL SPLIT  (deterministic across all fitness calls)
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(RANDOM_SEED)
idx_eval = np.random.choice(len(X_full),
                             min(EVAL_SAMPLE, len(X_full)), replace=False)
X_ev, y_ev = X_full[idx_eval], y_full[idx_eval]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25,
                              random_state=RANDOM_SEED)
tr_idx, te_idx = next(sss.split(X_ev, y_ev))
X_tr, X_te = X_ev[tr_idx], X_ev[te_idx]
y_tr, y_te = y_ev[tr_idx], y_ev[te_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FITNESS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
_cache: dict = {}   # memoise identical chromosomes

def evaluate(individual):
    """
    fitness = macro_balanced_accuracy  –  alpha * (n_sel / N_FEATURES)

    balanced_accuracy_score with adjusted=False gives per-class recall
    averaged over all 6 classes → no single majority class can dominate.
    """
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
        n_estimators = EVAL_N_TREES,
        class_weight = "balanced",   # crucial for minority classes
        n_jobs       = -1,
        random_state = RANDOM_SEED,
    )
    clf.fit(Xtr_s, y_tr)
    preds = clf.predict(Xte_s)

    # macro balanced accuracy = mean per-class recall
    ba      = balanced_accuracy_score(y_te, preds)
    penalty = ALPHA * (len(selected) / N_FEATURES)
    fit     = (float(ba - penalty),)

    _cache[key] = fit
    return fit


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DEAP TOOLBOX
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Setting up DEAP …")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def biased_bit(i):
    return 1 if random.random() < prob_on[i] else 0

def make_individual():
    ind = creator.Individual(biased_bit(i) for i in range(N_FEATURES))
    # guarantee MIN_FEATURES are on
    on = [i for i, b in enumerate(ind) if b == 1]
    if len(on) < MIN_FEATURES:
        for i in random.sample(range(N_FEATURES), MIN_FEATURES - len(on)):
            ind[i] = 1
    return ind

toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",   evaluate)
toolbox.register("mate",       tools.cxUniform, indpb=0.5)
toolbox.register("mutate",     tools.mutFlipBit, indpb=MUTPB)
toolbox.register("select",     tools.selTournament, tournsize=TOURNAMENT_K)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RUN GA
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4/5] Running GA  (pop={POP_SIZE}, gen={N_GEN}) …\n")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

population = toolbox.population(n=POP_SIZE)
hof        = tools.HallOfFame(5)

stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("max",  np.max)
stats.register("avg",  np.mean)
stats.register("min",  np.min)

logbook        = tools.Logbook()
logbook.header = ["gen", "nevals", "max", "avg", "min"]

# ── Gen 0 ────────────────────────────────────────────────────────────────────
t0 = time.time()
fits = list(map(toolbox.evaluate, population))
for ind, f in zip(population, fits):
    ind.fitness.values = f
hof.update(population)
logbook.record(gen=0, nevals=len(population), **stats.compile(population))
print(logbook.stream)

# ── Generational loop ─────────────────────────────────────────────────────────
for gen in range(1, N_GEN + 1):
    offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))

    # Crossover
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(c1, c2)
            del c1.fitness.values, c2.fitness.values

    # Mutation  (per-individual gate so not every member mutates)
    for mut in offspring:
        if random.random() < MUTPB * N_FEATURES:
            toolbox.mutate(mut)
            del mut.fitness.values

    # Evaluate invalids
    invalid = [ind for ind in offspring if not ind.fitness.valid]
    for ind, f in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = f

    # Elitism — carry best individual forward
    offspring = tools.selBest(offspring, len(offspring) - 1) + \
                tools.selBest(population, 1)

    population[:] = offspring
    hof.update(population)
    logbook.record(gen=gen, nevals=len(invalid), **stats.compile(population))
    print(logbook.stream)

print(f"\n  GA done in {(time.time()-t0)/60:.1f} min  |  cache hits: {len(_cache)}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Extracting results…")

best        = hof[0]
sel_idx     = [i for i, b in enumerate(best) if b == 1]
sel_feats   = [feature_cols[i] for i in sel_idx]

print(f"\n{'─'*65}")
print(f"  Best fitness (balanced_acc − penalty) : {best.fitness.values[0]:.4f}")
print(f"  Features selected : {len(sel_feats)} / {N_FEATURES}")
print(f"\n  Selected features:")
for k, f in enumerate(sel_feats, 1):
    print(f"    {k:2d}. {f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
results = {
    "target_column"   : TARGET_COL,
    "classes"         : list(classes),
    "n_classes"       : int(n_cls),
    "best_fitness"    : float(best.fitness.values[0]),
    "n_selected"      : len(sel_feats),
    "n_total"         : N_FEATURES,
    "selected_features": sel_feats,
    "selected_indices" : sel_idx,
    "ga_config": dict(
        pop_size=POP_SIZE, n_gen=N_GEN, cxpb=CXPB, mutpb=MUTPB,
        tournament_k=TOURNAMENT_K, min_features=MIN_FEATURES, alpha=ALPHA,
    ),+
    "top5_hall_of_fame": [
        dict(rank=r+1,
             fitness=float(ind.fitness.values[0]),
             n_features=int(sum(ind)),
             features=[feature_cols[i] for i, b in enumerate(ind) if b == 1])
        for r, ind in enumerate(hof)
    ],
}

json_path = OUTPUT_DIR / "ga_selected_features.json"
with open(json_path, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\n  Saved → {json_path}")

pd.DataFrame(logbook).to_csv(OUTPUT_DIR / "ga_log.csv", index=False)
print(f"  Saved → {OUTPUT_DIR / 'ga_log.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FINAL VALIDATION ON FULL DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Final validation (80/20 stratified split, full dataset) ──")

sss2       = StratifiedShuffleSplit(n_splits=1, test_size=0.20,
                                    random_state=RANDOM_SEED)
tr2, te2   = next(sss2.split(X_full, y_full))

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
                   n_estimators = 100,
                   class_weight = "balanced",
                   n_jobs       = -1,
                   random_state = RANDOM_SEED,
               )),
])
pipe.fit(X_full[tr2][:, sel_idx], y_full[tr2])
preds_final = pipe.predict(X_full[te2][:, sel_idx])
ba_final    = balanced_accuracy_score(y_full[te2], preds_final)

print(f"\n  Macro Balanced Accuracy : {ba_final:.4f}")
print("\n  Per-class Report:")
print(classification_report(y_full[te2], preds_final, target_names=classes))
print("  Confusion Matrix:")
print(confusion_matrix(y_full[te2], preds_final))


# ─────────────────────────────────────────────────────────────────────────────
# 9.  FL INTEGRATION GUIDE
# ─────────────────────────────────────────────────────────────────────────────
print("""
═══════════════════════════════════════════════════════════════════
  HOW TO PLUG INTO YOUR FL PIPELINE
═══════════════════════════════════════════════════════════════════

  Step 1 — Load selected features:
    import json
    with open("ga_results/ga_selected_features.json") as f:
        ga = json.load(f)
    SELECTED = ga["selected_features"]   # column names
    INDICES  = ga["selected_indices"]    # integer indices

  Step 2 — Prepare data BEFORE Dirichlet partitioning:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y  = le.fit_transform(df["traffic_category"])   # 0-5 integers
    X  = df[SELECTED].values                        # GA-reduced features

  Step 3 — Pass (X, y) into your Capped-Dirichlet partitioner as usual.
    Every client trains on the same reduced feature set automatically.

  Step 4 — Update your FL model:
    input_dim  = len(SELECTED)   # replaces 81
    output_dim = 6               # one per traffic_category class

  Step 5 — Use CrossEntropyLoss (not BCELoss) with class weights:
    import torch
    counts = torch.tensor([170151, 347431, 5884, 5145, 23388, 3279],
                          dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

  Metric to track → macro balanced accuracy (not plain accuracy).
    from sklearn.metrics import balanced_accuracy_score
    ba = balanced_accuracy_score(y_true, y_pred, adjusted=False)

═══════════════════════════════════════════════════════════════════
""")