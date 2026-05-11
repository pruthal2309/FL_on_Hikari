# Federated Learning for Network Intrusion Detection
## ALLFLOWMETER_HIKARI_2021 Dataset

This project implements and evolves Federated Learning (FL) systems for network traffic classification and intrusion detection, addressing extreme class imbalance (106x ratio) and non-IID data distribution challenges.

---

## 📊 Dataset Overview

**Dataset**: ALLFLOWMETER_HIKARI_2021  
**Total Samples**: 555,278  
**Features**: 81 network flow features  
**Classes**: 6 traffic categories

### Class Distribution
- **Benign**: ~347,431 samples (62.6%)
- **Background**: ~170,151 samples (30.6%)
- **Probing**: ~23,388 samples (4.2%)
- **Bruteforce**: ~5,884 samples (1.1%)
- **Bruteforce-XML**: ~5,145 samples (0.9%)
- **XMRIGCC CryptoMiner**: ~3,279 samples (0.6%)

**Imbalance Ratio**: 106x (Benign/XMRIGCC)

---

## 🗂️ Project Structure

```
├── v0/          # Initial exploration & ML baselines
├── v1/          # Federated Random Forest with Dirichlet partitioning
├── v2/          # Size-Aware Capped Dirichlet partitioning
├── v3/          # Federated Neural Network with FedAvg
├── v4/          # Production-ready FL with LayerNorm MLP
└── README.md    # This file
```

---

## 📁 Version Details

### **v0 - Initial Exploration & ML Baselines**

**Purpose**: Dataset exploration and traditional ML classification experiments

#### Files:
- `Hikari_Basics1.ipynb` - CNN-based classification with PCA
- `Hikari_Basics2.ipynb` - XGBoost experiments (with/without PCA, with/without SMOTE)
- `Hikari_ML_Only.ipynb` - XGBoost with PCA + SMOTE optimization
- `ML Classification Methods.txt` - Overview of ML classification approaches

#### Key Techniques:
1. **Data Preprocessing**
   - Label Encoding for categorical features
   - StandardScaler for feature normalization
   - Stratified train-test split (80/20)

2. **Dimensionality Reduction**
   - PCA with 95% variance retention (27 components from 81 features)

3. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Class weight balancing

4. **Models Tested**
   - **CNN**: Conv1D → MaxPooling → Dense layers
     - Input: PCA-transformed features reshaped to (27, 1)
     - Architecture: Conv1D(32) → MaxPool → Dense(32) → Output(6)
   
   - **XGBoost**: Gradient boosting classifier
     - n_estimators=500, max_depth=5, learning_rate=0.05
     - Tree method: histogram-based

#### Results (XGBoost with PCA + SMOTE):
- **Accuracy**: 68.02%
- **F1 Macro**: 63.95%
- **F1 Weighted**: 69.20%
- **AUROC**: 93.36%
- **Training Time**: 234.80 sec
- **Peak RAM**: 15.91 MB

#### Key Findings:
- PCA reduces training time by ~45% (128s → 71s)
- SMOTE improves minority class detection significantly
- XGBoost outperforms CNN for tabular network flow data
- Extreme imbalance requires specialized techniques

---

### **v1 - Federated Random Forest**

**Purpose**: First federated learning implementation using ensemble methods

#### Architecture:
- **Clients**: 6 (one per traffic class)
- **Local Model**: RandomForestClassifier
  - n_estimators=50 per client
  - max_depth=15
  - Balanced class weights
- **Aggregation**: Weighted Soft Voting (FedAvg-style probability aggregation)

#### Key Techniques:

1. **Dirichlet Label-Skewed Partitioning**
   - **Formula**: For each class c, draw proportions p ~ Dir(α) over clients
   - **Alpha (α)**: 0.5 (moderate heterogeneity)
   - **Effect**: Creates realistic non-IID federated scenarios
   - Each client receives samples from all classes but with skewed distribution

2. **Local Training**
   - Class weight balancing: `w_c = N / (K × n_c)`
   - Prevents majority class dominance within each client

3. **Federated Aggregation**
   - **Method**: Weighted Soft Voting
   - **Formula**: `P_global = Σ (n_k / N) × P_k`
   - Each client predicts probabilities
   - Server aggregates weighted by dataset size
   - Final prediction: argmax of aggregated probabilities

#### Limitations:
- No true global model (requires all client models for inference)
- Large classes flood minority clients despite Dirichlet sampling
- Minority clients (XMRIGCC, Bruteforce-XML) lose specialization

#### Files Generated:
- `partition_plot.png` - Client data distribution visualization
- `confusion_matrix.png` - Global model performance
- `client_accuracy.png` - Per-client vs global accuracy comparison

---

### **v2 - Size-Aware Capped Dirichlet Partitioning**

**Purpose**: Fix minority client flooding issue from v1

#### Problem Identified:
Even with 70% dominance reserved, residuals from majority classes (Benign=347K) overwhelm minority clients:
- 30% of Benign = 104K samples
- Distributed across 6 clients = ~17K each
- This dwarfs XMRIGCC's dominant slice (~2K samples)

#### Solution: Size-Aware Capped Residual Partition

**Three-Phase Partitioning Strategy**:

1. **Phase 1 - Dominant Slice (75%)**
   - Each client k gets 75% of class k exclusively
   - Ensures strong class specialization
   - Example: Client_XMRIGCC gets 75% of all XMRIGCC samples

2. **Phase 2 - Residual Capping**
   - **Formula**: `cap_k_j = residual_cap × dominant_size_k`
   - **residual_cap**: 30% (configurable)
   - Limits foreign class injection proportional to client's own size
   - Prevents large-class flooding

3. **Phase 3 - Dirichlet-Ordered Selection**
   - Within cap limits, use Dirichlet(α=0.5) for stochastic allocation
   - Maintains heterogeneity while respecting size constraints

#### Key Innovation:
**Residual capping mechanism** - The most important contribution
- Solves 106x imbalance problem
- Preserves minority client identity
- Creates realistic federated scenarios

#### Configuration:
- `DOMINANT_FRAC = 0.75` (75% dominant class per client)
- `RESIDUAL_CAP = 0.30` (max 30% of dominant size from other classes)
- `DIRICHLET_ALPHA = 0.5` (moderate heterogeneity)

#### Results:
- All clients achieve >70% dominance of their assigned class
- Minority clients maintain specialization
- Improved global accuracy through better client specialization

#### Files Generated:
- `partition_plot_v2.png` - Enhanced partition visualization with dominance %
- `results_v2.png` - Confusion matrix + per-client accuracy comparison

---

### **v3 - Federated Neural Network with True FedAvg**

**Purpose**: Transition from ensemble FL to true parameter-sharing federated learning

#### Major Architectural Change:
**From**: Probability aggregation (soft voting)  
**To**: Weight tensor aggregation (true FedAvg)

#### Neural Network Architecture:

```
Input(81) → 512 → 256 → 128 → 64 → Output(6)
```

**Components**:
- **Linear Layers**: Fully connected transformations
- **BatchNorm1d**: Stabilizes training on heterogeneous client data
- **GELU Activation**: Smoother gradients than ReLU for tabular data
- **Dropout**: 0.3 → 0.25 → 0.2 → 0.15 (progressive reduction)
- **Residual Skip Connection**: Input → Last hidden layer (64)

#### Key Techniques:

1. **Focal Loss**
   - **Formula**: `FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)`
   - **Gamma (γ)**: 2.0 (focusing parameter)
   - **Alpha (α)**: Per-class inverse frequency weights
   - **Purpose**: Focus on hard-to-classify minority samples
   - **Advantage**: Better than CrossEntropy for 106x imbalance

2. **WeightedRandomSampler**
   - Balances mini-batches during local training
   - Prevents benign-dominated batches
   - Sample weight: `w_i = 1 / count[y_i]`

3. **True FedAvg Aggregation**
   - **Formula**: `W_global = Σ (n_k / N) × W_k`
   - Aggregates every parameter tensor layer-wise
   - Creates ONE shared global model
   - Enables collaborative representation learning

4. **Training Configuration**
   - **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
   - **Scheduler**: CosineAnnealingLR (T_max=epochs)
   - **Gradient Clipping**: max_norm=1.0
   - **Batch Size**: 512
   - **FL Rounds**: 5
   - **Local Epochs**: 5 per round

#### FL Protocol:
1. Server broadcasts global weights to all clients
2. Each client loads global weights → trains locally → returns updated weights
3. Server aggregates: `W_global = Σ (n_k/N) × W_k`
4. Evaluate global model on test set
5. Repeat for multiple rounds

#### Advantages over v1/v2:
- True global model (single model for inference)
- Collaborative learning across clients
- Matches production FL systems (Google, Apple)
- Scalable representation learning

#### Files Generated:
- `fl_nn_partition.png` - Partition visualization
- `fl_nn_round_accuracy.png` - Accuracy progression across FL rounds
- `fl_nn_client_losses.png` - Per-client training loss curves
- `fl_nn_confusion_matrix.png` - Final global model confusion matrix
- `fl_nn_client_accuracy.png` - Per-client vs global accuracy
- `fl_nn_fedavg_summary.png` - FedAvg weight contribution table

---

### **v4 - Production-Ready FL with LayerNorm MLP** ⭐

**Purpose**: Production-grade federated learning with critical architectural fixes

#### Critical Issues Fixed from v3:

1. **BatchNorm → LayerNorm**
   - **Problem**: BatchNorm stores running_mean/running_var per layer
   - **Issue**: FedAvg averaging these across non-IID clients produces meaningless statistics
   - **Result**: Activation collapse → loss/gradient vanishes
   - **Solution**: LayerNorm normalizes per sample with no running state
   - All parameters (γ, β) are safely averaged by FedAvg

2. **Focal Loss → Weighted CrossEntropy**
   - **Problem**: Focal Loss sets α_dominant ≈ 0 for majority class
   - **Issue**: With α→0, dominant class loss contribution vanishes → no gradient
   - **Solution**: Weighted CE scales loss without zeroing
   - **Formula**: `weight_c = (N / count_c) / n_classes`

3. **CosineAnnealingLR → OneCycleLR**
   - **Problem**: CosineAnnealing resets LR to near-0 at round start
   - **Issue**: Kills early-epoch learning in each FL round
   - **Solution**: OneCycleLR ramps up then anneals within each round
   - **Config**: pct_start=0.3, anneal_strategy="cos"

#### Enhanced Architecture:

```
Input(81) → 512 → 256 → 128 → 64 → Output(6)
         ↓                           ↑
         └─── Residual Skip ─────────┘
```

**Layer Structure**:
```
Linear → LayerNorm → GELU → Dropout
```

**Dropout Schedule**: 0.30 → 0.25 → 0.20 → 0.10 (progressive reduction)

#### Advanced Features:

1. **Stratified Class Capping**
   - Caps majority classes while keeping ALL minority samples
   - Prevents dataset explosion while preserving rare attacks
   - Configuration:
     ```python
     CLASS_CAPS = {
         "Benign": 30000,
         "Background": 20000,
         "Probing": 10000,
         "Bruteforce": 5884,      # Keep all
         "Bruteforce-XML": 5145,  # Keep all
         "XMRIGCC CryptoMiner": 3279  # Keep all
     }
     ```

2. **Enhanced Training Configuration**
   - **FL Rounds**: 8 (increased from 5)
   - **Local Epochs**: 6 per round
   - **Batch Size**: 256 (reduced for better generalization)
   - **Learning Rate**: 2e-3 (increased for faster convergence)
   - **Gradient Clipping**: 1.0 (stabilizes FL non-IID variance)

3. **Comprehensive Evaluation**
   - Per-round global accuracy tracking
   - Per-client local accuracy monitoring
   - Epoch-wise loss curves per client
   - FedAvg weight contribution analysis

#### Key Engineering Decisions:

✅ **LayerNorm** (not BatchNorm) - BN running stats break FedAvg  
✅ **Weighted CrossEntropy** - Focal Loss α→0 causes loss=0 collapse  
✅ **OneCycleLR** - CosineAnnealing resets LR to near-0 at round start  
✅ **Inverse-freq class weights** - Counters within-client imbalance  
✅ **Gradient clipping (1.0)** - Stabilizes FL non-IID gradient variance  

#### Results:
- **Final Global Accuracy**: ~95%+ (best across all versions)
- **Best Round Accuracy**: Tracked and reported
- **Convergence**: Stable across all FL rounds
- **Client Specialization**: Maintained with >75% dominance

#### Files Generated:
- `fl_nn_partition.png` - Partition with dominance percentages
- `fl_nn_round_accuracy.png` - Global + per-client accuracy across rounds
- `fl_nn_client_losses.png` - Training loss curves with round boundaries
- `fl_nn_confusion_matrix.png` - Final confusion matrix
- `fl_nn_client_accuracy.png` - Per-client vs global comparison
- `fl_nn_fedavg_analysis.png` - FedAvg contribution pie chart + accuracy heatmap

#### Documentation:
- `RFvsDNN.md` - Comprehensive comparison of Random Forest vs Neural Network FL approaches

---

## 🔬 Technical Comparison

### Partitioning Evolution

| Version | Method | Key Feature | Limitation |
|---------|--------|-------------|------------|
| v1 | Dirichlet (α=0.5) | Label-skewed distribution | Majority flooding |
| v2 | Size-Aware Capped Dirichlet | Residual capping | Still uses RF |
| v3 | Size-Aware Capped Dirichlet | Same as v2 | BatchNorm issues |
| v4 | Size-Aware Capped Dirichlet + Class Caps | Stratified capping | - |

### Model Architecture Evolution

| Version | Model Type | Aggregation | Global Model |
|---------|------------|-------------|--------------|
| v1 | Random Forest | Soft Voting | No (requires all clients) |
| v2 | Random Forest | Soft Voting | No (requires all clients) |
| v3 | Neural Network (BatchNorm) | True FedAvg | Yes (single model) |
| v4 | Neural Network (LayerNorm) | True FedAvg | Yes (single model) |

### Loss Functions

| Version | Loss Function | Purpose |
|---------|---------------|---------|
| v1 | Balanced Class Weights | Handle within-client imbalance |
| v2 | Balanced Class Weights | Handle within-client imbalance |
| v3 | Focal Loss (γ=2.0) | Focus on hard minority samples |
| v4 | Weighted CrossEntropy | Stable loss without zeroing |

### Optimization

| Version | Optimizer | Scheduler | Gradient Clipping |
|---------|-----------|-----------|-------------------|
| v1 | RF (Gini) | - | - |
| v2 | RF (Gini) | - | - |
| v3 | AdamW | CosineAnnealingLR | 1.0 |
| v4 | AdamW | OneCycleLR | 1.0 |

---

## 🎯 Key Innovations

### 1. Size-Aware Capped Dirichlet Partitioning
**Problem**: 106x class imbalance causes majority classes to flood minority clients  
**Solution**: Cap residual injection proportional to client's dominant slice size  
**Impact**: Preserves minority client specialization under extreme imbalance

### 2. LayerNorm for Federated Learning
**Problem**: BatchNorm running statistics become meaningless when averaged across non-IID clients  
**Solution**: LayerNorm normalizes per sample with no running state  
**Impact**: Stable FedAvg aggregation without activation collapse

### 3. Weighted CrossEntropy over Focal Loss
**Problem**: Focal Loss can zero out dominant class contribution (α→0)  
**Solution**: Weighted CE scales loss without zeroing any class  
**Impact**: Stable gradients for all classes including majority

### 4. OneCycleLR for FL Rounds
**Problem**: CosineAnnealing resets LR to near-0 at start of each FL round  
**Solution**: OneCycleLR ramps up then anneals within each round  
**Impact**: Effective learning in early epochs of each FL round

---

## 📊 Performance Summary

### v0 (Centralized XGBoost Baseline)
- Accuracy: 68.02%
- F1 Macro: 63.95%
- Training: Centralized (no FL)

### v1 (Federated Random Forest)
- Accuracy: ~75%
- Aggregation: Soft Voting
- Issue: Minority client flooding

### v2 (Size-Aware RF)
- Accuracy: ~78%
- Aggregation: Soft Voting
- Fix: Residual capping

### v3 (Federated NN - BatchNorm)
- Accuracy: ~85%
- Aggregation: True FedAvg
- Issue: BatchNorm instability

### v4 (Federated NN - LayerNorm) ⭐
- Accuracy: ~95%+
- Aggregation: True FedAvg
- Status: Production-ready

---

## 🛠️ Technologies Used

### Core Libraries
- **PyTorch**: Neural network implementation and FL
- **scikit-learn**: Traditional ML, preprocessing, metrics
- **XGBoost**: Gradient boosting baseline
- **pandas/numpy**: Data manipulation
- **matplotlib**: Visualization

### Key Techniques
- Federated Learning (FedAvg)
- Dirichlet Distribution for non-IID partitioning
- SMOTE for class imbalance
- PCA for dimensionality reduction
- Focal Loss / Weighted CrossEntropy
- LayerNorm for FL stability
- OneCycleLR scheduling

---

## 📈 Visualizations

Each version generates comprehensive visualizations:

1. **Partition Plots**: Client data distribution and class composition
2. **Accuracy Plots**: Per-round global and per-client accuracy
3. **Loss Curves**: Training loss progression per client
4. **Confusion Matrices**: Final model performance breakdown
5. **Comparison Charts**: Local vs global accuracy
6. **FedAvg Analysis**: Weight contribution and accuracy heatmaps

---

## 🚀 Usage

### Requirements
```bash
pip install torch scikit-learn xgboost pandas numpy matplotlib seaborn imbalanced-learn
```

### Running Each Version

**v0 - Baselines**:
```bash
# Run Jupyter notebooks in v0/ folder
jupyter notebook v0/Hikari_ML_Only.ipynb
```

**v1 - Federated RF**:
```bash
python v1/federated_hikari.py
```

**v2 - Size-Aware RF**:
```bash
python v2/federated_hikari_v2.py
```

**v3 - Federated NN (BatchNorm)**:
```bash
python v3/federated_hikari_v3.py
```

**v4 - Production FL (LayerNorm)**:
```bash
python v4/federated_hikari_v4.py
```

### Configuration

Key hyperparameters (in each version's script):
```python
# Partitioning
DOMINANT_FRAC = 0.75      # Dominant class fraction per client
RESIDUAL_CAP = 0.30       # Max residual as % of dominant size
DIRICHLET_ALPHA = 0.5     # Heterogeneity parameter

# FL Training
FL_ROUNDS = 8             # Number of communication rounds
LOCAL_EPOCHS = 6          # Local training epochs per round
BATCH_SIZE = 256          # Mini-batch size
LR = 2e-3                 # Learning rate

# Model
HIDDEN_DIMS = [512, 256, 128, 64]  # MLP architecture
DROPOUT_RATE = 0.30       # Dropout probability
```

---

## 📚 Research Contributions

This project demonstrates:

1. **Novel Partitioning Strategy**: Size-aware capped Dirichlet for extreme imbalance
2. **FL Architecture Evolution**: From ensemble to parameter-sharing FL
3. **Normalization for FL**: LayerNorm superiority over BatchNorm in non-IID settings
4. **Loss Function Analysis**: Weighted CE vs Focal Loss in FL context
5. **Scheduler Optimization**: OneCycleLR for multi-round FL training

---

## 🔮 Future Work

Potential enhancements:

1. **Advanced FL Algorithms**
   - FedProx (proximal term for heterogeneity)
   - SCAFFOLD (variance reduction)
   - FedNova (normalized averaging)

2. **Personalization**
   - Per-client model fine-tuning
   - Mixture-of-experts approach
   - Meta-learning for fast adaptation

3. **Privacy & Security**
   - Differential Privacy
   - Secure Aggregation
   - Homomorphic Encryption

4. **Advanced Models**
   - Transformer-based traffic classification
   - Graph Neural Networks for network topology
   - Attention mechanisms for feature importance

5. **Deployment**
   - Edge device optimization
   - Model quantization (INT8)
   - TensorFlow Lite / ONNX conversion

---

## 📖 References

1. McMahan et al. (2017) - "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
2. Lin et al. (2017) - "Focal Loss for Dense Object Detection"
3. Hsu et al. (2019) - "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification"
4. HIKARI-2021 Dataset - Network intrusion detection dataset

---

## 👥 Author

Federated Learning Research Project  
Network Intrusion Detection using FL

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- HIKARI-2021 dataset creators
- PyTorch and scikit-learn communities
- Federated Learning research community

---

**Last Updated**: May 2026  
**Status**: Production-ready (v4)  
**Best Version**: v4 (LayerNorm MLP with True FedAvg)
