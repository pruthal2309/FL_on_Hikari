# Federated Learning Detailed Report

## 1. Introduction

This project focuses on implementing Federated Learning (FL) on the ALLFLOWMETER_HIKARI_2021 intrusion detection dataset. 
The primary challenge in this dataset is the extreme class imbalance and non-IID (Non-Independent and Identically Distributed) client distributions.

Two major FL architectures were implemented and analyzed:

1. Federated Random Forest with Weighted Soft Voting Aggregation
2. Federated Neural Network with True FedAvg Weight Aggregation

The goal of this project was:
• To simulate realistic federated learning conditions
• To solve severe class imbalance problems
• To maintain client specialization
• To improve global intrusion detection performance
• To compare ensemble-based FL vs parameter-aggregation FL


## 2. Problem Statement

The dataset contains highly imbalanced traffic classes.

Example:
• Benign Traffic: ~347K samples
• XMRIGCC: ~2.6K samples

Imbalance Ratio:
347000 / 2600 ≈ 106x

This creates multiple problems:
• Majority classes dominate training
• Minority attacks are ignored
• Standard FL partitioning fails
• Simple random splits destroy client specialization
• Global models become biased toward benign traffic

The key challenge was:
“How to create realistic federated clients while preserving minority class identity?”


## 3. Dataset Preprocessing

The following preprocessing techniques were used:

3.1 Label Encoding
Technique Used:
LabelEncoder()

Purpose:
Convert categorical labels into numerical IDs required for ML models.

Alternative Methods:
• One-Hot Encoding
• Ordinal Encoding
• Embedding Encoding

Reason for Choosing LabelEncoder:
Efficient and sufficient for target labels.

3.2 Standardization
Technique Used:
StandardScaler()

Formula:
z = (x - mean) / standard deviation

Purpose:
Normalize feature scales to stabilize training.

Alternative Methods:
• MinMaxScaler
• RobustScaler
• Normalizer

Reason for Choosing StandardScaler:
Best suited for neural network and tabular learning stability.

3.3 Stratified Train-Test Split
Technique Used:
train_test_split(..., stratify=y)

Purpose:
Preserve original class distribution in train and test sets.


## 4. Size-Aware Capped Dirichlet Partitioning

This is the most important innovation in the project.

Problem with Standard Dirichlet Partition:
Large classes flood minority clients.

Example:
30% of Benign residuals could dominate small XMRIGCC clients.

Solution:
Size-Aware Capped Dirichlet Partitioning.

4.1 Dominant Slice Allocation
Each client receives:
75% of its dominant class exclusively.

Example:
Client 0 → Mostly Benign
Client 1 → Mostly Bruteforce
Client 2 → Mostly XMRIGCC

Purpose:
Maintain specialization.

4.2 Residual Injection
Remaining residual samples are distributed across clients.

4.3 Dirichlet Stochastic Allocation
Technique Used:
Dirichlet(alpha=0.5)

Purpose:
Introduce realistic heterogeneity across clients.

Effect of Alpha:
• alpha < 1 → highly non-IID
• alpha = 1 → moderate heterogeneity
• alpha > 10 → nearly IID

Reason for Choosing alpha=0.5:
Provides realistic heterogeneous federated environments.

4.4 Residual Capping Mechanism
Core Formula:
cap = residual_cap × dominant_size

Example:
If dominant slice = 1000 samples
Residual cap = 30%
Maximum foreign samples = 300

Purpose:
Prevent majority-class flooding.

Why This Was Needed:
Without capping:
Minority clients lose their specialization.

Advantages:
• Preserves dominant class identity
• Prevents imbalance collapse
• Creates realistic FL scenarios

Alternative Methods Considered:
• IID splitting
• Oversampling
• Undersampling
• SMOTE
• Random partitioning

Reason for Not Using Them:
They either:
• destroy heterogeneity
• lose information
• introduce synthetic bias
• fail under 106x imbalance


## 5. Federated Random Forest Architecture

The first implementation used Random Forest classifiers on each client.

5.1 Local Training
Each client trained:
RandomForestClassifier(
    n_estimators=60,
    max_depth=15
)

Why Random Forest?
• Strong for tabular data
• Robust to noise
• Handles nonlinear patterns
• Fast local training
• Minimal tuning

Alternative Models:
• XGBoost
• LightGBM
• Logistic Regression
• SVM

Reason for Choosing Random Forest:
Best balance between interpretability, robustness, and performance.

5.2 Class Weight Balancing
Technique Used:
compute_class_weight("balanced")

Purpose:
Increase importance of minority attacks.

Formula:
w_c = N / (K × n_c)

Why Needed:
Without weighting:
Model predicts mostly benign traffic.

How It Works:
Minority class mistakes incur larger penalties.

Alternative Techniques:
• SMOTE
• ADASYN
• Focal Loss
• Weighted Sampling

Reason for Choosing Class Weights:
Simple, efficient, and directly compatible with Random Forest.

5.3 Weighted Soft Voting Aggregation
Problem:
Random Forest trees cannot be averaged mathematically.

Solution:
Aggregate prediction probabilities instead of parameters.

Working:
1. Each client predicts probabilities
2. Server weights predictions by client dataset size
3. Global probability is computed
4. Highest probability class selected

Formula:
P_global = Σ (n_k / N) × P_k

Advantages:
• Supports heterogeneous models
• Preserves specialization
• Stable aggregation

Limitation:
No true global model exists.

Inference Process:
Each test sample is sent to all client models.
All clients return probabilities.
Server performs weighted soft voting.


## 6. Federated Neural Network with True FedAvg

The second implementation upgraded the system into true federated learning.

6.1 Neural Network Architecture
Architecture:
Input(81) → 512 → 256 → 128 → 64 → Output(6)

Components Used:
• Linear Layers
• BatchNorm
• GELU Activation
• Dropout
• Residual Skip Connections

Purpose of Each:
BatchNorm:
Stabilizes heterogeneous FL training.

GELU:
Provides smoother gradients than ReLU.

Dropout:
Reduces overfitting on skewed local datasets.

Residual Connections:
Improve gradient flow in deep networks.

6.2 Focal Loss
Formula:
FL(p_t) = -alpha × (1 - p_t)^gamma × log(p_t)

Purpose:
Focus training on difficult minority samples.

Why Better than Class Weights:
• Hard examples receive more attention
• Easy benign examples contribute less
• More suitable for extreme imbalance

6.3 WeightedRandomSampler
Purpose:
Balance mini-batches during local training.

Problem Solved:
Without sampling:
Mini-batches may contain mostly benign traffic.

6.4 True FedAvg Aggregation
This implementation uses classical Federated Averaging.

Formula:
W_global = Σ (n_k / N) × W_k

Working:
1. Global server initializes model
2. Global weights sent to all clients
3. Clients train locally
4. Clients return updated weights
5. Server averages tensors layer-wise
6. New global model generated
7. Process repeated across multiple FL rounds

Key Difference from Soft Voting:
This creates ONE shared global model.

6.5 Communication Rounds
The system uses:
• Multiple FL rounds
• Local fine-tuning
• Global synchronization

This enables collaborative representation learning.


## 7. Comparison Between Both Systems

Random Forest FL:
• Ensemble-based
• No global model
• Probability aggregation
• Inference requires all client models

Neural FedAvg FL:
• Parameter aggregation
• Shared global model
• Collaborative learning
• Centralized inference possible

Why Neural FedAvg Is Closer to Real FL:
• Matches Google-style FL systems
• Uses true weight averaging
• Supports scalable representation learning


## 8. Key Research Contributions

The project demonstrates several advanced FL concepts:

• Size-Aware Capped Dirichlet Partitioning
• Residual capping for extreme imbalance
• Specialized non-IID client simulation
• Federated ensemble learning
• Transition from ensemble FL to true FedAvg
• Focal loss integration for intrusion detection
• Weighted sampling in FL environments

The strongest contribution is:
Residual-capped non-IID partitioning under 106x imbalance.


## 9. Future Improvements

Potential future enhancements:

• FedProx
• Scaffold
• Personalized FL
• Differential Privacy
• Secure Aggregation
• Homomorphic Encryption
• Transformer-based traffic classification
• Graph Federated Learning

These methods can improve:
• privacy
• convergence stability
• personalization
• robustness


## 10. Conclusion

This project successfully implemented and analyzed two federated learning systems for intrusion detection under severe class imbalance.

Major achievements:
• Realistic non-IID federated simulation
• Effective imbalance handling
• Specialized client learning
• Federated aggregation comparison
• Transition from ensemble FL to true parameter-sharing FL

The Random Forest system demonstrated strong practical ensemble learning, while the Neural FedAvg system implemented true collaborative federated representation learning.

The final system is highly aligned with modern federated learning research and production-grade FL architectures.


