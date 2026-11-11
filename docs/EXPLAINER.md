# Explainer: how this project works and why it is built this way

This document is the study guide for the project. Read it top to bottom once; after that the section headings work as a lookup table for viva questions.

Contents

1. [The problem in one paragraph](#1-the-problem-in-one-paragraph)
2. [The pipeline, step by step](#2-the-pipeline-step-by-step)
3. [How graph neural networks work](#3-how-graph-neural-networks-work)
4. [The evaluation protocol](#4-the-evaluation-protocol)
5. [Class imbalance, thresholds and the t=43 shift](#5-class-imbalance-thresholds-and-the-t43-shift)
6. [Design decisions and their reasons](#6-design-decisions-and-their-reasons)
7. [What the results show](#7-what-the-results-show)
8. [Explainability](#8-explainability)
9. [Limitations and ethics](#9-limitations-and-ethics)
10. [25 likely viva and interview questions](#10-25-likely-viva-and-interview-questions)

---

## 1. The problem in one paragraph

Bitcoin transactions form a graph: when transaction B spends an output of transaction A, there is an edge A → B. The Elliptic dataset contains 203,769 such transactions and 234,355 edges, spread over 49 time steps of about two weeks each. For each transaction we get 165 anonymised numeric features. About 2% are labelled **illicit** (4,545), 21% **licit** (42,019) and 77% have no label (157,205). The task is binary node classification: given a transaction's features and its position in the graph, predict whether it is illicit. The project asks three questions: do graph neural networks (GNNs) beat strong table models, do GNN outputs improve table models, and how badly do models degrade over time.

---

## 2. The pipeline, step by step

`python run_all.py --config configs/full.yaml` runs these phases in order. Each is also a module you can run alone with `python -m btc_aml.<module> --config ...`.

| Phase | Module | What it does | What it writes |
|---|---|---|---|
| data | `data.py` | Downloads the three CSVs (Kaggle API, falling back to PyG's downloader), checks every dataset fact, converts to tensors | `data/processed/elliptic_bundle.pt` |
| eda | `eda.py` | Counts labels per time step, degree distribution, class balance | 3 figures, `dataset_stats.json` |
| baselines | `train_baselines.py` | Logistic Regression, Random Forest, LightGBM, MLP on LF and AF | rows in `metrics.csv` |
| gnn | `train_gnn.py` | GCN, Skip-GCN, GraphSAGE, GAT on LF and AF | rows in `metrics.csv`, checkpoints in `models/` |
| ablation | `train_gnn.py --ablation` | Best GNN with directed and bidirectional edges | rows in `metrics.csv` |
| hybrid | `hybrid.py` | Tree models on features + GNN output | rows in `metrics.csv` |
| rolling | `rolling.py` | Retrains tree models before every test step | rows in `metrics.csv` |
| random_split | `random_split.py` | One clearly-labelled run with a random split | rows in `metrics.csv` |
| analysis | `analysis.py` | Aggregates over seeds, builds tables and figures | `summary.csv`, `tables.md`, figures |
| explain | `explain.py` | SHAP, permutation importance, GNNExplainer | figures, `feature_importance.csv` |

### 2.1 From CSV to tensors (`data.py`)

The features file has no header and 167 columns: column 0 is the transaction id, column 1 the time step, columns 2–94 are 93 **local** features (properties of the transaction itself) and columns 95–166 are 72 **aggregated** features (statistics of the transaction's one-hop neighbours, computed by the dataset's authors). We name them `local_01…local_93` and `agg_01…agg_72`.

`build_bundle` produces five aligned tensors: `tx_id`, `time_step`, `x` (203,769 × 165), `y` and `edge_index`. Two details matter:

- **Label encoding.** The CSV uses `"1"` (illicit), `"2"` (licit), `"unknown"`. We map them to 1, 0 and −1, so the rare class we care about is the positive class, which is what scikit-learn's metrics expect.
- **Node indices.** Transaction ids are arbitrary integers. A GNN library addresses nodes by position, so each id is mapped to its row number 0…N−1 and the edge list is rewritten in those indices.

`validate_bundle` then asserts every fact about the dataset (node, edge and label counts, time steps 1–49, no edge between different time steps, each time step is exactly one connected component). The same facts are asserted again in `tests/test_data.py`. A truncated download would otherwise give plausible but wrong results.

### 2.2 Two feature sets

- **LF** = the 93 local features.
- **AF** = all 165 features (local + aggregated).

The transaction id and the time step are never model inputs. The id carries no meaning, and the time step would let a model memorise "period X had many illicit transactions", which cannot transfer to a future period.

### 2.3 Splitting by time (`splits.py`)

| Split | Time steps | Role |
|---|---|---|
| train | 1–29 | fit model weights |
| validation | 30–34 | choose hyperparameters, early-stopping epoch, decision threshold |
| test | 35–49 | final score, evaluated once per model |

Because no edge crosses time steps, the graph of a split is just its nodes plus the edges among them (the induced subgraph). Nothing is cut and nothing leaks through an edge. `make_graph` builds one PyG `Data` object per split with:

- `x`: scaled features, `edge_index`: edges renumbered inside the split,
- `y`: labels including −1 for unknown,
- `labeled`: boolean mask, true where the label is known,
- `node_idx`: each node's index in the full dataset, to map predictions back.

Unknown-label nodes stay in the graph. They have features and they connect labelled nodes, so the GNN needs them for message passing. They are excluded from the loss and from every metric through the `labeled` mask.

### 2.4 Scaling

Features are standardised, $z = (x - \mu) / \sigma$, with $\mu$ and $\sigma$ computed **only on nodes of the training time steps** (`fit_scaler`). Statistics of later time steps never enter. A test (`test_scaler_uses_training_nodes_only`) changes all non-training rows by +1000 and checks that the scaler does not move.

### 2.5 Training and evaluating

Every model goes through the same function, `protocol.run_protocol`, described in [section 4](#4-the-evaluation-protocol). Each model family only supplies two methods: `fit_select` (train on steps 1–29) and `fit_final` (train on steps 1–34). This is the single most important design choice for correctness: there is exactly one place where test labels are read.

### 2.6 What a result row looks like

`results/metrics.csv` has one row per (experiment, model, feature set, variant, seed, split). `split` is `val` (model trained on 1–29, scored on 30–34) or `test` (final model, scored on 35–49). `results/per_timestep.csv` has the same keys plus one row per test time step. `results/predictions/*.npz` stores every test node's score, so plots and the app never retrain anything.

---

## 3. How graph neural networks work

### 3.1 Message passing

A table model sees one row. A GNN lets each node also look at its neighbours. One GNN layer does three things for every node $i$:

1. **Collect** the current vectors $h_j$ of its neighbours $j \in N(i)$.
2. **Aggregate** them into one vector (sum, mean, weighted sum…).
3. **Update**: combine the aggregate with the node's own vector, apply a learned linear map and a non-linearity.

In general form:

$$h_i^{(l+1)} = \sigma\Big(\text{UPDATE}\big(h_i^{(l)},\ \text{AGGREGATE}\{h_j^{(l)} : j \in N(i)\}\big)\Big)$$

with $h_i^{(0)} = x_i$ (the input features). After one layer a node knows about its direct neighbours. After two layers it knows about neighbours of neighbours: its **2-hop neighbourhood**. All models here use two layers, and the test `test_two_layers_see_exactly_two_hops` checks that changing a node 3 hops away does not change the output, while changing one 2 hops away does.

The same weights are used at every node. That is why a GNN trained on the graph of time steps 1–34 can be applied to the unseen graph of steps 35–49 (**inductive** learning): the weights describe how to combine a node with its neighbours, not which nodes exist.

The four architectures differ only in the aggregation step.

### 3.2 GCN (Graph Convolutional Network)

$$H^{(l+1)} = \sigma\big(\hat{A}\, H^{(l)}\, W^{(l)}\big), \qquad \hat{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2}$$

$A$ is the adjacency matrix, $A + I$ adds a self-loop so a node keeps its own information, and $\tilde{D}$ is the degree matrix of $A + I$. Per node:

$$h_i^{(l+1)} = \sigma\Big(W^{(l)} \sum_{j \in N(i) \cup \{i\}} \frac{h_j^{(l)}}{\sqrt{d_i\, d_j}}\Big)$$

The weight of a neighbour is fixed by the degrees: $1/\sqrt{d_i d_j}$. Nothing about how to weigh neighbours is learned, and the node's own vector is treated as just one more neighbour. If a node has many neighbours, its own features are averaged away.

### 3.3 Skip-GCN

GCN plus a shortcut from the raw input features $X$ into each layer:

$$H^{(l+1)} = \sigma\big(\hat{A}\, H^{(l)}\, W^{(l)} + X\, \tilde{W}^{(l)}\big)$$

The extra term $X \tilde{W}$ guarantees the node's own features reach every layer undiluted. This matters on Elliptic because the node features are already very informative. (In the original paper the shortcut feeds only the output layer; here it feeds each of the two hidden layers, because our output layer is a separate linear classifier. See decision D10.)

### 3.4 GraphSAGE

$$h_i^{(l+1)} = \sigma\Big(W_{\text{self}}^{(l)}\, h_i^{(l)} + W_{\text{neigh}}^{(l)} \cdot \frac{1}{|N(i)|}\sum_{j \in N(i)} h_j^{(l)}\Big)$$

Two differences from GCN: the neighbours are combined by a plain **mean**, and the node's own vector gets its **own weight matrix** $W_{\text{self}}$ instead of being mixed into the average. The model can therefore learn "mostly trust my own features, adjust a little using the neighbourhood", which GCN cannot express.

### 3.5 GAT (Graph Attention Network)

GAT learns how much attention node $i$ pays to each neighbour $j$:

$$e_{ij} = \text{LeakyReLU}\big(a^{\top}[\,W h_i \,\|\, W h_j\,]\big), \qquad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in N(i) \cup \{i\}} \exp(e_{ik})}$$

$$h_i^{(l+1)} = \sigma\Big(\sum_{j \in N(i) \cup \{i\}} \alpha_{ij}\, W h_j\Big)$$

$\|$ is concatenation and $a$ is a learned vector. The softmax makes the attention weights of a node sum to 1. We use 4 attention **heads**: four independent copies of this mechanism whose outputs are concatenated, which stabilises training. GAT has the most freedom and, with few labels, the most room to overfit.

### 3.6 From embeddings to a prediction

After two message-passing layers each node has a hidden vector (the **embedding**, size 64 or 128 here). A final linear layer turns it into two logits, and softmax turns logits into probabilities:

$$P(\text{illicit} \mid i) = \frac{e^{z_{i,1}}}{e^{z_{i,0}} + e^{z_{i,1}}}$$

Training minimises **weighted cross-entropy** over labelled nodes only:

$$\text{loss} = -\frac{1}{\sum_{i \in \text{labelled}} w_{y_i}} \sum_{i \in \text{labelled}} w_{y_i}\, \log P(y_i \mid i)$$

where $y_i$ is the true class of node $i$ and $w_c$ is the weight of class $c$ ([section 5.1](#51-class-weights)). Optimiser: Adam. Regularisation: dropout on hidden vectors and weight decay (L2).

### 3.7 Full-batch training

Each training step runs the model on the entire training graph at once (`neural.fit`). The graph is small enough (136,265 nodes for steps 1–34) that no neighbour sampling or mini-batching is needed, which also removes the dependence on PyG's optional compiled extensions.

### 3.8 Edge direction

An edge A → B means money flows from A to B. Three treatments (`splits.prepare_edges`, `models.GNN`):

- **undirected**: add the reverse of every edge. Information flows both ways through one set of weights. Default.
- **directed**: keep edges as given. A node hears only from the transactions that funded it.
- **bidirectional**: PyG's `DirGNNConv` runs one convolution along the edges and a second one against them, each with its own weights, and averages them. The model can then treat "who paid me" and "whom I paid" differently.

---

## 4. The evaluation protocol

### 4.1 The two phases (`protocol.py`)

**Phase 1, selection** (train = steps 1–29, validation = steps 30–34)

a. *Hyperparameters.* For every candidate in the config grid, train with the first seed and keep the candidate with the best validation PR-AUC. Every candidate's score is logged to `results/hparam_search.csv`.

b. *Per seed.* Train on steps 1–29. Models with early stopping (LightGBM, MLP, GNNs) report how many rounds or epochs were best on validation. Choose the decision threshold that maximises illicit F1 on validation.

**Phase 2, final** (trainval = steps 1–34, test = steps 35–49)

c. *Per seed.* Retrain from scratch on steps 1–34 with the chosen hyperparameters and the chosen number of rounds/epochs. There is no validation data left to early-stop on, so the count from phase 1 is reused.

d. Score the test steps **once**, with the threshold from step b.

Results are reported as mean ± sample standard deviation over 5 seeds.

### 4.2 Why a temporal split

In production, the model is trained on the past and scores transactions that arrive later. A random split breaks this in two ways:

1. **Same-period leakage.** Train and test rows come from the same fortnights, so the model is tested on the exact behaviour it trained on. Real fraud patterns change.
2. **Graph leakage.** Each time step is one connected component. With a random split, a test node's neighbours are training nodes with known labels, and the aggregated features of a test node were computed from those same neighbours.

The project measures the size of this effect in one clearly-labelled run (`random_split.py`); see [section 7.6](#76-random-split-versus-temporal-split).

### 4.3 Why not accuracy

Counting how often the model is right rewards predicting the majority class. `eda.py` computes what "predict licit for everything" would score (`results/dataset_stats.json`):

| Split | Illicit share of labelled | Accuracy of "everything is licit" | Illicit F1 |
|---|---|---|---|
| train (1–29) | 10.88% | 89.12% | 0 |
| validation (30–34) | 16.82% | 83.18% | 0 |
| test (35–49) | 6.50% | 93.50% | 0 |

A model that catches no criminal at all is 93.50% accurate on the test set. Any accuracy below that is worse than doing nothing, and accuracies above it look far more impressive than they are. So accuracy (equal to micro-F1 for this task) appears in the tables only for comparison with the original paper.

### 4.4 The metrics we do use

With TP = illicit correctly flagged, FP = licit wrongly flagged, FN = illicit missed:

$$\text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

- **Precision**: of the transactions we flagged, how many are really illicit. Low precision means innocent users get flagged.
- **Recall**: of the illicit transactions, how many we caught.
- **F1**: harmonic mean. It is high only when both are high, so a model cannot game it by flagging everything or nothing.
- **PR-AUC** (average precision): area under the precision–recall curve over all thresholds. It measures how well the model **ranks** illicit above licit, independent of any threshold. Its baseline for a random ranking is the illicit share (0.065 on test), so it stays honest under imbalance.
- **ROC-AUC**: probability that a random illicit transaction is scored above a random licit one. Reported, but it is optimistic under imbalance because the huge number of true negatives keeps the false-positive *rate* small even when false positives outnumber true positives.

True negatives appear in none of precision, recall, F1 or PR-AUC. That is exactly why these metrics are not inflated by the licit majority.

### 4.5 Per-time-step F1 and the alert budget

The overall test score hides *when* a model fails, so `evaluate.per_time_step_metrics` computes illicit precision, recall and F1 for each of the 15 test steps.

The **alert budget** metric models an analyst team with limited capacity. In each time step, take the K labelled transactions with the highest scores (K = 50 by default, `alert_k` in the config):

$$\text{precision@K} = \frac{\text{illicit among the top K}}{K}, \qquad \text{recall@K} = \frac{\text{illicit among the top K}}{\text{all illicit in the step}}$$

This needs no threshold: it only asks whether the model's ranking puts illicit transactions at the top of the review queue. Note that precision@K is capped by how many illicit transactions exist: a step with 5 illicit transactions cannot exceed 5/50 = 0.10.

---

## 5. Class imbalance, thresholds and the t=43 shift

### 5.1 Class weights

About 11% of labelled training transactions are illicit. An unweighted loss can be made small by mostly ignoring them. Every model therefore uses "balanced" class weights:

$$w_c = \frac{n_{\text{samples}}}{2 \cdot n_c}$$

so both classes contribute equally to the loss in total. For neural models this is the `weight` argument of cross-entropy (`neural.class_weights`); for scikit-learn models it is `class_weight="balanced"`; Random Forest uses `"balanced_subsample"`, which recomputes weights inside each tree's bootstrap sample.

We do not oversample (e.g. SMOTE). Synthetic nodes would have no place in the graph, and for trees class weights achieve the same effect without inventing data.

### 5.2 Threshold tuning

A classifier outputs a probability; a decision needs a cut-off. The default 0.5 is rarely right here, for two reasons: class weighting deliberately inflates the predicted illicit probabilities, and the best trade-off between precision and recall depends on the class ratio. `evaluate.best_f1_threshold` computes the precision–recall curve on **validation** scores and returns the threshold with the highest F1. That one number is then frozen and applied to the test set.

Tuning the threshold on the test set would make the reported F1 the best of thousands of tries on the data it is judged on. PR-AUC and precision@K do not depend on the threshold at all, which is why they are reported next to F1.

One weakness to know: the threshold is tuned on steps 30–34, where 16.82% of labelled transactions are illicit, and applied to steps 35–49, where 6.50% are. A threshold chosen for one class ratio is not optimal for another.

### 5.3 The dark-market shutdown at t=43

According to the original paper, a dark market was closed at time step 43. In the data this shows up as a sharp drop in illicit transactions: steps 43–49 contain 24, 24, 5, 2, 22, 36 and 56 labelled illicit transactions, against 239 in step 42 (`results/labels_per_time_step.csv`).

This is **distribution shift**: the relationship between features and label changes. The illicit transactions the models learned to recognise (those associated with the market) disappear, and the illicit activity that remains looks different. A model trained on steps 1–34 cannot know this. [Section 7.4](#74-degradation-over-time-and-the-t43-collapse) shows how large the effect is and what rolling retraining recovers.

---

## 6. Design decisions and their reasons

| # | Decision | Reason |
|---|---|---|
| D1 | Python 3.12 virtual environment, CPU | The machine's default Python (3.15) is newer than PyTorch supports. No CUDA GPU is present; `device: auto` uses CUDA when one exists. Apple's MPS backend is opt-in (`device: mps`) because CPU is fast enough and behaves identically on Windows and Linux. |
| D2 | Illicit = 1, licit = 0, unknown = −1 | The class of interest is the positive class, matching the convention of all metric functions. |
| D3 | Build our own graph from the raw CSVs | PyG's `EllipticBitcoinDataset` drops the time-step column, which the whole protocol depends on. It is used only as a fallback downloader. |
| D4 | One shared `run_protocol` for all models | Guarantees identical treatment and a single place where test labels are read. |
| D5 | Final models retrained on steps 1–34 | Matches the original paper's 34/15 split, and a deployed model would use all labelled history. |
| D6 | Early-stopping count from phase 1 reused in phase 2 | No validation data remains in phase 2. Reusing the count is simple and standard; it slightly under-trains because phase 2 has 13% more labelled data. |
| D7 | Threshold from phase 1 applied to the phase-2 model | Same reason. The alternative (holding out part of 1–34 again) would mean the final model is not trained on all of 1–34. |
| D8 | Hyperparameter search uses the first seed only, criterion = validation PR-AUC | Searching with all seeds would multiply the cost by five. PR-AUC is threshold-free, so the search does not depend on a threshold choice. |
| D9 | Scaler fitted on all nodes of the training steps, including unlabelled ones | Their features are available at training time and no label is involved. Fitting on later steps would leak. |
| D10 | GNN = two message-passing layers of the same hidden size + a separate linear classifier | Gives every architecture a well-defined "penultimate-layer embedding" of the same size for the hybrid experiment. Consequence: Skip-GCN's shortcut feeds both hidden layers rather than the output layer as in the paper. |
| D11 | Balanced class weights instead of the paper's 0.3/0.7 | 0.3/0.7 was tuned by the paper's authors for their GCN; balanced weights follow from the data and are used consistently by every model. |
| D12 | Full-batch, inductive training | The graph fits in memory, so sampling adds complexity without benefit; and avoiding it avoids the optional compiled PyG extensions (clean Windows install). |
| D13 | Undirected edges as the default | The standard choice for GCN-style models. Whether it is the best choice is tested by the ablation, not assumed. |
| D14 | `DirGNNConv` with an explicit self-weight | `DirGNNConv` switches off the wrapped layer's own self-connection. Its built-in replacement assumes input and output sizes that do not match for GAT (heads), so the model adds its own `Linear` for the node's own vector. |
| D15 | "Best GNN" for ablation, hybrid, app is chosen by **validation** PR-AUC | Choosing by test score would be model selection on the test set. |
| D16 | Two hybrid variants: embeddings (`emb`) and out-of-fold GNN score (`oof`) | The embedding variant is the classic recipe and was requested. It has a flaw (see [7.3](#73-hybrid-does-gnn-output-help-tree-models)), which the second variant removes. |
| D17 | Out-of-fold variant stacks a probability, not an embedding | Embeddings of different GNNs do not share a coordinate system; a probability means the same thing for all of them. |
| D18 | Rolling retraining keeps hyperparameters, round count and threshold fixed | Isolates the effect of the training window, and avoids any tuning on test data. Run with 3 seeds instead of 5 to save time. |
| D19 | Alert budget computed among **labelled** transactions | Without a label we cannot say whether an alert was correct. In reality the top K would be drawn from all transactions. |
| D20 | SHAP and permutation importance computed on a test sample | We want to know what the final models rely on when scoring new data. This is interpretation after the fact; nothing is tuned with it. |
| D21 | GNNExplainer run on the transaction's whole time step | Each time step is a separate connected component, so this is exact. Cutting out only the 2-hop subgraph would change GCN's degree normalisation at the border. |
| D22 | LightGBM capped at one thread on macOS | PyTorch and LightGBM each load their own OpenMP runtime on macOS and crash when both use threads. See `btc_aml/__init__.py`. Linux and Windows are not affected. |
| D23 | MLP trained through the same loop as the GNNs | The MLP is "the GNN without the graph". Sharing the loop means any difference comes from message passing, not from a different optimiser or loss. |
| D24 | Before/after-shutdown scores computed by pooling predictions | F1 cannot be averaged over time steps: a step with 2 illicit transactions would count as much as one with 239. |
| D25 | Sample standard deviation (ddof = 1) over seeds; single-seed runs show no ± | With one seed there is no spread to report, and showing "± 0.000" would be misleading. Logistic Regression is deterministic, so its ± 0.000 over 5 seeds is real. |
| D26 | `run_all.py` archives results made with a different config | A quick test run cannot silently destroy hours of full-run results. |
| D27 | No batch normalisation in the GNNs | It would add statistics estimated on the training period, another thing that can go stale under distribution shift, and it was not needed for stable training. |
| D28 | Every GNN explanation is checked by re-running the model (all edges removed; only marked edges kept) | A mask is only a claim about the model. Re-running the model tests the claim, and it caught a misleading drawing during development (section 8.5). |

**A note on how the test set was used during development.** While building the pipeline, the quick config (1 seed) was run several times and its test numbers were visible in the logs. The full config's search grids and all protocol choices were fixed before the full run was started, and nothing was changed after seeing full-run test results. The out-of-fold hybrid variant was added after the embedding variant lowered **validation** PR-AUC (validation and test both dropped; the validation drop alone motivates the change). Three things were added after the full run, none of which retrains a model or changes a reported result: the threshold diagnostic and the graph-reliance table (both re-score saved models or saved predictions), and the corrected GNN explanations.

---

## 7. What the results show

All numbers in this section come from the full run (`configs/full.yaml`, 5 seeds, CPU) and can be found in `results/tables.md`, which is generated from `results/metrics.csv`. "±" is the sample standard deviation over seeds. Numbers attributed to "the paper" are read from Tables 1 and 2 of Weber et al. (2019).

### 7.1 Question 1: do GNNs beat strong non-graph models?

**No. On this data and protocol the tree models win clearly, on both feature sets, in illicit F1 and in PR-AUC.**

| Model | Features | Illicit F1 | PR-AUC |
|---|---|---|---|
| Random Forest | AF | 0.813 ± 0.011 | 0.797 ± 0.001 |
| LightGBM | AF | 0.806 ± 0.011 | 0.805 ± 0.001 |
| Random Forest | LF | 0.789 ± 0.004 | 0.789 ± 0.001 |
| Skip-GCN (best GNN by test F1) | AF | 0.652 ± 0.019 | 0.643 ± 0.019 |
| GraphSAGE (best GNN by validation PR-AUC) | AF | 0.642 ± 0.025 | 0.615 ± 0.023 |
| MLP | AF | 0.630 ± 0.044 | 0.595 ± 0.044 |

- The best GNN is about 0.16 F1 below Random Forest on the same features. The gap is many times larger than the seed-to-seed spread of either model.
- Random Forest on the 93 local features alone (0.789) beats every GNN on all 165 features.
- **The MLP is the telling comparison.** It shares the GNNs' loss, optimiser and training loop and has no graph. On AF it scores 0.630 ± 0.044; the two best GNNs score 0.652 ± 0.019 and 0.642 ± 0.025. That difference is within the noise. On LF the MLP (0.655 ± 0.098) is not behind GraphSAGE (0.626 ± 0.029) or GAT (0.629 ± 0.041) either. So most of the gap to the trees is "neural network versus trees on tabular features", not "graph versus no graph".
- Plain GCN is by far the weakest GNN on LF (0.337 ± 0.068). Giving the node its own path helps: Skip-GCN (0.585) and GraphSAGE (0.626) are both well above it. On AF, GAT is the weakest (0.510 ± 0.063).
- ROC-AUC hides most of this: it ranges only from 0.867 to 0.940 across all GNNs and tree models, and Skip-GCN on LF (0.916) even edges past Random Forest on LF (0.913). This is the reason ROC-AUC is not a headline metric here.

**"And when?"** We looked for any setting where GNNs come out ahead:

- *Before the shutdown* (steps 35–42): Random Forest on AF 0.906 ± 0.006, best GNN (Skip-GCN on AF) 0.753 ± 0.024. Trees win.
- *After the shutdown* (steps 43–49): every model is at or below 0.054 F1. Nothing wins.
- *Alert budget after the shutdown*: the GNNs on AF put slightly more illicit transactions in the top 50 than the trees (GraphSAGE precision@50 0.088 against Random Forest 0.034 and LightGBM 0.062). But Logistic Regression (0.097) and the MLP (0.095) do the same, so this is not an effect of the graph, and all of these values are very low.

We found no setting in which a GNN beats the tree models.

The precision–recall curves (`results/figures/pr_curves.png`, first seed, AF) show the same thing in one picture. The tree models keep precision near 1.0 until they have found about 70% of the illicit transactions. The MLP and GraphSAGE curves start low: some of their most confident flags are licit transactions.

**Is it just the threshold?** No. The threshold diagnostic recomputes F1 with the best possible threshold *for the test set* (an upper bound, not a valid result). Some models lose a lot to threshold transfer (GCN on LF: +0.122 with the oracle threshold; MLP on LF: +0.102), the trees almost nothing (Random Forest on AF: +0.014). But even with an oracle threshold no GNN exceeds 0.671 F1, far below the trees' honestly-reported 0.81. The gap is in the ranking itself, which PR-AUC confirms.

**Do the GNNs even use the graph?** `results/graph_reliance.csv` scores every trained GNN twice, once normally and once with all edges removed (nothing retrained):

- On **validation** the graph clearly helps every GNN. GraphSAGE on AF: PR-AUC 0.955 ± 0.001 with edges, 0.763 ± 0.008 without.
- On **test** that advantage is gone. In four of the eight model/feature combinations the PR-AUC is *higher* without edges (GCN on LF: 0.466 with, 0.607 without; GAT on LF: 0.631 with, 0.700 without; GraphSAGE on AF: 0.615 with, 0.650 without; GAT on AF: 0.570 with, 0.637 without), and in the other four it is lower. Only 2.5–4.7% of test decisions change when edges are removed.

Reading: the GNNs do learn to use neighbourhoods, and that knowledge works on data close to the training period. It does not carry over reliably to later time steps. What they learn about a transaction's own features transfers better than what they learn about its neighbourhood. (Caveat: scoring without edges is a condition the model never saw in training, so this measures reliance on the graph, not what a graph-free model would score. The MLP row answers that.)

**Why do the GNNs lose?** The evidence above supports three reasons; the last two points are interpretation.

1. *The node features already carry most of the signal*, and trees are better than neural networks at exploiting such tabular features (MLP ≈ GNN ≪ trees).
2. *Neighbourhood patterns do not transfer over time* (the graph-reliance result).
3. *The graph is sparse*: the mean degree is 2.30 and the median is 2 (`results/dataset_stats.json`), and 77% of nodes are unlabelled, so there is little neighbourhood to aggregate.
4. *Few labels for a neural model*: 3,462 illicit training transactions, and a five-step validation period that makes early stopping and thresholds noisy (GNN standard deviations are several times those of the trees).

**Comparison with the original paper.** The ranking agrees: the paper also found Random Forest strongest. Illicit F1, paper → this project: Random Forest on AF 0.788 → 0.813; Skip-GCN 0.705 → 0.652; GCN 0.628 → 0.536; MLP on AF 0.653 → 0.630; Logistic Regression on AF 0.481 → 0.439. The setups differ, so the numbers are not expected to match: the paper trains on the first 34 time steps without describing a validation split, uses Random Forest with 50 trees and 50 max features, a 2-layer GCN with embedding size 100 trained for 1000 epochs with class weights 0.3/0.7, and reports single numbers. We choose thresholds and epochs on a held-out validation period and report means over five seeds. The paper's micro-averaged F1 for Random Forest on AF is 0.977; ours is 0.978 ± 0.002.

### 7.2 LF versus AF: can a GNN learn the aggregated features by itself?

The 72 aggregated features are hand-made summaries of each transaction's one-hop neighbourhood. A GNN layer computes a learned summary of the same neighbourhood. So in principle a GNN on LF could rebuild what AF hands to the trees.

- What AF is worth to the trees: +0.024 F1 for Random Forest (0.789 → 0.813) and +0.036 for LightGBM (0.769 → 0.806). A real but modest gain.
- GNNs on LF: the best reach 0.629 ± 0.041 (GAT) and 0.626 ± 0.029 (GraphSAGE). That is below the MLP on LF (0.655 ± 0.098), far below Random Forest on LF (0.789) and further still from the trees on AF.

**So no: in this project message passing on LF does not recover what the aggregated features give the tree models.** The graph-reliance table explains the mechanism: on validation, GraphSAGE on LF does profit from its edges (PR-AUC 0.957 with, 0.819 without), so it learned neighbourhood information; on the test period that information stops helping (0.654 with, 0.544 without for GraphSAGE; but 0.466 with, 0.607 without for GCN).

Other observations from the LF/AF table: GCN gains the most from AF (+0.200 F1), which fits its weakness of diluting a node's own features; GAT gets *worse* with AF (−0.118 F1), consistent with the most flexible model overfitting when given more inputs; SHAP attributes 69.7% of LightGBM's total attribution to local features and 30.3% to aggregated ones (`results/feature_importance.csv`).

### 7.3 Hybrid: does GNN output help tree models?

**With the classic recipe (embeddings as extra columns): no, it hurts. With an out-of-fold GNN score: a small gain on AF.**

| Tree | Features | Features only | + GNN embeddings | + out-of-fold GNN score |
|---|---|---|---|---|
| Random Forest | AF | 0.813 ± 0.011 | 0.652 ± 0.019 | 0.827 ± 0.005 |
| LightGBM | AF | 0.806 ± 0.011 | 0.652 ± 0.015 | 0.815 ± 0.002 |
| Random Forest | LF | 0.789 ± 0.004 | 0.696 ± 0.013 | 0.795 ± 0.013 |
| LightGBM | LF | 0.769 ± 0.016 | 0.716 ± 0.012 | 0.686 ± 0.091 |

(Illicit F1. GNN: GraphSAGE for AF, Skip-GCN for LF, each chosen by validation PR-AUC.)

**Why embeddings hurt.** The embeddings of the *training* rows come from a GNN that was trained on those very rows and their labels, so on training rows they separate the classes almost perfectly. The tree learns to rely on them. On new data the embeddings are much less clean (see the t-SNE figure: tidy illicit cluster in the training period, illicit points scattered among licit ones in the test period), and the tree's trust is misplaced. The damage is visible already on validation: Random Forest on AF drops from validation PR-AUC 0.990 to 0.956 when embeddings are added. This is the well-known stacking mistake of fitting a second model on in-sample outputs of the first.

The paper reports a small gain from the same recipe (Random Forest AF 0.788 → 0.796 with node embeddings). We cannot say why the outcomes differ; one candidate is that their GCN fits the training data less tightly than our GraphSAGE.

**The fix and what it gives.** The out-of-fold variant gives every training row a GNN score produced by a GNN that did not train on that row's time block (three blocks), exactly like a test row. The harm disappears (validation PR-AUC 0.991) and on AF there is a small gain: Random Forest +0.013 F1 and LightGBM +0.009, with PR-AUC up by 0.005 and 0.008. The gains are about the size of one standard deviation, so they are suggestive, not conclusive. On LF it does nothing for Random Forest (+0.006) and LightGBM becomes unstable (0.686 ± 0.091; PR-AUC falls from 0.788 to 0.768).

**Answer to question 2:** GNN output adds at most a little to tree models here, and only when it is produced out-of-fold. Used the naive way it makes them clearly worse.

### 7.4 Degradation over time and the t=43 collapse

Pooled over the test steps before and after the shutdown (`results/period_metrics.csv`):

| Model (AF) | F1, steps 35–42 | Recall, steps 35–42 | F1, steps 43–49 | Recall, steps 43–49 |
|---|---|---|---|---|
| Random Forest | 0.906 ± 0.006 | 0.854 ± 0.002 | 0.030 ± 0.002 | 0.018 ± 0.000 |
| LightGBM | 0.904 ± 0.008 | 0.865 ± 0.003 | 0.031 ± 0.004 | 0.020 ± 0.003 |
| GraphSAGE | 0.729 ± 0.029 | 0.646 ± 0.054 | 0.026 ± 0.001 | 0.018 ± 0.000 |
| Random Forest, rolling retraining | 0.914 ± 0.002 | 0.850 ± 0.005 | 0.183 ± 0.064 | 0.105 ± 0.039 |
| LightGBM, rolling retraining | 0.917 ± 0.002 | 0.865 ± 0.008 | 0.469 ± 0.070 | 0.323 ± 0.068 |

- **Before the shutdown the models hold up well.** For eight time steps (about four months) after the end of training, Random Forest keeps F1 near 0.9.
- **At t = 43 everything collapses, for every model.** Random Forest goes from catching 85% of illicit transactions to catching 1.8%. No static model exceeds 0.054 F1 after the shutdown. GNNs are not more robust than trees.
- **The overall test F1 of 0.81 is therefore an average of two very different regimes**, which is why the per-time-step view matters.
- Accuracy barely notices: Random Forest's micro-F1 over the whole test period is 0.978 while it misses 98% of illicit transactions in the last seven steps.

**What helps: rolling retraining, partly.** Retraining before every test step (with labels of all earlier steps) lifts overall F1 from 0.813 to 0.840 ± 0.006 for Random Forest and from 0.806 to 0.865 ± 0.006 for LightGBM (3 seeds). After the shutdown, LightGBM recovers to F1 0.469 ± 0.070 with recall 0.323 and precision 0.872; Random Forest only to 0.183 ± 0.064. Step by step, rolling LightGBM scores 0.069, 0.168, 0.000, 0.667, 0.134, 0.576 and 0.743 on steps 43 to 49, against 0.000, 0.050, 0.000, 0.192, 0.000, 0.021 and 0.031 for the static model (`results/per_timestep.csv`).

So retraining is necessary but it is not a cure:

- It cannot help at the moment of the change: at steps 43–45 it has seen no (or almost no) examples of the new behaviour.
- It recovers only as examples accumulate (steps 48 and 49), and there are very few of them: steps 43–49 contain 24, 24, 5, 2, 22, 36 and 56 illicit transactions.
- It assumes labels arrive within two weeks, which is optimistic.
- The paper reports the same: a Random Forest retrained after every test step "is not able to reliably capture new illicit transactions after the dark market shutdown".

A plausible reason LightGBM adapts faster than Random Forest (not tested here): boosting keeps fitting the examples the current trees get wrong, so a handful of new-style illicit transactions gets attention, while a forest's bootstrap-averaged trees are dominated by the thousands of old-style examples.

**Answer to question 3:** degradation is mild until an external event changes what illicit activity looks like, and then it is total. Frequent retraining recovers part of the loss with a delay. Detecting the change quickly (monitoring recall on fresh labels, or drift in the score distribution) matters as much as the model.

### 7.5 Edge-direction ablation

GraphSAGE on AF:

| Edge treatment | Illicit F1 | PR-AUC (test) | PR-AUC (validation) |
|---|---|---|---|
| undirected | 0.642 ± 0.025 | 0.615 ± 0.023 | 0.955 ± 0.001 |
| directed (as given) | 0.550 ± 0.063 | 0.484 ± 0.051 | 0.867 ± 0.050 |
| bidirectional, separate weights | 0.558 ± 0.024 | 0.547 ± 0.035 | 0.942 ± 0.004 |

Undirected is best. Using edges only as given loses half of each node's neighbourhood (a node no longer hears from the transactions that spend its outputs) and is worse on validation and test alike. Separate weights per direction are nearly as good as undirected on validation but worse on test: the extra parameters do not buy anything that survives the time shift. For this data, knowing *that* two transactions are connected matters more than knowing which way the money went.

### 7.6 Random split versus temporal split

One clearly-labelled run each, first seed, AF (`experiment = random_split` in `metrics.csv`):

| Model | F1, temporal split | F1, random split | Inflation |
|---|---|---|---|
| Random Forest | 0.800 | 0.940 | +0.140 |
| GraphSAGE | 0.621 | 0.901 | +0.280 |

The same models look far better under a random split, and the GNN gains twice as much as the forest, because in a random split a test node's neighbours are training nodes from the same time step. A random split would have made the GNN look almost competitive (0.901 against 0.940). The temporal protocol shows it is not (0.621 against 0.800).

### 7.7 Run time

Measured wall-clock time on the development laptop (Apple M5, 10 cores, CPU only; `results/timings.json`):

- Full config: 150.8 minutes (GNNs 78.4, hybrid 35.8, baselines 13.4, ablation 11.0, rolling 8.8, random split 2.5, analysis and explainability under 1 each).
- Quick config: 11.7 minutes in the last complete quick run in this folder (its `timings.json` is kept under `archive/`), and 9.7 minutes in a verification run from a fresh clone, including the data download.

What was cut to keep the full run in that budget: the neural hyperparameter grids are small (hidden size 64 or 128, learning rate 0.01 or 0.003, dropout fixed at 0.5), the search uses one seed, and rolling retraining uses 3 seeds instead of 5.

---

## 8. Explainability

Code: `src/btc_aml/explain.py`. Figures: `results/figures/tree_feature_importance.png`, `gnn_explanation_*.png`.

### 8.1 SHAP for LightGBM

For one prediction, SHAP assigns each feature a contribution $\phi_k$ such that

$$\text{raw score}(x) = \phi_0 + \sum_{k=1}^{165} \phi_k(x)$$

where $\phi_0$ is the average raw score and the raw score is the log-odds: $P(\text{illicit}) = \text{sigmoid}(\text{raw score})$. A positive $\phi_k$ pushed this transaction towards illicit, a negative one towards licit. The values come from Shapley values in game theory: a feature's contribution is its average marginal effect over all orders in which features could be "revealed". For trees this can be computed exactly and fast (TreeSHAP).

- **Global view**: average $|\phi_k|$ over a sample of test transactions and rank the features.
- **Local view** (the app): the eight largest $|\phi_k|$ for the inspected transaction.

### 8.2 Permutation importance for Random Forest

Shuffle one feature column, re-score, and measure how much PR-AUC drops. A large drop means the model relied on that column. It is model-agnostic and simple, but it understates correlated features: if two columns carry the same information, shuffling one barely hurts because the model reads the other.

### 8.3 GNNExplainer for GNN predictions

For one node, GNNExplainer learns two soft masks with values in [0, 1]: one value per edge and one per feature. It optimises them by gradient descent so that the model, run on the masked graph, still makes the same prediction, while a penalty pushes the masks to be small and close to 0 or 1:

$$\min_{M_E,\, M_F}\ \ \underbrace{-\log P_\theta\big(\hat{y} \mid A \odot \sigma(M_E),\ X \odot \sigma(M_F)\big)}_{\text{keep the prediction}} \ +\ \underbrace{\lambda_1 \lVert \sigma(M_E) \rVert_1 + \lambda_2 H(\sigma(M_E))}_{\text{few edges, decisive values}} \ +\ (\text{same for } M_F)$$

Edges that keep a high mask value are the ones the prediction depends on. We draw the transaction's 2-hop neighbourhood (capped at 200 nodes) with edge width and darkness following the mask, node fill showing the true label and a dark ring showing whether the GNN flags that node.

Implementation notes:

- It runs on the transaction's whole time step, which is exact because time steps are disconnected (decision D21).
- Edge importance is the raw mask value between 0 and 1; edges with a value above 0.5 count as "marked important".
- Every explanation is checked by re-running the model with all edges removed and with only the marked edges kept (section 8.5).
- For a transaction with many neighbours under mean aggregation, each single edge matters little, so GNNExplainer often keeps most edges at a similar value. That is a truthful answer ("the prediction rests on the neighbourhood as a whole"), just not a sparse one.

### 8.4 What these explanations can and cannot tell you

They can show **which** inputs and **which** neighbouring transactions a model relied on. That is useful for debugging (is the model leaning on something silly?) and for pointing an analyst at the neighbours worth looking at.

They cannot say **why** in human terms, because the features are anonymised: "`local_53` pushed the score up" is as far as it goes. They are also explanations of the model, not of reality. A confident, well-explained prediction can still be wrong.

### 8.5 What the explanations show in this project

**Tree models** (`results/feature_importance.csv`, sample of 2,000 labelled test transactions). Local features dominate: 12 of LightGBM's top 15 features by mean |SHAP| are local, and all of Random Forest's top 15 by permutation importance are. The top SHAP features are `local_53`, `local_59` and `local_61`; the top aggregated one is `agg_32`. Local features account for 69.7% of LightGBM's total attribution. That fits section 7.2: the aggregated features add a modest amount on top of the local ones.

**GNN** (`results/gnn_explanations.json`, GraphSAGE on AF, first seed). Every explanation is checked by re-running the model: once with all edges removed and once with only the edges GNNExplainer marked important (mask > 0.5).

| Transaction | Time step | Score | Score, all edges removed | Edges marked important | Score with only those edges |
|---|---|---|---|---|---|
| 12673136 | 41 | 0.999 | 0.139 | 3 of 31 | 1.000 |
| 96929263 | 40 | 1.000 | 0.975 | 2 of 7 | 1.000 |
| 70328737 | 35 | 0.981 | 0.958 | 9 of 13 | 0.997 |

The first row is a case where the neighbourhood decides the prediction: without edges the score falls from 0.999 to 0.139, and three edges are enough to restore it. `gnn_explanation_1.png` shows them. In the other two the score hardly moves when all edges are removed, so the prediction comes from the transaction's own features and the edge mask has little to explain.

These three were *selected* as the flagged illicit transactions (with at least three neighbours) whose score depends most on the neighbourhood. They are not typical. Typical is the second kind: across the whole test set, removing every edge changes only 2.7% of GraphSAGE's decisions on AF (section 7.1). The app shows the "all edges removed" score for every transaction and says so when the edges do not matter.

A lesson from building this: an earlier version rescaled each explanation so that its strongest edge had importance 1.0. For a transaction whose edges did not matter at all, every edge had a tiny, similar mask value, and after rescaling they all looked maximally important. The drawing was confidently wrong. The fix was to show raw mask values and to add the two re-run checks. An explanation method should always be tested against the model it explains.

---

## 9. Limitations and ethics

### 9.1 Limitations of the data

- **Anonymised features.** We cannot say what any feature means, cannot sanity-check a model's reasoning against domain knowledge, and cannot engineer new features from the raw transaction.
- **Labels are partial and not random.** 77% of transactions are unlabelled. The labelled ones were chosen by the dataset's authors through a heuristics-based process, so they are probably easier to classify than a random transaction. Every metric here is measured on labelled transactions only; performance on the unlabelled majority is unknown.
- **"Licit" and "illicit" are labels of entities, not court verdicts.** A transaction is labelled by who is believed to control it. Label noise exists and we cannot measure it.
- **The features were standardised by the publisher over the whole dataset**, including the test period (`data.py` checks: every column has mean 0 and standard deviation 1 across all 203,769 rows). That is a small leak we cannot undo. We re-standardise with training statistics, which is the best that can be done.
- **One dataset, one time span.** The dataset covers 49 time steps of one blockchain. The conclusions may not transfer to other periods or chains.
- **The aggregated features already contain one-hop graph information**, so "table model on AF" is not a graph-free baseline.

### 9.2 Limitations of the method

- **Threshold and epoch counts transfer imperfectly** from validation (steps 30–34) to the final model (decisions D6, D7). The threshold diagnostic in section 7 shows how much this costs each model.
- **The validation period is short** (five time steps, 591 illicit transactions), so model selection is noisy, especially for neural models whose validation PR-AUC moves from epoch to epoch.
- **Hyperparameter search is small** and uses one seed (decision D8). A larger search might move individual numbers; it is unlikely to reverse a gap as large as the one between trees and GNNs.
- **Static GNNs.** Each time step is a separate graph and the models have no notion of time. Temporal models (EvolveGCN) were not implemented.
- **Rolling retraining assumes labels arrive within one time step** (about two weeks). Real investigations take much longer, so the rolling numbers are an optimistic bound.
- **Explanations are approximations.** GNNExplainer's masks depend on its random start and its regularisation strength. SHAP values are exact for the tree model but describe the model, not the world.

### 9.3 Ethics

- **False positives hurt real people.** A flagged transaction can lead to a frozen account, a blocked payment or a report to authorities. For someone living on that money, a frozen account means missed rent, and proving innocence is slow. Precision is therefore not just a metric: at precision 0.9, one in ten flagged people did nothing wrong.
- **False negatives also have victims**: ransomware payments, scams and trafficking proceeds that go through.
- **The model must not decide alone.** The right use of these scores is to **rank transactions for human review**, which is what the alert-budget metric measures. The demo app shows a probability, the neighbourhood and an explanation, and states that a flag is the start of a review, not a verdict.
- **Guilt by association.** A GNN scores a transaction partly by its neighbours. Receiving money from a flagged address does not make someone a criminal (think of a shop paid by a customer who used a mixer). The more weight a model puts on neighbours, the more this matters, and the edge explanations exist to make that weight visible.
- **Models go stale silently.** Section 7 shows a model that looked excellent for months and then stopped detecting almost anything after one external event, while its accuracy stayed high. Any deployment needs monitoring of recall on freshly labelled cases, not just of accuracy.
- **Feedback loops.** If only flagged transactions get investigated, only flagged transactions get labels, and the next model learns to find more of what the last one already found. New kinds of illicit activity stay invisible.
- **Bias we cannot audit.** With anonymised features we cannot check whether the model's signals correlate with legitimate but unusual behaviour (privacy tools, certain regions' exchanges). A real deployment would need that audit.
- **Dual use.** A detector's weaknesses, published in detail, are a guide for evading it. This project uses a public research dataset with anonymised features, which limits that risk.

---

## 10. 25 likely viva and interview questions

Short answers; each points back to the section with the detail.

### The problem and the data

**Q1. What is the task, in one sentence?**
Binary node classification on a transaction graph: for each Bitcoin transaction, predict illicit or licit from its 165 features and its connections, training on earlier time steps and testing on later ones.

**Q2. What does an edge mean, and what are LF and AF?**
An edge A → B means an output of transaction A is spent as an input of B, so money flows from A to B. LF is the 93 local features describing the transaction itself. AF is all 165: the local ones plus 72 aggregated ones, which are statistics of the transaction's one-hop neighbours computed by the dataset's authors.

**Q3. What happens to the unlabelled transactions?**
They stay in the graph and take part in message passing, because their features are useful context and they connect labelled nodes. They are masked out of the loss and of every metric. (Section 2.3)

### Evaluation

**Q4. Why a temporal split instead of a random one?**
Deployment means training on the past and scoring the future. A random split tests on the same fortnights, and the same connected components, the model trained on. We measured the effect: Random Forest scores F1 0.800 under the temporal protocol and 0.940 under a random split; GraphSAGE 0.621 and 0.901. The random split inflates both and makes the GNN look almost as good as the forest, which it is not. (Sections 4.2, 7.6)

**Q5. Why not report accuracy?**
Because 93.50% of labelled test transactions are licit, a model that flags nothing is 93.50% accurate with an illicit F1 of 0. In our own results Random Forest's accuracy over the test period is 0.978 while it misses 98% of illicit transactions after time step 43. (Sections 4.3, 7.4)

**Q6. Precision or recall: which matters more in anti-money-laundering?**
Both, and they pull against each other. Low recall lets criminal money through. Low precision floods analysts with false alarms and harms innocent customers. That is why the primary metric is F1 and why the alert-budget metric asks a practical question: if analysts can review only K cases per period, how many of them are real?

**Q7. Why PR-AUC rather than ROC-AUC?**
ROC-AUC uses the false-positive rate FP / (FP + TN). With many licit transactions TN is huge, so the rate stays tiny even when false positives outnumber true positives. PR-AUC uses precision, which compares FP with TP directly. Example from our results: GCN on LF has ROC-AUC 0.867, which sounds fine, and PR-AUC 0.466 with F1 0.337, which is not.

**Q8. How did you handle class imbalance, and why not SMOTE?**
Class weights inversely proportional to class frequency in every model's loss, plus a tuned decision threshold. SMOTE creates synthetic minority rows by interpolation; a synthetic transaction has no edges, so it has no place in the graph, and for trees class weights do the same job without inventing data. (Section 5.1)

**Q9. How was the decision threshold chosen, and why not 0.5?**
On validation scores only: the threshold with the highest illicit F1 on the precision–recall curve, then frozen. 0.5 is arbitrary once class weights shift the probabilities. The diagnostic table shows it matters: LightGBM on LF scores 0.769 at its validation threshold and 0.700 at 0.5. It also shows the limit of the method: thresholds tuned on steps 30–34 do not always transfer (GCN on AF would score 0.630 at 0.5 but gets 0.536 at its validation threshold). (Sections 5.2, 7.1)

**Q10. What is data leakage, and where could it have crept in here?**
Leakage is information from the evaluation data reaching the model or its settings. Candidates and how each is closed: a random split (temporal split); scaling with test statistics (scaler fitted on training steps, with a test); tuning threshold, epochs or hyperparameters on test (all chosen on validation); edges between splits (none exist, asserted by tests); picking "the best GNN" by test score (picked by validation PR-AUC); hybrid inputs from a GNN that saw test labels (trained on steps 1–34 only). One leak cannot be closed: the publisher standardised the features over the whole dataset.

**Q11. Why mean ± standard deviation over seeds? What does a seed change?**
The seed fixes the random choices: initial weights and dropout masks for neural models, bootstrap samples and feature subsets for trees. One run can be lucky. Here the spread differs a lot by family: Random Forest on LF varies by ±0.004 F1, Skip-GCN on LF by ±0.103. Without several seeds a single GNN run could have landed anywhere in a 0.2-wide band.

### Graph neural networks

**Q12. What is message passing?**
Each GNN layer lets every node collect its neighbours' vectors, aggregate them, combine the result with its own vector and apply a learned transformation. Stacking k layers lets information travel k hops. The weights are shared by all nodes. (Section 3.1)

**Q13. How do GCN, GraphSAGE and GAT differ?**
Only in how neighbours are weighted. GCN: fixed weights from node degrees, $1/\sqrt{d_i d_j}$, with the node itself treated as one more neighbour. GraphSAGE: plain mean of the neighbours, and a separate weight matrix for the node's own vector. GAT: a learned attention weight per edge, normalised with softmax. (Sections 3.2–3.5)

**Q14. What is Skip-GCN and why does it help?**
GCN with a direct connection from the input features into the layers: $\sigma(\hat{A} H W + X \tilde{W})$. Plain GCN averages a node's own features with its neighbours' and dilutes them; the skip term keeps them intact. In our results that is worth a lot on LF: GCN 0.337, Skip-GCN 0.585.

**Q15. Why only two layers? What is over-smoothing?**
Two layers cover the 2-hop neighbourhood. Every extra layer averages over a wider area; with many layers the vectors of all nodes in a connected component converge and become indistinguishable. Each Elliptic time step is one connected component, so a deep GNN would blur a whole time step together.

**Q16. Is the training transductive or inductive?**
Inductive. The GNN is trained on the graph of steps 1–34 and applied to the separate, unseen graph of steps 35–49. That works because the layer weights do not depend on which nodes exist. The random-split illustration is the transductive case: one graph, with test nodes present (unlabelled) during training.

### Results

**Q17. Did the GNNs beat Random Forest?**
No. Random Forest on AF: illicit F1 0.813 ± 0.011. LightGBM: 0.806 ± 0.011. Best GNN: Skip-GCN 0.652 ± 0.019. Random Forest on local features alone (0.789) beats every GNN on all features. The original paper found the same ordering. (Section 7.1)

**Q18. Why do the GNNs lose?**
Mainly because neural networks lose to trees on these tabular features: the MLP, which has no graph, scores 0.630 on AF, about the same as the GNNs. The graph adds little on top: on the test period, removing all edges from a trained GNN changes under 5% of its decisions and in half the cases raises its PR-AUC. The graph is sparse (median degree 2), labels are few, and what the GNNs learn about neighbourhoods does not carry over to later time steps. (Section 7.1)

**Q19. The aggregated features are neighbourhood summaries. Does a GNN on LF learn them by itself?**
Not usefully. AF is worth +0.024 F1 to Random Forest and +0.036 to LightGBM. GNNs on LF reach at most 0.629, below the MLP on LF (0.655) and far below Random Forest on LF (0.789). On validation the GNNs do profit from edges, so they learn something about neighbourhoods, but it stops helping on the test period. (Section 7.2)

**Q20. Do GNN embeddings improve tree models?**
Used the classic way, they make them worse: Random Forest on AF falls from 0.813 to 0.652. The embeddings of training rows come from a GNN trained on those rows, so they are unrealistically clean there and the tree over-trusts them. When each training row instead gets a GNN score from a GNN that did not train on its time block (out-of-fold), the harm disappears and there is a small gain on AF: 0.827 ± 0.005 for Random Forest, 0.815 ± 0.002 for LightGBM. (Section 7.3)

**Q21. What happened at time step 43, and what is this kind of failure called?**
A dark market was shut down. The illicit activity the models had learned disappeared, and what remained looked different. Random Forest's recall falls from 0.854 on steps 35–42 to 0.018 on steps 43–49; no static model exceeds F1 0.054 afterwards. This is distribution shift, more precisely concept drift: the relationship between features and label changed. (Sections 5.3, 7.4)

**Q22. Does retraining fix it?**
Partly, and with a delay. Retraining before every test step lifts LightGBM's F1 on steps 43–49 from 0.031 to 0.469 and Random Forest's from 0.030 to 0.183. It cannot help at the moment of the change, because no examples of the new behaviour exist yet, and it assumes labels arrive within two weeks. A deployment would also need monitoring that notices the change: recall on freshly labelled cases, drift in the score distribution. (Section 7.4)

**Q23. Does edge direction matter?**
Treating edges as undirected was best for GraphSAGE on AF (F1 0.642). Keeping only the given direction (0.550) throws away half of each node's neighbourhood. Separate weights per direction (0.558) matched undirected on validation but not on test. (Section 7.5)

**Q24. How do you explain a prediction, and how do you know the explanation is right?**
For trees, SHAP splits a prediction into per-feature contributions that add up exactly to the score. For the GNN, GNNExplainer learns a mask over edges that preserves the prediction with as few edges as possible. We check it by re-running the model: for transaction 12673136 the score is 0.999, falls to 0.139 with all edges removed, and returns to 1.000 with only the 3 marked edges, so that explanation is faithful. For most transactions the score does not move when edges are removed, and then the honest explanation is "the model used the transaction's own features". The features are anonymised, so none of this says what a feature means. (Section 8)

**Q25. Would you deploy this? What would you do next?**
Not as an automatic decision-maker. As a ranking tool for analysts it has value while the world is stable: before the shutdown, 91% of Random Forest's top-50 alerts per time step are truly illicit (precision@50 0.911). It needs frequent retraining, drift monitoring, human review of every flag and an appeal path, because a false positive can freeze an innocent person's funds. Next steps: a temporal model such as EvolveGCN, drift detection to trigger retraining, pseudo-labelling of the unknown nodes, calibrated probabilities with cost-sensitive thresholds, and wallet-level detection on Elliptic++. (Section 9)
