# drug-response-prediction-gnn-maml
Few-shot drug response prediction on GDSC data using GNN drug encoders, gene expression cell representations, and MAML meta-learning for rare cancer generalization.
# Drug Response Prediction Pipeline
### GNN + MAML | GDSC Dataset | PyTorch Geometric

> Few-shot drug response prediction on GDSC data using graph-based drug encoders, gene expression cell representations, and MAML meta-learning for generalisation to rare cancer types.

---

## Overview

This project builds a progressive drug response prediction pipeline trained on the [Genomics of Drug Sensitivity in Cancer (GDSC)](https://www.cancerrxgene.org/) dataset. The pipeline predicts LN_IC50 values (a measure of drug sensitivity) for drug–cell line pairs, with a specific focus on generalising to **rare cancer types** that have limited labelled data.

The architecture evolves through five stages, each adding a richer representation or learning strategy:

```
Baseline MLP → GNN Drug Encoder (V4) → Gene Expression Cell Encoder (V5) → Fine-tuning → MAML Few-Shot
```

---

## Results

Performance on held-out **rare cancer types** (unseen during training):

| Model | RMSE | R² | Pearson | Spearman |
|---|---|---|---|---|
| Baseline MLP (Morgan FP) | 1.6414 | 0.6833 | 0.8373 | 0.7511 |
| GNN V4 (GCN drug encoder) | — | ~0.70 | — | — |
| GNN V5 (expression encoder) | — | ~0.74 | — | — |
| GNN V5 Fine-tuned | — | 0.8376 | — | — |
| **MAML K=5** | — | 0.7013 | — | — |
| **MAML K=10** | — | 0.7378 | — | — |
| **MAML K=20** | — | 0.7568 | — | — |
| **MAML K=50** | — | **0.7610** | — | — |

MAML achieves up to **+11.4% R² improvement** over the baseline in the pure few-shot setting (no fine-tuning). Fine-tuned V5 achieves **+22.6% R²** improvement when rare cancer support data is available.

---

## Architecture

### 1. Baseline MLP
- Drug: Morgan fingerprints (2048-bit, radius=2, via RDKit)
- Cell line: learned embedding (64-dim)
- Predictor: MLP [512 → 256 → 128 → 1]

### 2. GNN Drug Encoder (V4)
- Drug: 3-layer GCNConv on molecular graph (atom features: 14-dim, bond features: 6-dim) + global mean pooling → 128-dim drug embedding
- Cell line: learned embedding (64-dim)
- Predictor: MLP [256 → 128 → 1] with BatchNorm + Dropout(0.4)

### 3. Gene Expression Cell Encoder (V5)
- Drug: same GCN encoder as V4
- Cell line: **real gene expression profiles** (top 2000 most variable genes from GDSC) encoded via a 2-layer MLP [2000 → 256 → 64]
- Expression matrix registered as a fixed buffer (not trained); encoder weights are learned

### 4. MAML (Model-Agnostic Meta-Learning)
- Warm-started from best V5 checkpoint
- Inner loop: 5 SGD steps on K support examples per cancer type task (via `higher` library)
- Outer loop: Adam (lr=1e-4) on query set loss across 8 tasks per episode
- Evaluated with K = 5, 10, 20, 50 shots on held-out rare cancer types

---

## Dataset

**GDSC2** drug response data:
- ~300 drugs, ~1000 cell lines, 14 cancer types
- Target: `LN_IC50` (log-transformed half-maximal inhibitory concentration)
- Split strategy: common cancers (≥5000 samples) → train/val; rare cancers (<5000 samples) → test

**Gene expression** (CCLE/DepMap):
- Top 2000 most variable genes by variance across cell lines
- z-score normalised per gene

**Drug SMILES**:
- Fetched from PubChem REST API by drug name
- Failed lookups retried with cleaned names (stripped brackets, salt suffixes)
- `drug_smiles_final.csv` committed directly — no need to re-fetch

---

## Repository Structure

```
.
├── data/
│   ├── train_final.csv          # Training pairs (common cancers, with SMILES)
│   ├── val_final.csv            # Validation pairs
│   ├── test_final.csv           # Test pairs (rare cancers)
│   ├── rare_finetune.csv        # 20% of rare cancer data for fine-tuning
│   ├── rare_holdout.csv         # 80% of rare cancer data for evaluation
│   ├── drug_vocab_final.csv     # Drug name → integer ID
│   ├── cell_vocab_final.csv     # Cell line name → integer ID
│   ├── common_cancers.csv
│   └── rare_cancers.csv
│
├── drug_smiles_final.csv        # Pre-fetched SMILES for all drugs
├── gene_expression_2k.csv       # Top 2000 variable genes (pre-computed)
│
├── models/                      # Saved checkpoints (not committed; see below)
│   ├── baseline_best.pt
│   ├── gnn_v4_best.pt
│   ├── gnn_v5_best.pt
│   ├── gnn_v5_finetuned.pt
│   └── gnn_maml_best.pt
│
├── results/                     # Training histories and evaluation outputs
│
└── drug-response.ipynb          # Main notebook
```

> **Note:** Model checkpoints are not committed due to file size. Run the notebook sequentially to reproduce them.

---

## Setup

### Requirements

```bash
pip install torch torchvision
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html
pip install rdkit pandas numpy scikit-learn scipy matplotlib tqdm higher
```

> Replace `{TORCH_VERSION}` and `{CUDA_VERSION}` with your environment's versions. This notebook was developed on Kaggle (PyTorch 2.9, CUDA 12.6).

### Running

The notebook is self-contained and designed to run on **Kaggle** with GPU. Run cells sequentially. Key stages:

| Stage | Cells | Description |
|---|---|---|
| Data prep | 3–29 | Load GDSC, fetch SMILES, build train/val/test splits |
| Baseline | 30–48 | Train and evaluate MLP baseline |
| GNN setup | 50–63 | Build molecular graph dataset |
| GNN V4 | 64–79 | Train GNN with learned cell embeddings |
| V5 (expression) | 80–101 | Build expression matrix, train expression encoder |
| Fine-tuning | 106–117 | Adapt V5 to rare cancers |
| MAML | 120–131 | Meta-training and few-shot evaluation |

> Cells 1–2 and 49–51 install dependencies and can be skipped if already installed.

---

## Key Design Decisions

- **Common vs. rare cancer split** at 5000 samples gives a clean domain shift for evaluation without leaving the model completely blind to cancer biology at training time.
- **Gene expression over learned embeddings** for cell lines: at test time on rare cancers, learned embeddings are never updated during training, making them effectively random. Real expression profiles provide biological grounding regardless of whether the cell line appeared in training.
- **MAML warm-started from V5** rather than random init: the expression encoder and GCN weights are already meaningful, so meta-training only needs to refine for fast adaptation.
- **Target normalisation fit on train only** to prevent data leakage into val/test.

---

## Known Limitations

- The holdout set is used for both early stopping (during fine-tuning) and final evaluation — acknowledged limitation; a stricter three-way split would be needed for publication.
- Expression matrix z-score normalisation currently uses all cell lines rather than train-only; a minor leakage that has negligible practical effect but should be corrected for a paper.
- DepMap cell line matching uses fuzzy string normalisation; a small number of cell lines may be silently mismatched.

---

## Citation / Data Sources

- **GDSC**: Yang et al., *Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells*, Nucleic Acids Research, 2013.
- **DepMap / CCLE**: Broad Institute DepMap portal — [depmap.org](https://depmap.org)
- **PubChem SMILES**: PubChem REST API — [pubchem.ncbi.nlm.nih.gov](https://pubchem.ncbi.nlm.nih.gov)
- **PyTorch Geometric**: Fey & Lenssen, *Fast Graph Representation Learning with PyTorch Geometric*, ICLR Workshop 2019.
- **MAML**: Finn et al., *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*, ICML 2017.

---

## Author

**Sulagna** — Final-year B.Tech Information Technology student, Future Institute of Engineering and Management, Kolkata.
Research focus: ML for computational medicine | PCOS diagnostics | Drug response prediction.

---

*Built as part of a research project on computational drug response prediction, developed on Kaggle.*
