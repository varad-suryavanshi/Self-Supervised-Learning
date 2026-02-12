# Self-Supervised Learning: From Scratch to Foundation-Model Distillation

This repository contains my **Deep Learning course project** exploring **self-supervised representation learning** across two major paradigms:

1. **Training SSL models from scratch** (SimCLR, BYOL, SwAV)
2. **Knowledge distillation from a pretrained foundation model (DINOv2)** using cosine and VICReg-style objectives, including **multicrop experiments**

The project is structured to progressively study **stability, representation quality, and compute tradeoffs** across modern SSL methods.

---

## 🔍 Motivation

Self-supervised learning enables representation learning from large-scale **unlabeled data**, reducing reliance on costly annotations.
This project aims to:

* Understand how **different SSL objectives** shape learned representations
* Compare **contrastive, non-contrastive, and clustering-based** methods
* Study how **foundation models (DINOv2)** can be distilled into smaller students
* Analyze the role of **regularization (VICReg)** and **multicrop augmentation**
* Balance **performance vs compute cost** in practical training setups

---

## 📁 Repository Structure

```
.
├── simclr/                # Contrastive learning from scratch
├── byol/                  # Bootstrap Your Own Latent (non-contrastive)
├── swav/                  # SwAV / DeepCluster-v2 (clustering + multicrop)
├── dinov2/        # DINOv2 teacher–student distillation
│   └── multicrop/         # Multicrop KD variants
├── .gitignore
└── README.md
```

Each subdirectory is **self-contained** and includes training scripts and (where applicable) cluster/Slurm launch files.

---

## 🧠 Methods Implemented

### 1️⃣ Training SSL Models from Scratch

#### **SimCLR**

* Contrastive learning with strong augmentations
* InfoNCE-style objective
* Learned representations via instance discrimination
* Implemented with ResNet backbone and custom augmentation pipeline

📂 `simclr/`

---

#### **BYOL**

* Non-contrastive, bootstrap-based SSL
* Online–target network with EMA updates
* Avoids negative samples
* Demonstrates stable representation learning without contrastive pairs

📂 `byol/`

---

#### **SwAV / DeepCluster-v2**

* Clustering-based SSL with **swapped assignments**
* Multi-crop training (global + local views)
* No explicit pairwise contrastive loss
* Includes evaluation scripts (linear / semi-supervised)

📂 `swav/`


---

### 2️⃣ Foundation Model Distillation (DINOv2)

Rather than training from scratch, this phase distills knowledge from a **pretrained DINOv2 teacher** into a student network (96×96 resolution).

#### **Baseline Distillation**

* Student trained to match teacher embeddings
* Cosine / MSE-style invariance objective

#### **Distillation + VICReg**

* Adds **variance** and **covariance** regularization
* Prevents representation collapse
* Encourages informative, decorrelated features

#### **Multicrop Distillation**

* Student receives multiple views per image
* Teacher operates only on global crops
* Stronger invariance signal, higher compute cost

📂 `distill_dinov2/`

---

## 📊 Evaluation Strategy (Described)

To assess representation quality, I used **feature-based evaluation**:

1. Freeze the encoder
2. Extract embeddings on labeled downstream data
3. Evaluate using:

   * **k-NN classification** in embedding space
   * **Linear probing** (train linear classifier on frozen features)

This isolates the **quality of learned representations** from downstream model capacity.


---

## ⚖️ Key Observations & Tradeoffs

* **VICReg-style regularization** significantly improves stability in distillation
* **Multicrop training** improves invariance but increases training cost substantially
* **Distillation from DINOv2** yields strong representations with fewer epochs than training from scratch
* Different SSL paradigms impose **very different inductive biases** on learned features

These tradeoffs are explicitly documented in the code and READMEs.

---

## 🚀 How to Run

Each method can be run independently:

* **SimCLR**: `simclr/run.py` or `run_simclr.sbatch`
* **BYOL**: `byol/train_byol.py`
* **SwAV**: `swav/main_swav.py` or `run_swav.sbatch`
* **DINOv2 KD**: see `distill_dinov2/README.md`

Cluster / Slurm scripts are included where applicable.

---

## 🧾 Attribution

* SwAV / DeepCluster-v2 components follow the original implementation style and license.
* Please cite the original papers if using this code in academic work:

  * SimCLR
  * BYOL
  * SwAV
  * DINOv2
  * VICReg

---

## ✨ Summary

This project provides a **hands-on, end-to-end exploration of modern self-supervised learning**, from classic contrastive methods to **foundation-model distillation**.
It emphasizes **practical implementation details**, **training stability**, and **compute-aware decision making**, reflecting real-world ML research and engineering constraints.

---


