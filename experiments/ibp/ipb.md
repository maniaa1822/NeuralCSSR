# NeuralCSSR Inductive Bias & FER Research Plan

## Overview

This document outlines a systematic plan to study **fragmented entangled representations (FER)**, **inductive bias**, and **transfer learning** using the NeuralCSSR AR/EBM codebase and the `ebm_ibp.py` probe script. The aim is to measure how well learned hidden representations align with true causal states and to study their generalization across machines and tasks.

---

## 1. Goals

1. **Quantify fragmentation and entanglement** of hidden representations using R-IB / D-IB metrics.
2. **Probe inductive bias** of the model: does it learn meaningful causal states or just superficial correlations?
3. **Evaluate OOD generalization** by transferring representations between different synthetic machines.
4. **Visualize internal states** to obtain human-interpretable insights similar to FER on images.

---

## 2. Experimental Setup

### 2.1 Datasets

* **Golden Mean Process (GM)**: standard causal state sequence.
* **Even Process (EP)**: parity-based causal states.
* **Custom synthetic machines**: create sequences with different state structures for OOD tests.

### 2.2 Models

* **EnergyBasedBinaryLM (EBM)**
* **AutoRegressiveBinaryLM (AR)**
* **Checkpoints**: pretrained on GM or EP, with context window adapted to sequence length.

### 2.3 Tasks

* **Random surjective mapping** of states → labels (to compute R-IB / D-IB).
* **Custom tasks**:

  * `even_state->1` / `even_state->0`
  * `length_even`
  * `last3_sum_odd`
  * Extended tasks: `last5_sum_odd`, `parity_mod3`
* **Probing approaches**:

  * Fine-tune full model vs head only
  * Train separate linear probe on last hidden layer

### 2.4 Metrics

* **R-IB**: fraction of same-state pairs mapped to same predictions.
* **D-IB**: fraction of different-state pairs mapped to different predictions.
* **Next-token metrics**: CE, accuracy pre/post fine-tuning
* **Visualization**: t-SNE / UMAP of hidden representations colored by causal states

---

## 3. Experiment Pipeline

### 3.1 Baseline Representation Analysis

1. Load pretrained model (GM or EP).
2. Compute hidden states on validation sequences.
3. Train probe heads for causal states.
4. Compute R-IB / D-IB and log next-token metrics.
5. Visualize hidden state clustering with t-SNE / UMAP.

### 3.2 Transfer Learning Across Machines

1. Train model on source machine (GM).
2. Fine-tune probe or LM head on target machine (EP or custom).
3. Measure:

   * Change in R-IB / D-IB
   * Next-token metrics on target machine
   * Representation alignment via t-SNE / UMAP

### 3.3 Complexity Scaling

* Generate tasks of increasing difficulty (longer token history, modulo sums, composite parity).
* Evaluate probe performance and representation quality as task complexity increases.

### 3.4 Layer-wise Representation Study

* Probe multiple hidden layers, not just last layer.
* Observe at which layer R-IB / D-IB is highest.
* Identify where FER emerges or fades.

### 3.5 Statistical Evaluation

* Repeat each experiment with multiple seeds.
* Compute mean and standard deviation of R-IB / D-IB.
* Perform paired comparisons between pre/post fine-tuning or across machines.

---

## 4. Deliverables

1. **Quantitative metrics**: R-IB, D-IB, next-token CE/accuracy, per task/machine/layer.
2. **Visualizations**: t-SNE / UMAP plots, probe activation heatmaps.
3. **Transfer learning analysis**: tables showing representation portability.
4. **Report / paper**: description of methodology, results, and interpretation in terms of FER and inductive bias.

---

## 5. Optional Extensions

* Compare AR vs EBM performance on OOD tasks.
* Implement synthetic interventions on sequences to test causal robustness.
* Correlate representation alignment (R-IB / D-IB) with predictive accuracy.
* Explore few-shot probing: how many examples are needed to recover FER?

---

## 6. Implementation Notes

* Use `ebm_ibp.py` as the core probe script.
* Ensure consistent train/validation splits across tasks.
* Use GPU when available for faster fine-tuning and probing.
* Log all hyperparameters and seeds for reproducibility.
* Save hidden states for offline visualization and analysis.

---

## References

* Fragmented Entangled Representations (FER) literature.
* NeuralCSSR codebase: EBM & AR models.
* Inductive Bias Probe methodology (Vafa et al., 2025).
