# Amortized Epsilon-Machines from Transformers

This note captures the idea of using CSSR as a self-supervised teacher to learn models that map sequences (or model representations) directly to epsilon-machines / belief-state dynamics. The goal is to refactor “run CSSR on a neural oracle” into a learned, amortized mapping that outputs an interpretable epsilon-machine in one shot.

## 1. Two training regimes

We distinguish two regimes for how the transformer oracle is trained.

### 1.1 One model per one machine

- Data come from a single ground-truth epsilon-machine, call it \(E\).
- A small transformer \(f\) is trained as a next-token oracle on sequences from \(E\).
- Running two-stage oracle CSSR on \(f\) (our `two_stage_oracle_cssr.py`) recovers a discrete state space that, in the ideal limit, matches the causal states of \(E\) up to isomorphism.

In this regime:
- Each history \(x_{<t}\) has a ground-truth causal state \(S_E(x_{<t})\).
- The transformer hidden state is \(h_t = h(x_{<t})\), a deterministic function of the same history.
- There exists some mapping \(\phi\) with

  $$
  S_E(x_{<t}) \approx \phi(h_t)
  $$

  up to relabeling of states.

CSSR gives us ground-truth-like discrete targets for supervised learning:
- We can collect pairs \((h_t, s_t)\) where \(s_t\) is the CSSR state label for history \(x_{<t}\).
- A probe \(g(h_t) \to s_t\) can learn how the transformer encodes causal states.

### 1.2 One universal model over many machines

- Data come from a mixture of machines \(M \in \{1, \dots, K\}\).
- A single transformer \(f\) is trained on sequences drawn from this mixture.

For prediction, the sufficient statistic is the belief over machine and machine-state:

$$
b_t(m, s) = P(M = m, S_t^m = s \mid x_{<t}).
$$

In this regime:
- The optimal internal representation is a belief state \(b_t\) rather than a single machine’s state.
- Geometrically, hidden states lie on a continuous manifold approximating the simplex of beliefs over \((m, s)\).
- CSSR run on the universal model no longer recovers the states of any single machine; instead, it recovers equivalence classes of histories that induce the same predictive distribution under the mixture. These are “belief-state clusters” of the universal model.

We can still supervise a mapping \(g(h_t) \to s_t\), but now the discrete labels \(s_t\) correspond to approximate belief states, not to a single machine’s causal states.

## 2. What we ultimately want

Ultimate goal: a model that receives data from an unknown process and outputs an epsilon-machine (or belief-state machine) describing that process, without explicitly running CSSR at test time.

Two flavors:

1. **Sequence → epsilon-machine (amortized CSSR)**  
   - Input: a set of sequences from a process.  
   - Output: a finite-state predictive model (epsilon-machine) for that process.

2. **Model representation → epsilon-machine (meta over oracles)**  
   - Input: a representation of a trained transformer oracle (weights and/or behavior).  
   - Output: the epsilon-machine discovered by CSSR when run on that oracle.

CSSR acts as a non-differentiable teacher that provides discrete machine targets; we then amortize that mapping with a differentiable model.

## 3. Representing epsilon-machines as fixed tensors

To train with gradient descent we need a fixed-format representation for the machine.

Assumptions for now:
- Binary alphabet \(A = \{0, 1\}\).
- At most \(S_{\max}\) states.

One convenient parametrization:

- State indices: \(s \in \{1, \dots, S_{\max}\}\).
- Initial state distribution \(\pi \in [0,1]^{S_{\max}}\) with

  $$
  \sum_s \pi_s = 1.
  $$

- Transition–emission tensor \(T \in [0,1]^{S_{\max} \times |A| \times S_{\max}}\) where

  $$
  T[s, x, s'] = P(S_{t+1} = s', X_{t+1} = x \mid S_t = s),
  $$

  with normalization

  $$
  \sum_{x, s'} T[s, x, s'] = 1 \quad \text{for each } s.
  $$

We also need a canonical ordering of states to break permutation symmetry. For CSSR-derived machines we can:
- Choose the start state.
- Do a BFS/DFS over the reconstructed graph, ordering states by first visit under a lexicographic order over emitted symbols.
- Reindex states according to this traversal and export \((\pi, T)\) accordingly.

This gives us a fixed-size target \((\pi, T)\) per process, suitable for supervised learning.

## 4. Regime A: universal model, sequence-level supervision

Here we focus on a single universal transformer trained on a fixed mixture of machines. We want a model \(F\) that maps sequences from that mixture directly to an epsilon-machine describing the mixture’s dynamics.

### 4.1 Teacher pipeline

For a chosen mixture of ground-truth machines:
- Train a universal transformer oracle \(f\) on sequences from the mixture.
- Run two-stage oracle CSSR on \(f\) to obtain an approximate epsilon-machine \(\hat{M}\):
  - Discovered states and minimal suffixes.
  - Transition structure (we can estimate \((\pi, T)\) by rolling out the model and mapping contexts to CSSR states).

This \(\hat{M}\) serves as a discrete teacher machine for that mixture.

### 4.2 Student model \(F: \{\text{sequences}\} \to (\pi, T)\)

Inputs:
- A set of \(K\) sequences \(x^{(i)}\) from the mixture, each of length \(L\).

Architecture sketch:
- Encode each sequence with a shared encoder (e.g., transformer or bidirectional RNN) to get per-sequence embeddings \(z^{(i)}\).
- Pool the set \(\{z^{(i)}\}\) into a single process-level representation \(z\) via:
  - mean pooling, or
  - a DeepSets-style set encoder, or
  - cross-attention from a learned “process token” into the sequence embeddings.
- Pass \(z\) through an MLP / hypernetwork to produce:
  - logits for \(\pi\), normalized with softmax,
  - logits for each row of \(T\), normalized with softmax across \((x, s')\) for each \(s\).

Targets:
- The teacher epsilon-machine \((\pi_{\text{CSSR}}, T_{\text{CSSR}})\) extracted as above.

Loss:
- Cross-entropy between predicted and teacher \(\pi\).
- Cross-entropy between predicted and teacher transition distributions \(T[s, :, :]\) for all active states.
- Optional functional loss: simulate from the student machine on training sequences and match its predictive distributions to the teacher transformer’s predictions.

At test time:
- Given sequences from the same mixture family, \(F\) outputs an epsilon-machine without running CSSR or a large transformer.

## 5. Regime B: many model–machine pairs (meta-learning over oracles)

Here each trained transformer on a single machine is a **data point**, and we learn a meta-mapping from model representations to epsilon-machines.

### 5.1 Data generation

For each ground-truth machine \(E_i\):
- Generate training sequences from \(E_i\).
- Train a small transformer oracle \(f_i\) on \(E_i\).
- Run two-stage oracle CSSR on \(f_i\) to obtain an epsilon-machine \(\hat{M}_i = (\pi_i, T_i)\).

We now have a dataset of pairs \((f_i, \hat{M}_i)\).

### 5.2 Choosing a model representation

Two broad options:

1. **Weight-based representation**
   - Flatten selected layers’ weights, or
   - Extract a lower-dimensional summary via a small “model encoder” network (e.g., per-layer projections plus pooling).

2. **Behavior-based representation**
   - Fix a probe set of histories \(H = \{h_j\}\).
   - For each oracle \(f_i\), collect:
     - hidden states \(h_i(h_j)\) at those histories, and/or
     - predictive distributions \(p_i(\cdot \mid h_j)\).
   - Pool across \(H\) to get a compact embedding \(z_i\).

In both cases we obtain a fixed-size vector \(x_i\) summarizing model \(f_i\).

### 5.3 Meta-model \(G: x_i \to (\pi_i, T_i)\)

Train a parametric map \(G\) (e.g., MLP) from model embeddings \(x_i\) to the corresponding epsilon-machine parameters \((\pi_i, T_i)\).

Loss:
- Same as before: cross-entropy on \(\pi_i\) and rows of \(T_i\).

Generalization:
- \(G\) learns structure in the space of processes: how changes in underlying dynamics reflect in the learned transformer and its CSSR machine.
- A well-trained \(G\) can potentially generalize to new machines drawn from the same family (new oracles, new \(\hat{M}\)).

## 6. Optimal belief-state machines for GT mixtures

For mixtures of known ground-truth epsilon-machines we can, in principle, construct the **minimal belief-state epsilon-machine** analytically, without CSSR.

Setup:
- Ground-truth machines \(E_1, \dots, E_K\) with known transitions.
- Mixture prior \(P(M = m)\).
- Hidden state at time \(t\) is the pair \((M, S_t^M)\).

Belief state:

$$
b_t(m, s) = P(M = m, S_t^m = s \mid x_{<t}).
$$

Construction sketch:

1. Build the joint hidden-state model over \((M, S_t^M)\) with known emission probabilities.
2. Start from the prior belief \(b_0\) over \((m, s)\).
3. For each observable symbol \(x\), update belief via Bayes to obtain successors \(b'\).
4. Explore reachable beliefs by iterating this update.
5. Define an equivalence relation: \(b_1 \sim b_2\) if they induce identical predictive distributions over all finite futures.
6. Merge equivalent beliefs and construct the resulting minimal deterministic machine over belief-states.

This belief-state epsilon-machine is the optimal predictive model for the mixture and is the conceptual target we would like our amortized models to approximate.

In practice:
- For finite mixtures of finite machines the belief-state machine is finite but can be large.
- CSSR on a universal transformer is an approximate, sample-based route to a similar object.

## 7. Open design questions

Some choices we still need to nail down:

- How large should \(S_{\max}\) be, and how do we handle variable numbers of active states (masking vs explicit “null” state)?
- Which representation of transformer models is most informative for \(G\) (weights vs behavior vs hybrid)?
- For the sequence-level amortized model \(F\), how many sequences and what lengths are needed per process to reliably reconstruct the machine?
- How do we best combine structural losses (matching \((\pi, T)\)) with functional losses (matching predictive distributions)?
- For empirical work, do we start with the single-machine regime (cleaner mapping from hidden states to causal states) before moving to mixtures?

This document is intended as a living design sketch we can iterate on as we start implementing probes and meta-models in this repo.
