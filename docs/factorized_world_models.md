# Factorizing World Models with Neural CSSR

This note integrates a set of control/Minigrid experiments with the broader Neural CSSR story. The core idea is:

- Train a single autoregressive transformer on joint streams of observations and actions.
- Treat it as an oracle over **controlled sequences**.
- Use Neural CSSR–style equivalence relations to factor the joint predictive structure into:
  - an **environment (physics) machine**, and
  - an **agent (policy/intent) machine**.

This extends the “transformers internalize causal/belief states” story to **controlled processes**, where the joint epsilon-machine mixes physics and agency.

---

## 1. Joint machines and the Two Views hypothesis

### 1.1 Joint epsilon-machine in an agent–environment loop

In a controlled setting we observe a stream
\[
X_1, X_2, X_3, \dots \in \{O, A\}
\]
that alternates between observations and actions:

```text
obs_t, act_t, obs_{t+1}, act_{t+1}, ...
```

Conceptually:

- The **environment** has a hidden state \(S_{\text{env}, t}\) and transition kernel
  \[
  P(O_{t+1}, S_{\text{env}, t+1} \mid S_{\text{env}, t}, A_t).
  \]
- The **agent** has internal state \(S_{\text{agent}, t}\) and policy
  \[
  P(A_t \mid S_{\text{env}, t}, S_{\text{agent}, t}).
  \]

If we ignore the internal decomposition, the combined system is just a **single stochastic process** over the joint stream of tokens. Its causal states (in the CSSR sense) form a **joint epsilon-machine**:

- states summarize sufficient information for predicting the entire future stream \((O, A)\),
- environment and agent dynamics are entangled in those states.

### 1.2 Two equivalence relations on histories

The Two Views hypothesis is that we can define two distinct but related causal partitions over histories \(h = X_{1:t}\), by changing the **equivalence relation** we use in Stage 1 of Neural CSSR:

1. **Environment view (\(\epsilon_{\text{env}}\))**  
   Histories are equivalent if they have the same **affordances**: for each possible forced action \(a\), they induce the same conditional distribution over next observations. Formally (using the transformer as an oracle):

   ```text
   h1 ~_env h2  IFF  For all actions a:
                     P(O_next | h1, do(A=a)) ≈ P(O_next | h2, do(A=a))
   ```

   - We implement this by querying the model under counterfactual actions and comparing
     \(P(O_{t+1} \mid h, A_t = a)\) across histories using JS divergence.
   - This is an **interventional** view: “if I force action a, what happens?”

2. **Agent view (\(\epsilon_{\text{agent}}\))**  
   Histories are equivalent if they induce the same **policy**:

   ```text
   h1 ~_agent h2  IFF  P(A_next | h1) ≈ P(A_next | h2)
   ```

   - Here we compare the model’s action distributions given histories.
   - This is an **intentional** view: “what does the agent want to do next?”

Both are Neural CSSR–style equivalence relations: instead of clustering histories by their full future distributions, we cluster by **specific conditional distributions** exposed by the model-as-oracle.

### 1.3 Why one joint model instead of separate world/policy nets?

We deliberately train a **single** autoregressive transformer over the flat stream `(obs, act, obs, act, ...)`:

- This matches the “GPT recipe” and is easy to scale.
- The joint model learns the full **closed-loop dynamics** of agent + environment:
  \[
  P(O_{t+1}, A_{t+1} \mid O_{1:t}, A_{1:t}).
  \]
- Factorization (into environment and agent machines) is treated as a **post-hoc analysis** step rather than an architectural constraint.

This is conceptually appealing:

- We can leverage strong sequence models and then apply Neural CSSR to recover interpretable structure (map, agency, intent) from the joint machine.

### 1.4 Tokenization and role knowledge

In all of these experiments we assume we know which tokens correspond to **observations** and which correspond to **actions**. This is a mild and natural assumption in RL/control settings:

- The log format already distinguishes controllable actions from passive observations.
- The Env and Agent views explicitly need to know “which symbols can I intervene on?” (actions) vs “which symbols are consequences?” (observations).

Concretely:

- We use flat sequences like `obs_4 act_up obs_1 act_left ...` or typed tokens like `OBS_4`, `ACT_UP`.
- Env view:
  - Condition on histories that end in an observation.
  - Intervene on the next **action token** by forcing different `ACT_*` choices and comparing `P(OBS_* | history, do(ACT_*))`.
- Agent view:
  - Condition on histories that end in an observation.
  - Compare `P(ACT_* | history)` over the **action tokens**.

This small amount of role knowledge is enough to define all three machines:

- joint machine (no special typing needed),
- world-only machine (Env view over action tokens),
- agent-only machine (Agent view over action tokens, optionally conditioned on env state).

#### Note on more unsupervised variants

If we wanted to reduce supervision further, we could try to infer which tokens are “actions” vs “observations” from the trace itself, using simple heuristics:

- **Alternation pattern:** in many control logs, symbols alternate in a consistent pattern (e.g., “type A then type B”), so we can cluster positions into two roles.
- **Intervention sensitivity:** tokens whose counterfactual choices strongly change future observation distributions are likely **actions**; tokens whose distributions depend on those choices are likely **observations**.
- **Causal directionality tests:** estimate asymmetric influence statistics (e.g., `I(token_t → token_{t+1:t+k})` vs `I(token_{t+1} → token_t)`) and separate tokens that primarily exert influence (actions) from those that primarily respond (observations).

These ideas would let us push toward a more unsupervised setting, but for the core factorization experiments we assume the action/observation typing is given, which is standard in RL. 

---

## 2. Experiment Q2: Factorization of control (agency vs noise)

**Question:** Can the model distinguish regions where actions matter from regions where they do not?

### 2.1 Setup

Environment:

- 4 physical states divided into two regions: `A1`, `A2` (Controlled), `B1`, `B2` (Random).
- Region A (Controlled):
  - Actions: `flip` swaps `A1` ↔ `A2`, `hold` leaves state unchanged.
  - High agency.
- Region B (Random):
  - Next state is uniformly random in `{B1, B2}`, regardless of action.
  - No agency.
- 5% chance per step to switch between regions (A ↔ B).

Data:

- Tokens: `obs_A1`, `obs_A2`, `obs_B1`, `obs_B2`, `act_flip`, `act_hold`.
- Sequence: interleaved observation/action, e.g.:

  ```text
  obs_A1 act_flip obs_A2 act_hold obs_A2 ...
  ```

- A small nanoGPT is trained on 5k episodes of this process.

### 2.2 Probing the oracle for control

We probe the trained transformer using an **agency metric**:

- For each history \(h\), compute the JS divergence between **counterfactual observation distributions** under different forced actions:
  \[
  \text{Agency}(h) = JS\big(P(O_{t+1} \mid h, A_t = \text{flip}) \;\big\|\;
                           P(O_{t+1} \mid h, A_t = \text{hold})\big).
  \]

Empirical results (averaged by region):

| Region      | Mean JS (bits) | Interpretation                         |
|------------|----------------|----------------------------------------|
| Controlled | 0.6008         | High control: actions shape the future |
| Random     | 0.0001         | No control: actions irrelevant         |

Interpretation:

- The neural oracle has learned an internal **agency map**: it knows where actions matter.
- This scalar can be seen as a form of **empowerment** signal and could act as intrinsic reward in RL.

Connection to Neural CSSR:

- The environment-view equivalence relation is implicitly using these counterfactuals:
  histories in controlled vs random regions are clearly separated when we consider intervention responses.

---

## 3. Experiment Q3: Simple unsupervised factorization

**Question:** Can we recover separate environment and agent machines without labels in a simple setting?

### 3.1 Setup

Environment:

- 2 physical states `{L, R}` with “swap” dynamics:
  - If in `L`: `left` → `L`, `right` → `R`.
  - If in `R`: `left` → `R`, `right` → `L`.

Agent:

- 2 latent intentions `{GoLeft, GoRight}`.
- Policy (biased random walk):
  - `GoLeft`: \(P(\text{left}) = 0.9\), \(P(\text{right}) = 0.1\).
  - `GoRight`: \(P(\text{right}) = 0.9\), \(P(\text{left}) = 0.1\).
- Mode switching: 1% chance per step.

Data:

- Tokens: `obs_L`, `obs_R`, `act_left`, `act_right`.
- Stream:

  ```text
  obs_L act_right obs_R act_right obs_L ...
  ```

The transformer is trained on this joint stream.

### 3.2 Two Views clustering results

We apply the Two Views clustering on transformer histories using agglomerative clustering with \(k=2\).

Results:

| View        | Clusters | ARI vs ground truth | Interpretation                         |
|-------------|----------|---------------------|----------------------------------------|
| Env view    | 2        | 1.00                | Perfect recovery of `{L, R}`           |
| Agent view  | 2        | 0.77                | Strong recovery of `{GoLeft, GoRight}` |

Interpretation:

- From the same joint model:
  - The **Env view** (affordances) cleanly recovers physical states and is largely invariant to agent mode.
  - The **Agent view** (action distribution) recovers latent intent and largely ignores physical state.
- Factorization is possible and robust when the policy is **state-independent**: “Go Left” means the same thing everywhere.

Connection to Neural CSSR:

- Each view is an instance of Stage 1 with a different equivalence relation (what distribution we match: \(P(O \mid h, a)\) vs \(P(A \mid h)\)).
- In this simple setting, both environment and agent machines emerge cleanly from unsupervised clustering over the joint predictive structure.

---

## 4. Experiment Q4: Gridworld stress test and hierarchical factorization

**Question:** Can we still factorize when the agent is “smart,” i.e., its policy strongly depends on location?

### 4.1 Setup

Environment:

- 3×3 grid (states 0–8).
- Deterministic transitions:
  - Actions: `up`, `down`, `left`, `right`, `stay`.
  - Hitting a wall results in `stay`.

Agent:

- 2 latent goals:
  1. `Seek_0`: go to top-left (0,0).
  2. `Seek_8`: go to bottom-right (2,2).
- Policy:
  - For current goal, compute Manhattan distance for each action’s successor.
  - Choose an action that decreases distance with probability 0.8 (greedy), random action with probability 0.2.
- Goal switching: very rare (0.5% per step).

Data:

- Tokens: `obs_0` … `obs_8`, `act_up` … `act_stay`.
- Stream examples:

  ```text
  obs_4 act_up obs_1 act_left obs_0 act_stay obs_0 ...
  ```

### 4.2 Environment view: success

Using the Env view with an automatic JS threshold (e.g. \(\epsilon = 0.05\)):

- Clusters discovered: 9 (exactly the grid states).
- ARI vs true positions: 1.00 (perfect).

Insight:

- The transformer learned a **topological map** of the grid.
- Distinctions like “from 4, `up` → 1; from 1, `up` → wall” are captured despite the agent’s bias (some transitions are rarely seen but generalized correctly).
- Env view effectively recovers the **physics machine**.

### 4.3 Agent view: global factorization fails

Using the Agent view (\(P(A \mid h)\)):

- Clusters discovered: ~8 (mixed, entangled).
- ARI vs latent goals: ≈ 0.

Insight:

- The agent’s policy is **contextual**:
  - Under `Seek_0`, optimal move from state 5 is `left`, but from state 7 it is `up`.
  - There is no single global “Go Left” or “Go Right” signature that defines intent regardless of position.
- A global agent-view clustering cannot cleanly separate intent from location.

Conclusion:

- **Environment factorization works globally; agent factorization must be local or hierarchical.**

---

## 5. Hierarchical Oracle CSSR for agent factorization

The Q4 failure suggests a hierarchical factorization strategy:

1. **Map discovery (Env view).**  
   Run Env-view clustering or oracle CSSR to discover environment states \(\mathcal{S}_{\text{env}} = \{s_1, \dots, s_K\}\).

2. **State localization.**  
   Train a classifier (e.g., a probe) that maps histories \(h\) to environment states \(s\), using cluster assignments as labels.

3. **Contextualized policy views.**  
   For each environment state \(s\), collect histories \(H_s\) that localize to \(s\) and cluster them using the Agent view (\(P(A \mid h)\)) to discover **local modes** of behavior.

4. **Mode alignment across states.**  
   Align local modes across different \(s\) using transition patterns and trajectories:
   - “Mode A at state 1” and “Mode A at state 2” are the same global intent if there is high probability to move toward the same goal or produce consistent long-term behavior.

Effectively:

- First build an epsilon-machine for the **world** (environment/physics) using Env-view Neural CSSR.
- Then, condition on each world state and build **local agent machines** that describe how actions depend on state.
- A higher-level structure (meta-machine) glues local agent modes into global intents.

This matches the intuition:

> You cannot ask “what does the agent want?” until you know “where is the agent?”.

---

## 6. How this fits into the broader research program

These control experiments support and extend the core Neural CSSR narrative:

- **Transformers as predictive state encoders**
  - On joint (obs, act, obs, …) streams, a single autoregressive model internalizes both environmental dynamics and policy.
  - Using Env and Agent views, we can extract **physics maps**, **agency maps**, and **intent modes**.

- **Oracle CSSR on controlled processes**
  - The environment-view equivalence relation is a controlled version of Neural CSSR: histories are equivalent if their interventional futures under different actions match.
  - The agent-view equivalence relation focuses on policy, analogous to clustering by \(P(A \mid \text{history})\).

- **Factorization and interpretability**
  - In simple settings (Q2, Q3), we can factor the joint epsilon-machine into separate environment and agent machines.
  - In more complex settings (Q4), we see the need for **hierarchical** factorization, starting from environment states.

Taken together, these results show that:

- Neural sequence models trained on joint streams naturally learn **factorized causal models** (world + agent) even without architectural separation.
- Neural CSSR–style oracles and equivalence relations let us recover and **interpret** that structure.

This provides a concrete, empirical counterpart to the theory laid out in `docs/transformers_and_causal_states.md` and the algorithmic design in `cssr_discovery/2-stage.md`.

---

## 7. Synchronization length in controlled processes

Synchronization length is what makes all of these finite-state factorizations well-posed.

- For a finite-state epsilon-machine, the **sync length** \(L_{\text{sync}}\) is the smallest history length such that any suffix of length \(\ge L_{\text{sync}}\) uniquely determines the causal state (almost surely).
- Classical CSSR (and our two-stage variant) rely on picking \(L_{\max} \ge L_{\text{sync}}\) so that “long enough” histories can be clustered into stable states.

In the controlled / agentic setting there are several relevant sync lengths:

- **Joint sync length \(L_{\text{sync}}^{\text{joint}}\)**  
  The sync length of the full closed-loop process over `(obs, act, ...)`. Joint Neural CSSR needs \(L_{\max} \ge L_{\text{sync}}^{\text{joint}}\) to recover the joint epsilon-machine.

- **Environment sync length \(L_{\text{sync}}^{\text{env}}\)**  
  The sync length of the environment viewed as a controlled process \(P(O_{t+1} \mid h, \text{do}(A_t=a))\). Env-view CSSR needs \(L_{\max} \ge L_{\text{sync}}^{\text{env}}\) so that length-\(L_{\max}\) suffixes are rich enough to distinguish physical states via affordances.

- **Local agent sync lengths**  
  Once we condition on an environment state \(s\), the effective history needed to infer agent “mode” is often much shorter than the joint sync length. Hierarchical factorization exploits this:
  - Stage 1 (Env view): choose \(L_{\max} \ge L_{\text{sync}}^{\text{env}}\) to discover world states.
  - Stage 2 (Agent view per state): within each env state, only a short local history is needed to synchronize to an intent/policy mode.

In practice we do not know \(L_{\text{sync}}\) a priori, so we:

- Increase \(L_{\max}\) (and k-horizons) gradually.
- Monitor when the induced partitions **stabilize** (no further splits as \(L_{\max}\) or k increase).

This stabilization is the empirical signature that we have effectively reached a sync length for the view in question (joint, env, or local agent), and that further history length will not change the recovered causal structure. It justifies using fixed-\(L_{\max}\) two-stage CSSR in these controlled, agentic settings. 
