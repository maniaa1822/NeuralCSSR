Perfect, Stage A is all about **probing**, so the dataset just needs to be clean and labelled. Let’s make a concrete, implementable plan.

---

## 1. Choose a simple but non-trivial env + agent

Start with **Env-1 + Agent-1** (the 2×2 toy we discussed):

* **Environment states**: `E = {L, R}`
* **Observations**: `O = {obs_L, obs_R}` (just encode state)
* **Actions**: `A = {left, right}`

Environment dynamics:

* If in `L`:

  * `left` → stay in `L`
  * `right` → go to `R`
* If in `R`:

  * `right` → stay in `R`
  * `left` → go to `L`

**Agent internal modes**: `M = {GoRight, GoLeft}`

* In `GoRight`:

  * `P(right) = 0.9`, `P(left) = 0.1`
* In `GoLeft`:

  * `P(left) = 0.9`, `P(right) = 0.1`
* Mode switching:

  * At each step, with probability `p_switch` (e.g. 0.01), flip mode; otherwise keep same.

This gives **4 joint states**: `(L,GoRight)`, `(R,GoRight)`, `(L,GoLeft)`, `(R,GoLeft)`.

Later you can add Env-2 / Agent-2 with more states/modes, but this is enough for Stage A.

---

## 2. Decide sequence format for the transformer

Use a **single flat token sequence** over a small vocabulary:

* Token set:

  * `obs_L`, `obs_R`
  * `act_left`, `act_right`
* At each time step `t`, log **two tokens**:
  `obs_t`, then `act_t`

So one episode of length `T` looks like:

`obs_1, act_1, obs_2, act_2, ..., obs_T, act_T`

The transformer task: **next-token prediction** over this sequence.

---

## 3. What labels you store for probing

For every time step `t` (i.e. for each `(obs_t, act_t)` pair) store:

* `env_state_t` ∈ {L, R}
* `agent_mode_t` ∈ {GoRight, GoLeft}
* `joint_state_t` = (env_state_t, agent_mode_t) — you can encode as `{0,1,2,3}`
* Raw tokens:

  * `obs_token_t` ∈ {obs_L, obs_R}
  * `act_token_t` ∈ {act_left, act_right}

You’ll also keep:

* `episode_id`
* `step_index`

This becomes a nice table you’ll later join with transformer hidden states.

---

## 4. Data generation algorithm

For each episode:

1. Sample initial env state `E_1` ∈ {L, R} (e.g. uniform).
2. Sample initial agent mode `M_1` ∈ {GoRight, GoLeft} (uniform).
3. For `t = 1..T`:

   * **Observation**: `obs_t` is `obs_L` if `E_t = L`, else `obs_R`.
   * **Action**:

     * Draw `a_t` from policy given `M_t`:

       * If `M_t = GoRight`: `right` with 0.9, `left` with 0.1
       * If `M_t = GoLeft`: `left` with 0.9, `right` with 0.1
   * **Log labels**: `E_t`, `M_t`, `obs_t`, `a_t`.
   * **Environment transition**: update `E_{t+1}` using `(E_t, a_t)`.
   * **Mode transition**: with probability `p_switch`, flip `M_{t+1}`, else `M_{t+1} = M_t`.

Parameters you can fix:

* Episode length `T`: e.g. 100 steps.
* Number of episodes:

  * Train: 20k episodes
  * Val: 2k
  * Test: 2k
    (You can adjust, but this is plenty for a tiny transformer.)

---

## 5. Transformer training dataset

From the generated episodes:

* For each episode, create one long token sequence:
  `obs_1, act_1, obs_2, act_2, ..., obs_T, act_T`
* Optionally:

  * Add a special `BOS`/`EOS` token if you want.
* Split into:

  * Train sequences (from train episodes),
  * Validation / test sequences.

Train a small transformer (e.g. 2–4 layers, d_model 128–256) on **next-token cross-entropy** over this data.

---

## 6. Probing dataset (Stage A)

After training:

1. Run the trained transformer on **test episodes**, recording:

   * For each token position, the **hidden state** at a chosen layer (e.g. last layer) → `h_t`.
2. Join with your logged labels:

   * `h_t` ↔ `env_state_t`, `agent_mode_t`, `joint_state_t`, `obs_{t+1}`, `act_{t+1}`.

This gives you a probing dataset:

* Inputs: hidden representation `h_t`.
* Targets:

  * joint state label,
  * env state label,
  * agent mode label.

You can now:

* Train **linear probes** for:

  * joint state,
  * env state,
  * agent mode.
* Compare:

  * Single joint probe vs factorized probes (two heads).
* Measure how separable these are.

---

## 7. Optional extra variants (nice but not required)

Once the basic dataset is working, you can add:

1. **Env-2**: 3–4 env states in a ring or grid.
2. **Agent-2**: 3+ modes (e.g. “go left”, “go right”, “stay”).
3. **Stochastic observations**: add noise so observation isn’t perfectly equal to env state.

But for now, **Env-1 + Agent-1 as above is enough** to test Stage A: does the transformer’s representation allow env/agent factorization?

---

If you want, next we can turn this into a small Python pseudo-code sketch to make implementation straightforward.
