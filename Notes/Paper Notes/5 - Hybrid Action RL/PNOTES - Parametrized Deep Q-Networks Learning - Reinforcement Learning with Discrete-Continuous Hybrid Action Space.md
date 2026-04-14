[[Parametrized Deep Q-Networks Learning - Reinforcement Learning with Discrete-Continuous Hybrid Action Space.pdf]]

# **Abstract:**

The paper proposes **Parametrized Deep Q-Networks (P-DQN)**, a framework that handles discrete-continuous hybrid action spaces *without* approximation or relaxation. Most existing deep reinforcement learning (DRL) methods specialize in either purely discrete action spaces (DQN) or purely continuous action spaces (DDPG). Real-world sequential decision-making problems — particularly in Real-Time Strategy (RTS) games — exhibit a hierarchical, hybrid structure where an agent must first select a discrete action $k$ from $[K]$ and then specify a continuous parameter $x_k \in X_k$.

- P-DQN integrates DQN's discrete-action selection (the $\max$ operator) with DDPG's continuous-parameter optimization (deterministic policy gradients), avoiding the structural loss of discretization and the complexity overhead of relaxation.
- Empirical validation spans three environments: a point-mass simulation, the Half Field Offense (HFO) domain in RoboCup 2D soccer, and the solo mode of King of Glory (KOG), a commercial MOBA game.
- In HFO, P-DQN achieved a 99.7% scoring rate using 2 CPUs in 1 hour, compared to DDPG's three days on a Titan-X GPU. In KOG, P-DQN learned significantly faster than DDPG applied to a relaxed action space.

- *Context*:
	- P-DQN is a direct structural improvement over the relaxation approach of [[PNOTES - Deep Reinforcement Learning in Parameterized Action Space|Hausknecht & Stone (2016)]], which embedded hybrid actions into a higher-dimensional continuous space and applied standard DDPG. That approach works but forces the Q-network to learn over a bloated action space and cannot leverage human gameplay data (since human players produce discrete actions, not the continuous relaxation parameters $f_k$). P-DQN sidesteps both problems by preserving the natural $(k, x_k)$ hierarchy.

---

# 1. Introduction

The paper identifies a structural gap in DRL: most methods require the action space to be *either* finite and discrete (Go, Atari) *or* continuous (MuJoCo, Torcs). However, many practical problems — especially RTS and MOBA games — feature a **hierarchical hybrid action space** where the agent first selects a discrete action type $k$ and then specifies an associated continuous parameter $x_k$.

- The hybrid action space is formally defined as:

$$
A = \{(k, x_k) \mid x_k \in X_k \text{ for all } k \in [K]\}
$$

- Two existing approaches are identified as inadequate:
	- **Discretization**: Approximate each $X_k$ by a finite subset. This loses the natural structure of the continuous parameter space and requires exponentially many discrete actions for good coverage of multi-dimensional $X_k$.
	- **Relaxation** (Hausknecht & Stone, 2016): Define an approximate space $\tilde{A} = \{(f_{1:K}, x_{1:K}) \mid f_k \in F_k, x_k \in X_k\}$ where continuous variables $f_k$ encode discrete selections. This significantly increases the dimensionality of the action space.
- P-DQN's approach: define a **deterministic policy network** $x_k(\cdot; \theta)$ that maps states to continuous parameters, and an **action-value network** $Q(s, k, x_k; \omega)$ that evaluates discrete-continuous action pairs. The discrete action is selected by taking $\max_k Q$, while the continuous parameters are optimized via gradient descent on $Q$.

- *Context*:
	- The relaxation approach of Hausknecht & Stone (2016) forces the agent to learn over the full Cartesian product $F_1 \times \ldots \times F_K \times X_1 \times \ldots \times X_K$. For KOG with $K = 6$ action types and multiple parameters per type, this produces a substantially higher-dimensional action space than the natural $(k, x_k)$ structure. P-DQN avoids this by treating discrete selection and continuous parameterization as fundamentally different operations — the $\max$ for discrete, gradient ascent for continuous — rather than conflating both into a single continuous output.

---

# 2. Background

## 2.1 Reinforcement Learning Methods for Finite Action Space

The paper reviews two classes of RL methods for discrete action spaces:

- **Value-based methods**: Estimate $Q^*$ and derive the greedy policy. **Q-learning** is founded on the Bellman equation:

$$
Q(s, a) = \mathbb{E}_{r_t, s_{t+1}}\left[r_t + \gamma \max_{a' \in A} Q(s_{t+1}, a') \mid s_t = s, a_t = a\right]
$$

- **Deep Q-Network (DQN)** (Mnih et al., 2013, 2015) approximates $Q^*$ with a neural network $Q(s, a; w)$ and updates weights by minimizing:

$$
L_t(w) = \left\{Q(s_t, a_t; w) - \left[r_t + \gamma \max_{a' \in A} Q(s_{t+1}, a'; w_t)\right]\right\}^2
$$

- **Policy-based methods**: Directly optimize $J(\pi_\theta)$ via the stochastic policy gradient theorem (Sutton et al., 2000):

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s,a}\left[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s, a)\right]
$$

- **Actor-critic methods** (Konda & Tsitsiklis, 2000; Mnih et al., 2016) combine both perspectives by using a neural network to estimate the value function while updating the policy via gradient ascent.

- *Context*:
	- The $\max_{a' \in A}$ operator in the Bellman equation is computationally tractable only when $A$ is finite. This is the fundamental barrier that prevents DQN from handling continuous or hybrid action spaces directly. P-DQN's core insight is to decompose this operator: use $\max_k$ over discrete actions (tractable) while replacing $\sup_{x_k}$ over continuous parameters with a deterministic policy network.

## 2.2 Reinforcement Learning Methods for Continuous Action Space

When $A$ is continuous, the $\max_a Q(s, a; w)$ operation in the Bellman equation becomes a non-convex optimization problem — NP-hard in the worst case because $Q(s, a; w)$ is non-convex in $a$.

- **Continuous Q-learning** (Gu et al., 2016) circumvents this by parameterizing the advantage function $A(s, a; \theta_A)$ as a quadratic in $a$, yielding an analytic maximum.
- **Deterministic Policy Gradient (DPG)** (Silver et al., 2014) proposes a deterministic policy $\mu_\theta : S \to A$ and derives the gradient:

$$
\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s \sim \rho^{\mu_\theta}}\left[\nabla_\theta \mu_\theta(s) \nabla_a Q^{\mu_\theta}(s, a)\big|_{a = \mu_\theta(s)}\right]
$$

- **DDPG** (Lillicrap et al., 2016) scales DPG to high-dimensional continuous spaces using deep neural networks.

- *Context*:
	- The DPG theorem shows that the deterministic policy gradient is the limit of the stochastic policy gradient as the policy variance goes to zero. This is the theoretical foundation that P-DQN adapts: the deterministic policy network $x_k(\cdot; \theta)$ plays the same role as $\mu_\theta$ in DDPG, but scoped to the continuous parameter associated with each discrete action $k$.

## 2.3 Reinforcement Learning Methods for Hybrid Action Space

The paper reviews three prior approaches to hybrid action spaces:

- **Hausknecht & Stone (2016)**: Apply DDPG on the relaxed continuous action space $\tilde{A}$ (Eq. 1.2). The discrete action is selected via $\arg\max_i f_i$ or $\text{softmax}(f)$. This is effectively an on-policy Q-function estimator when softmax is used.
- **Masson et al. (2016)**: Alternate between Q-learning (SARSA) for discrete actions and policy search (eNAC) for continuous parameters. This is on-policy and requires assuming a distribution over continuous parameters.
- **Khamassi et al. (2017)**: Q-learning for discrete actions and policy gradient for continuous parameters. Also on-policy.

- *Context*:
	- All three prior methods have structural limitations. Hausknecht & Stone's relaxation inflates the action space. Masson et al. and Khamassi et al. are on-policy, meaning they cannot use experience replay and are therefore less sample-efficient. P-DQN is the first off-policy method (when $n = 1$) that handles hybrid action spaces with deterministic policies, enabling experience replay and more efficient use of collected data.

---

# 3. Parametrized Deep Q-Networks (P-DQN)

This section introduces the core algorithmic framework. The paper considers an MDP with the hybrid action space $A$ defined in Eq. 1.1, where the action-value function is denoted $Q(s, k, x_k)$ for state $s$, discrete action $k \in [K]$, and continuous parameter $x_k \in X_k$.

- The **Hybrid Bellman Equation** extends the classical Bellman equation to account for the two-level optimization over discrete and continuous components (**See MATHEMATICS §2**):

$$
Q(s_t, k_t, x_{k_t}) = \mathbb{E}_{r_t, s_{t+1}}\left[r_t + \gamma \max_{k \in [K]} \sup_{x_k \in X_k} Q(s_{t+1}, k, x_k) \mid s_t = s, a_t = (k_t, x_{k_t})\right]
$$

- The $\sup_{x_k \in X_k}$ is computationally intractable for a neural network Q-function. However, if the optimal continuous parameter $x_k^* = \arg\sup_{x_k} Q(s, k, x_k)$ is known for each $k$, the right-hand side reduces to a tractable $\max_k$ operation.
- The paper defines a **deterministic policy network** $x_k(\cdot; \theta) : S \to X_k$ that approximates $x_k^*$ for each discrete action $k$ (**See MATHEMATICS §3**):

$$
Q(s, k, x_k(s; \theta); \omega) \approx \sup_{x_k \in X_k} Q(s, k, x_k; \omega) \quad \text{for each } k \in [K]
$$

- With the policy network providing continuous parameters, the **n-step target** $y_t$ is defined as (**See MATHEMATICS §4**):

$$
y_t = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n \max_{k \in [K]} Q(s_{t+n}, k, x_k(s_{t+n}; \theta_t); \omega_t)
$$

- The **loss functions** for training the value and policy networks are (**See MATHEMATICS §5, §6**):

$$
\ell_t^Q(\omega) = \frac{1}{2}\left[Q(s_t, k_t, x_{k_t}; \omega) - y_t\right]^2 \quad \text{and} \quad \ell_t^\Theta(\theta) = -\sum_{k=1}^{K} Q(s_t, k, x_k(s_t; \theta); \omega_t)
$$

- The two networks are updated on different timescales: $\omega$ with stepsize $\alpha_t$ and $\theta$ with stepsize $\beta_t$, where $\alpha_t$ is asymptotically negligible compared to $\beta_t$. This **two-timescale update rule** (Borkar, 1997) ensures that from the policy network's perspective, the value network appears approximately stationary.
	- The stepsizes $\{\alpha_t, \beta_t\}$ must satisfy the **Robbins-Monro condition** for stochastic approximation validity.

- *Context*:
	- The two-timescale trick is essential for convergence. The policy network $x_k(\cdot; \theta)$ is trained to maximize $Q$ with $\omega$ held approximately fixed. If both networks updated at the same rate, the moving target problem would be amplified: $\theta$ would chase a Q-landscape that $\omega$ is simultaneously reshaping. By making $\alpha_t \ll \beta_t$, the policy converges on a faster timescale while the value function adapts slowly, mimicking the fixed-$\omega$ assumption that justifies gradient ascent on $\ell_t^\Theta$.

- **Algorithm 1 (P-DQN with Experience Replay)**: The agent maintains a replay buffer $D$ from which mini-batches of $B$ transitions are sampled. At each step:
	1. Compute $x_k \leftarrow x_k(s_t; \theta_t)$ for all $k$.
	2. Select action via $\epsilon$-greedy: with probability $\epsilon$, sample from a distribution $\xi$ over $A$; otherwise, select $k_t = \arg\max_k Q(s_t, k, x_k; \omega_t)$.
	3. Execute action, observe reward $r_t$ and next state $s_{t+1}$.
	4. Store transition in $D$, sample mini-batch, compute gradients, and update weights.

- *Context*:
	- The exploration distribution $\xi$ is typically uniform over $A$, meaning a uniform random discrete action paired with uniformly random continuous parameters. This is simpler than Ornstein-Uhlenbeck noise used in DDPG, and naturally covers the discrete dimension of the action space.

- **Remark 3.1** identifies three key differences from prior methods:
	1. **Explicit Q-maximization**: P-DQN selects discrete actions by $\arg\max_k Q(s, k, x_k; \omega)$, whereas Hausknecht & Stone (2016) parameterize discrete selections as continuous values $f_k$ and use $\arg\max_i f_i$ or $\text{softmax}(f)$.
	2. **Off-policy**: P-DQN is off-policy (when $n = 1$), enabling experience replay. Masson et al. (2016) and Khamassi et al. (2017) are on-policy. Hausknecht & Stone's Q-network is also on-policy when using softmax selection.
	3. **Human data compatibility**: P-DQN can incorporate human demonstration data for pre-training because human players produce discrete action labels $k$. Hausknecht & Stone's relaxation requires data for the continuous parameters $f_k$, which human players do not generate.

- *Context*:
	- The human data advantage is practically significant. In commercial game AI applications like KOG, vast quantities of human gameplay recordings exist. P-DQN can use these for supervised pre-training of both the Q-network and the policy network, then fine-tune via RL. The relaxation approach cannot, since it would need to infer what continuous $f_k$ values correspond to a human player's discrete action choice — a fundamentally ill-posed problem.

## 3.1 Asynchronous n-step P-DQN

P-DQN can be accelerated using asynchronous gradient descent following Mnih et al. (2016). Multiple local processes run independent game environments, compute local gradients, and send them to a global **parameter server** that maintains the shared weights $\omega$ and $\theta$.

- Each local process synchronizes parameters from the server, runs $t_{\max}$ steps (or until a terminal state), and computes accumulated gradients $d\omega$ and $d\theta$.
- The target computation unrolls backward from the end of the local trajectory:

$$
y = \begin{cases} 0 & \text{for terminal } s_t \\ \max_{k \in [K]} Q\left[s_t, k, x_k(s_t; \theta'); \omega'\right] & \text{for non-terminal } s_t \end{cases}
$$

followed by $y \leftarrow r_i + \gamma \cdot y$ for $i = t-1, \ldots, t_{\text{start}}$.

- Global parameters are updated using **RMSProp** with the aggregated gradients.

- *Context*:
	- The n-step bootstrap introduces a critical trade-off. When $n > 1$, the algorithm is **no longer off-policy**: the n-step target uses rewards collected under a potentially outdated policy (the local parameters $\omega'$, $\theta'$ may diverge from the global parameters during the local trajectory). This sacrifices the theoretical purity of off-policy learning in exchange for faster convergence in delayed-reward or long-episode settings. The variance reduction from aggregating gradients across many independent workers partially compensates for the bias introduced by stale parameters.

---

# 4. Experiments

The paper validates P-DQN across three environments of increasing complexity, comparing against Hausknecht & Stone (2016)'s DDPG on relaxed spaces and DQN with discretized actions.

- A **dueling layer** (Wang et al., 2016) replaces the last fully-connected layer in both DQN and P-DQN to accelerate training.

## 4.1 A Simulation Example

**Environment**: A 2×2 squared plate with a unit point mass. The goal is to pull the mass into a target circle (radius $r = 0.1$) using either a unit force with constant direction or a soft brake that reduces velocity by 0.1. Physics follows Newton's mechanics on a frictionless surface.

- **State**: 8-dimensional vector $s = (x, \dot{x}, y, d(x, y), \mathbf{1}_{d(x,y) < r})$ capturing position, velocity, target location, distance, and an indicator for being within the target.
- **Action space**: $A = \{(\text{brake}), (\text{pull}, \alpha)\}$ where $\alpha$ is the direction of force.
- **Reward**: $r_t = d(x_t, y) - d(x_{t+1}, y) + \mathbf{1}_{\text{goal}}$ — distance reduction plus a terminal bonus.
- **Episode termination**: Mass stops in the target circle, leaves the plate, or episode length exceeds 200.

- **Periodic parameter handling**: The direction $\alpha$ is periodic, so the paper represents it as $(\cos\alpha, \sin\alpha)$ — a normalized 2D vector — and adds a normalization layer at the network output. This transformation is used across all three experiments.

- *Context*:
	- The periodic parameter problem is subtle but critical. If $\alpha$ is represented as a scalar angle, the values $-179°$ and $+179°$ are numerically distant but semantically adjacent. The network would need to learn this discontinuity, wasting capacity. The $(\cos\alpha, \sin\alpha)$ representation eliminates the discontinuity entirely by mapping the angular space onto a smooth manifold.

- **Results**: P-DQN converged in fewer than 150k iterations. Five independent agents were trained for each method and evaluated with 100 trials each:
	- P-DQN converged faster and more stably than both DQN and DDPG.
	- DQN converged quickly but to a sub-optimal solution due to the 8-direction discretization, and exhibited high variance.
	- DDPG on the relaxed space converged more slowly and less stably.

## 4.2 HFO (Half Field Offense)

The paper uses the same experimental settings as Hausknecht & Stone (2016): scoring goals without a goalie.

- **State**: 58 continuously-valued features from the Helios-Agent2D world model, encoding relative positions of the ball, goal, and landmarks.
- **Action space**: $\{\text{Dash}(\text{power}, \text{direction}), \text{Turn}(\text{direction}), \text{Kick}(\text{power}, \text{direction})\}$ with directions in $[-180, 180]$ degrees and power in $[0, 100]$.
- **Reward**: The same hand-crafted intensive reward used in Hausknecht & Stone (2016):

$$
r_t = d_t(a, b) - d_{t+1}(a, b) + I_{t+1}^{\text{kick}} + 3(d_t(b, g) - d_{t+1}(b, g)) + 5 \cdot I_{t+1}^{\text{goal}}
$$

- **Training**: Asynchronous P-DQN with 24 workers. Training cost approximately **1 hour on 2 Intel Xeon CPUs**.

- *Context*:
	- The computational efficiency gain is striking: 1 hour on 2 CPUs versus 3 days on an NVidia Titan-X GPU for the DDPG baseline in Hausknecht & Stone (2016). This is partly due to the asynchronous architecture (parallelism across 24 workers) and partly due to P-DQN's more efficient use of the action space structure. By avoiding the relaxation overhead, the Q-network has fewer output dimensions and the optimization landscape is simpler.

- **Learning curve interpretation**: In the first 250k iterations, the agent learns to approach and kick the ball — mean episode length increases because episodes are set to end if the ball is not kicked within 100 frames. After 250k iterations, the agent learns to score as quickly as possible, driven by the discount factor $\gamma$.

- **Comparison results** (9 independently trained P-DQN agents, evaluated over 1000 trials each):

| Agent | Scoring % | Avg. Steps to Goal |
|---|---|---|
| Helios' Champion | 0.962 | 72.0 |
| SARSA | 0.81 | 70.7 |
| Best DDPG | 1.00 | 108.0 |
| P-DQN (best) | 0.997 | 78.1 |
| P-DQN (median) | 0.992 | 78.7 |
| P-DQN (worst) | 0.979 | 78.5 |

- P-DQN scores more accurately and faster than DDPG with **more stable performance** across independently trained agents.
- P-DQN's average steps to goal (78.1–87.9) are substantially lower than DDPG's (104.8–119.1), indicating the agent learns more efficient scoring strategies.

- *Context*:
	- The narrower variance across P-DQN agents (0.979–0.997 scoring rate) compared to DDPG agents (0.80–1.00) suggests that P-DQN's optimization landscape is smoother. The relaxed space used by DDPG introduces extraneous dimensions that create local optima and sensitivity to initialization. P-DQN, by preserving the natural action hierarchy, provides a more structured landscape that different random seeds navigate more consistently.

## 4.3 Solo Mode of King of Glory

**King of Glory (KOG)** is the most popular mobile MOBA game in China with over 80 million daily active players (as of July 2017). The experiment focuses on the **solo mode** (1v1) using the hero **Lu Ban** against the internal game AI.

- **State**: 179-dimensional feature vector from the game engine, consisting of:
	- Basic attributes of units (hero stats, level, equipment)
	- Relative positions and attack relations between units
- **Action space**: $K = 6$ discrete action types, each with associated continuous parameters:

| Action Type | Parameter | Description |
|---|---|---|
| Move | $\alpha$ | Direction of movement |
| Attack | — | Attack default target |
| UseSkill1 | $(x, y)$ | Skill at position $(x, y)$ |
| UseSkill2 | $\alpha$ | Skill in direction $\alpha$ |
| UseSkill3 | $\alpha$ | Skill in direction $\alpha$ |
| Retreat | — | Return to base |

- **Usable action modification**: Not all 6 actions are always available due to skill level-up requirements, **Magic Point (MP)** constraints, or **Cool Down (CD)** timers. The paper modifies the $\max_k$ operator to $\max_{k \in [K], k \text{ is usable}}$ when selecting actions and computing multi-step targets.

- *Context*:
	- The "usable action" modification is a practical necessity that changes the mathematical structure of the Bellman operator. By restricting the $\max$ to feasible actions, the effective action space becomes *state-dependent* — a departure from the fixed $A$ assumed in the theoretical framework. This introduces no formal complications (state-dependent action sets are standard in MDP theory) but requires implementation care to ensure the Q-network does not waste capacity estimating values for infeasible actions.

## 4.4 Reward for KOG

The paper defines a **reward shaping** scheme using time-differenced game statistics:

- **Gold Difference** ($GD$): $\text{Gold}_0 - \text{Gold}_1$
- **Health Point Difference** ($HPD$): $\text{HeroRelativeHP}_0 - \text{HeroRelativeHP}_1$
- **Kill/Death** ($KD$): $\text{Kills}_0 - \text{Kills}_1$
- **Tower HP Difference** ($THP$): $\text{TowerRelativeHP}_0 - \text{TowerRelativeHP}_1$
- **Base HP Difference** ($BHP$): $\text{BaseRelativeHP}_0 - \text{BaseRelativeHP}_1$
- **Tower Destroyed** ($TD$): $\text{AliveTower}_0 - \text{AliveTower}_1$
- **Winning Game** ($W$): $\text{AliveBase}_0 - \text{AliveBase}_1$
- **Moving Forward** ($MF$): $x + y$ (coordinates of the hero)

The overall reward is a weighted sum of time-differenced statistics:

$$
r_t = 0.5 \times 10^{-5}(MF_t - MF_{t-1}) + 0.5(HPD_t - HPD_{t-1} + KD_t - KD_{t-1} + TD_t - TD_{t-1}) + 0.001(GD_t - GD_{t-1}) + (THP_t - THP_{t-1} + BHP_t - BHP_{t-1}) + 2W
$$

- Coefficients are set roughly inversely proportional to the scale of each statistic. The paper notes the algorithm is not very sensitive to reasonable changes in these coefficients.
- Training uses **Algorithm 2** with 48 parallel workers and **frame skipping** (actions every 3 frames, or 0.2 seconds).

- *Context*:
	- The reward function is a carefully engineered dense signal for an otherwise sparse-reward problem. The time-differencing of statistics ensures the reward reflects *changes* in game state rather than absolute values, providing a gradient signal for incremental improvement. The tiny coefficient on $MF$ ($5 \times 10^{-6}$) is an exploration incentive — pushing the agent to move forward and engage rather than staying safely near base — while being small enough not to dominate the signal from combat outcomes.

- **Results** (Figure 5 comparison):
	- P-DQN learns the value and policy networks significantly faster than DDPG on the relaxed space.
	- **Episode length dynamics**: Average game length increases initially, peaks when the two players' strengths are approximately matched, and decreases as the agent begins to dominate — a signature of successful training in adversarial settings.
	- Total rewards increase consistently in both training and validation.

- *Context*:
	- The inverted-U shape of episode length is a reliable diagnostic for learning progress in competitive games. Early short games indicate quick losses; lengthening games indicate the agent can survive longer; shortening games after the peak indicate the agent is winning decisively. The fact that DDPG's curves (Figure 5b) are noisier and slower to develop the same pattern confirms P-DQN's superior sample efficiency in the KOG domain.

---

# 5. Conclusion

The paper establishes P-DQN as an effective framework for hybrid discrete-continuous action spaces. By extending DQN with deterministic policies for continuous parameters — rather than approximating or relaxing the action space — P-DQN preserves the natural structure of the problem and achieves superior efficiency and stability across all three experimental domains.

- The key architectural insight is that discrete actions should be selected via $\max_k Q$, not via continuous parameterization and softmax/argmax. This keeps the discrete dimension discrete and the continuous dimension continuous.
- The method replaces the **Inverting Gradients** technique from Hausknecht & Stone (2016) with a simpler **Square Loss Penalty** on out-of-range parameters, achieving comparable constraint enforcement with less implementation complexity.

- *Context*:
	- The replacement of Inverting Gradients with a Square Loss Penalty is a pragmatic simplification noted in the appendix. Inverting Gradients smoothly attenuates and inverts the gradient near boundaries, but requires custom gradient manipulation at each training step. The Square Loss Penalty simply adds a quadratic cost term for parameters that exceed their bounds, leveraging standard gradient descent to push them back. This is a significant practical contribution: it means standard neural network training infrastructure can enforce action bounds without custom gradient hooks.

---

# Appendix

## A.1 More Information on King of Glory

KOG is a MOBA game where two teams fight to destroy the opposing base. Three lanes connect the bases, each guarded by three towers. Heroes advance levels and gain gold by killing units and destroying towers. In **solo mode**, only the middle lane is active, and both players control the same hero.

- A typical solo game lasts 10–20 minutes with instantaneous decision requirements.
- Four main RL challenges: (1) massive state space, (2) complicated hybrid action space, (3) ill-defined reward function, (4) real-time decision-making (ruling out heuristic search).

## A.2 Parameter Settings

- **Simulation**: Minibatch $B = 32$, replay memory 10k, network sizes $x(\theta)$: 64-32, $Q(\omega)$: 64-32-32, ReLU activation, $\epsilon$-greedy annealed from 1 to 0.1 over 30k iterations.
- **HFO**: $B = 32$, replay memory 1k per worker, network sizes $x(\theta)$: 256-128-64, $Q(\omega)$: 256-128-64-64, $\epsilon$ annealed from 1 to 0.1 over 150k iterations. Uses **Square Loss Penalty** instead of Inverting Gradients for parameter bounding.
- **KOG**: Same network sizes as HFO. Frame skipping parameter set to 2 (actions every 3 frames). $t_{\max} = 20$ (4 seconds). $\epsilon = 0.255$ with non-uniform action sampling: first 5 actions sampled at probability 0.05 each, "Retreat" at 0.005. If the sampled action is infeasible, the greedy policy over feasible actions is executed instead.

- *Context*:
	- The non-uniform exploration probabilities in KOG reflect domain knowledge: retreating is rarely the optimal action and excessive retreat exploration would waste training episodes. The effective exploration rate is lower than $\epsilon$ because infeasible actions trigger fallback to the greedy policy, creating a natural curriculum where exploration decreases as the agent learns to use its skills more aggressively.

---

# MATHEMATICS

The mathematical framework of P-DQN proceeds from the definition of hybrid action spaces through the modified Bellman optimality equation to the two-network training architecture. The derivation chain establishes how the intractable supremum over continuous parameters in the hybrid Bellman equation is replaced by a learnable deterministic policy, reducing the hybrid-action problem to a tractable discrete $\max$ operation augmented with gradient-based continuous optimization.

### 1. Hybrid Action Space

The foundational definition structures the action space as a union of discrete action types, each paired with continuous parameters.

**Hybrid Action Space ($A$):**

$$
A = \{(k, x_k) \mid x_k \in X_k \text{ for all } k \in [K]\} \tag{1}
$$

where $K$ is the number of discrete action types, $[K] = \{1, \ldots, K\}$ is the discrete action set, $k$ is a discrete action index, $x_k \in X_k$ is the continuous parameter associated with action $k$, and $X_k$ is a compact set in Euclidean space. An action $a \in A$ is a tuple $(k, x_k)$.

This contrasts with the **relaxed action space** of Hausknecht & Stone (2016):

$$
\tilde{A} = \{(f_{1:K}, x_{1:K}) \mid f_k \in F_k, x_k \in X_k, \forall k \in [K]\} \tag{2}
$$

where $f_k \in F_k \subseteq \mathbb{R}$ are continuous proxies for the discrete action selection. The relaxed space $\tilde{A}$ embeds the $K$-way discrete choice into $\mathbb{R}^K$, significantly increasing dimensionality. P-DQN avoids this by preserving the natural $(k, x_k)$ structure.

### 2. Hybrid Bellman Equation

Extending the classical Bellman equation to the hybrid action space requires a two-level optimization: a $\sup$ over continuous parameters for each discrete action, followed by a $\max$ over discrete actions.

**Hybrid Bellman Equation ($Q$):**

$$
Q(s_t, k_t, x_{k_t}) = \mathbb{E}_{r_t, s_{t+1}}\left[r_t + \gamma \max_{k \in [K]} \sup_{x_k \in X_k} Q(s_{t+1}, k, x_k) \;\middle|\; s_t = s, a_t = (k_t, x_{k_t})\right] \tag{3}
$$

where $Q(s, k, x_k)$ is the action-value function for state $s$, discrete action $k$, and continuous parameter $x_k$, $r_t$ is the immediate reward, $\gamma \in [0, 1]$ is the discount factor, and the $\sup$ is taken over the compact set $X_k$ for each $k$. The inner $\sup$ solves for $x_k^* = \arg\sup_{x_k \in X_k} Q(s_{t+1}, k, x_k)$, and the outer $\max$ selects the discrete action with the highest Q-value at $x_k^*$.

The $\sup_{x_k \in X_k}$ is computationally intractable for a neural network $Q$-function because $Q(s, k, x_k; \omega)$ is non-convex in $x_k$. However, the right-hand side is efficiently evaluable once $x_k^*$ is provided. This motivates the introduction of a deterministic policy network.

### 3. Deterministic Policy Approximation

For a fixed Q-function, the mapping $x_k^Q : S \to X_k$ defined by $x_k^Q(s) = \arg\sup_{x_k \in X_k} Q(s, k, x_k)$ is a deterministic function. The paper approximates this with a neural network.

**Deterministic Policy Approximation:**

$$
Q(s, k, x_k(s; \theta); \omega) \approx \sup_{x_k \in X_k} Q(s, k, x_k; \omega) \quad \text{for each } k \in [K] \tag{4}
$$

where $x_k(\cdot; \theta) : S \to X_k$ is the deterministic policy network with weights $\theta$, and $\omega$ are the weights of the Q-network. With this approximation, the Hybrid Bellman Equation reduces to:

$$
Q(s_t, k_t, x_{k_t}) \approx \mathbb{E}_{r_t, s_{t+1}}\left[r_t + \gamma \max_{k \in [K]} Q(s_{t+1}, k, x_k^Q(s_{t+1})) \;\middle|\; s_t = s\right]
$$

which resembles the classical DQN Bellman equation with $A = [K]$, since the continuous optimization has been absorbed into the policy network.

### 4. n-step Target

For a fixed $n \geq 1$, the multi-step target aggregates $n$ observed rewards before bootstrapping from the Q-network.

**n-step Target ($y_t$):**

$$
y_t = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n \max_{k \in [K]} Q(s_{t+n}, k, x_k(s_{t+n}; \theta_t); \omega_t) \tag{5}
$$

where $r_{t+i}$ are the observed rewards over the $n$-step window, $\gamma^i$ is the $i$-step discount, $\omega_t$ and $\theta_t$ are the current weights at step $t$, and the terminal condition sets the bootstrapped Q-value to zero when $s_{t+n}$ is terminal. When $n = 1$, this reduces to the standard one-step target $y_t = r_t + \gamma \max_k Q(s_{t+1}, k, x_k(s_{t+1}; \theta_t); \omega_t)$.

The n-step target introduces a bias-variance trade-off: larger $n$ reduces the reliance on the Q-estimate (lower bias from bootstrapping) but increases variance from the accumulated stochastic rewards. When $n > 1$, the algorithm loses its off-policy property because the intermediate rewards $r_{t+1}, \ldots, r_{t+n-1}$ were collected under a potentially different policy.

### 5. Value Network Loss

The Q-network weights $\omega$ are updated by minimizing the squared difference between the predicted Q-value and the n-step target.

**Value Network Loss ($\ell_t^Q$):**

$$
\ell_t^Q(\omega) = \frac{1}{2}\left[Q(s_t, k_t, x_{k_t}; \omega) - y_t\right]^2 \tag{6}
$$

where $Q(s_t, k_t, x_{k_t}; \omega)$ is the predicted action-value for the observed transition $(s_t, k_t, x_{k_t})$, and $y_t$ is the n-step target from $\S 4$. The gradient $\nabla_\omega \ell_t^Q(\omega)$ is used to update $\omega$ with stepsize $\alpha_t$.

### 6. Policy Network Loss

The policy network weights $\theta$ are updated to maximize the Q-value across all discrete actions, with value weights $\omega_t$ held fixed.

**Policy Network Loss ($\ell_t^\Theta$):**

$$
\ell_t^\Theta(\theta) = -\sum_{k=1}^{K} Q(s_t, k, x_k(s_t; \theta); \omega_t) \tag{7}
$$

where the sum runs over all $K$ discrete actions, $x_k(s_t; \theta)$ is the policy network's output for action $k$ at state $s_t$, and $\omega_t$ are the current (fixed) Q-network weights. The negative sign converts the maximization of Q into a minimization problem suitable for gradient descent. The gradient $\nabla_\theta \ell_t^\Theta(\theta)$ is used to update $\theta$ with stepsize $\beta_t$.

The policy network is updated on a **faster timescale** than the value network ($\beta_t \gg \alpha_t$ asymptotically), satisfying the two-timescale stochastic approximation conditions (Borkar, 1997) under the Robbins-Monro constraint.

### 7. KOG Reward Function

The KOG reward is a weighted combination of time-differenced game statistics, providing dense feedback for an otherwise sparse-reward environment.

**KOG Reward ($r_t$):**

$$
\begin{aligned}
r_t &= 0.5 \times 10^{-5}(MF_t - MF_{t-1}) \\
&\quad + 0.5(HPD_t - HPD_{t-1} + KD_t - KD_{t-1} + TD_t - TD_{t-1}) \\
&\quad + 0.001(GD_t - GD_{t-1}) \\
&\quad + (THP_t - THP_{t-1} + BHP_t - BHP_{t-1}) \\
&\quad + 2W
\end{aligned} \tag{8}
$$

where $MF_t = x + y$ is the hero's forward position (exploration incentive), $HPD_t$ is the health point difference, $KD_t$ is the kill/death difference, $TD_t$ is the towers-destroyed difference, $GD_t$ is the gold difference, $THP_t$ is the tower HP difference, $BHP_t$ is the base HP difference, and $W = \text{AliveBase}_0 - \text{AliveBase}_1$ is the winning indicator. Subscript 0 denotes the agent's side, subscript 1 denotes the opponent. Coefficients are set roughly inversely proportional to each statistic's scale.

---

### **1. Problem:**

The paper addresses the inability of existing DRL methods to handle **discrete-continuous hybrid action spaces** without structural loss. Standard DQN requires a finite action set, and standard DDPG requires a continuous action space. Real-world control problems — particularly in RTS/MOBA games — feature a hierarchical structure where an agent selects a discrete action type $k$ and then specifies continuous parameters $x_k$. Prior solutions either discretize the continuous parameters (losing precision) or relax the discrete-continuous structure into a purely continuous space (inflating dimensionality and preventing the use of human demonstration data).

### **2. Setup:**

The method assumes a Markov Decision Process with a parameterized action space $A = \{(k, x_k) \mid x_k \in X_k, k \in [K]\}$ where $X_k$ are compact subsets of Euclidean space. The architecture consists of two networks: a deterministic policy network $x_k(\cdot; \theta)$ mapping states to continuous parameters, and an action-value network $Q(s, k, x_k; \omega)$ mapping state-action pairs to scalar values. The networks are updated on different timescales using stochastic gradient descent with experience replay (Algorithm 1) or asynchronous parallel training (Algorithm 2). Experiments range from a 2D physics simulation to a full commercial MOBA game with 179-dimensional state and 6 action types.

### **3. Key Idea:**

P-DQN integrates value-based discrete selection ($\max_k Q$) with policy-gradient-based continuous optimization ($\nabla_\theta \ell_t^\Theta$) in a single architecture. By approximating the intractable $\sup_{x_k} Q$ with a deterministic policy network, the hybrid Bellman equation reduces to a standard discrete $\max$ operation — preserving the natural action hierarchy without relaxation or discretization.

### **4. Assumptions:**

**Explicit:**
- The environment follows an MDP structure $\{S, A, p, p_0, \gamma, r\}$
- Continuous parameter spaces $X_k$ are compact subsets of Euclidean space
- Stepsizes $\alpha_t$, $\beta_t$ satisfy the Robbins-Monro condition for stochastic approximation convergence
- Actions follow the hierarchical $(k, x_k)$ structure where each $k$ is associated with a specific $X_k$

**Implicit:**
- The 179-dimensional feature vector in KOG provides sufficient state representation for approximately Markovian decision-making
- The deterministic policy network can adequately approximate the true $\arg\sup_{x_k} Q$ despite the non-convexity of the Q-function in $x_k$
- In KOG, discrete actions are assumed to have a well-defined "usability" flag (accounting for CD/MP), modifying the $\max_k$ operator to a state-dependent feasible set
- The asynchronous implementation assumes access to significant parallel computational resources (24–48 workers)
- The Square Loss Penalty provides sufficient parameter bounding without the more sophisticated Inverting Gradients technique

### **5. Limitation:**

- **Scalability beyond solo mode**: While successful in 1v1 KOG, the method has not been tested in 5v5 settings where strategic collaboration and a vastly larger state space introduce qualitatively different challenges.
- **n-step off-policy trade-off**: Using $n > 1$ sacrifices the off-policy property, which is one of P-DQN's claimed advantages over prior methods. The optimal $n$ is domain-dependent and must be tuned.
- **Periodic parameter representation**: Raw angular parameters fail; the $(\cos\alpha, \sin\alpha)$ transformation is required. This is a manual design choice that does not generalize to arbitrary parameter topologies.
- **Reward engineering**: All three experiments rely on hand-crafted, domain-specific reward functions. The framework provides no mechanism for automatic reward discovery.
- **Approximation quality**: No theoretical guarantee that the deterministic policy network actually achieves $\sup_{x_k} Q$ — the approximation in Eq. 3.2 is justified empirically but not formally.

### **6. Relevance & Open Questions:**

P-DQN demonstrates that preserving the natural structure of hybrid action spaces yields substantial gains in efficiency, stability, and data compatibility over relaxation-based approaches. This has implications for any domain where actions combine categorical decisions with continuous parameterization — robotics (tool selection + motion path), financial trading (order type + size/price), and game AI.

- **Open**: Can the Square Loss Penalty universally replace Inverting Gradients, or are there domains where the smoother gradient modulation of Inverting Gradients is necessary?
- **Open**: How does performance degrade as $K$ (number of discrete actions) and the dimensionality of $X_k$ (continuous parameters per action) scale?
- **Open**: Can P-DQN's framework be extended with distributional Q-learning or risk-sensitive objectives to handle uncertainty in the continuous parameter estimates?
- **Open**: The paper notes that P-DQN can leverage human demonstration data for pre-training. What is the empirical effect of pre-training on convergence speed and final policy quality?

---

### Integration:

* **Problem:** P-DQN addresses the structural mismatch between existing DRL methods and hybrid action spaces by proposing an architecture that natively handles the $(k, x_k)$ hierarchy. For research in RL-based decision-making under structured action constraints — such as financial portfolio management where the agent must choose *which* instrument to trade (discrete) and *how much* to allocate (continuous) — P-DQN provides a template for preserving action-space semantics rather than flattening them. The demonstrated ability to use human data for pre-training is particularly relevant for domains where expert demonstrations are abundant but expensive to replicate via pure exploration.

* **Limitation:** The most significant limitation for broader application is the **lack of convergence guarantees** for the deterministic policy approximation (Eq. 3.2) — the framework relies on the two-timescale trick and empirical validation rather than formal proofs of optimality. Additionally, the **reward engineering dependency** across all three domains suggests that P-DQN's sample efficiency may be partially attributable to carefully shaped dense rewards rather than the architecture alone. For risk-sensitive applications, the absence of any mechanism for *uncertainty quantification* over continuous parameters — the policy network outputs a point estimate, not a distribution — limits the framework's applicability to settings where parameter uncertainty has asymmetric consequences.
