[[Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space.pdf]]

# **Abstract:**

The paper proposes a **hybrid actor-critic architecture** for reinforcement learning in **parameterized action spaces** — action spaces where the agent must simultaneously select a discrete action and specify continuous parameters for that action. The architecture decomposes this structured space into multiple **parallel sub-actor networks**, each handling a simpler action subspace, supervised by a single **critic network** that estimates the state-value function $V(s)$.

- The paper presents **Hybrid Proximal Policy Optimization (H-PPO)**, a concrete instantiation of the hybrid actor-critic architecture that uses PPO's clipped surrogate objective independently on the discrete and continuous sub-actors.
- While focused on parameterized action spaces, the architecture generalizes to **hierarchical action spaces** with tree-structured multi-layer classifications.
- Empirical evaluation on four parameterized-action environments demonstrates that H-PPO outperforms P-DQN, DDPG, and DQN baselines in success rate, convergence speed, and variance.

- *Context*:
	- The core challenge the paper addresses is that standard RL algorithms are designed for either *purely discrete* or *purely continuous* action spaces. Parameterized action spaces are neither — they require the agent to make a categorical choice (which action type?) and then a regression decision (what parameters for that action?). Naively flattening this structure (discretizing the continuous part or relaxing the discrete part into continuous space) either loses fine-grained control or inflates the effective dimensionality of the problem.
	- The paper's architectural solution — parallel sub-actors sharing a state-value critic — sidesteps the **over-parameterization problem** that plagues action-value critics in hybrid spaces. When $Q(s, a, x_{a_1}, \ldots, x_{a_k})$ must take parameters for *all* discrete actions as input, it processes irrelevant noise from non-selected actions. A state-value function $V(s)$ avoids this entirely by conditioning only on the state.

---

# 1. Introduction

The paper opens by situating RL's recent successes (Atari via DQN, Go via AlphaGo, robotic control) within the context of their action space assumptions. Most algorithms assume either a finite discrete set or a single continuous manifold.

- **Parameterized action spaces** break this assumption. A parameterized action is a discrete action paired with a real-valued parameter vector. The agent selects an action $a$ from a finite set $A_d$ and simultaneously selects a parameter $x \in \mathcal{X}_a$ specific to that action.
	- Example: In Half Field Offense (HFO), the agent selects **Kick** (discrete) with parameters *power* and *direction* (continuous).
	- Parameterized actions naturally arise in robotics, where **meta-actions** define high-level behavioral categories and fine-grained parameters control execution.

- *Context*:
	- The distinction between *flat* and *structured* action spaces is critical. A flat discrete space treats every action as an independent, unrelated choice. A parameterized action space imposes a two-level hierarchy: the discrete choice defines a *category* of behavior, and the continuous parameters define *how* that behavior is executed. This structure is ubiquitous in real-world control — consider a self-driving car choosing between ACCELERATE, BRAKE, and TURN (discrete), each with continuous intensity or angle parameters.

- The paper extends the concept beyond two-level parameterized spaces to **general hierarchical action spaces** with tree structures of arbitrary depth. Each internal node is a discrete selection, and leaf nodes may be either discrete or continuous.
	- OpenAI Five on Dota 2 is cited as an example of manually constructing a hierarchical taxonomy to reduce an intractably large action space into a tractable tree of small selections.

- *Context*:
	- The tree-structured action space is a powerful abstraction. Instead of choosing from millions of primitive actions, the agent makes a sequence of small decisions along a path in the tree. Each node has only a handful of branches. The parameterized action space is a special case: a tree with one discrete layer followed by one continuous layer.

- The paper's contribution: a **hybrid actor-critic architecture** with parallel sub-actor networks (one per action-selection sub-problem) and a single global critic that computes advantage estimates for all sub-actors. The specific instantiation using PPO is called **H-PPO**.

---

# 2. Related Work

The paper surveys existing approaches to parameterized action spaces, identifying their fundamental limitations.

## 2.1 RL Methods for Discrete Action Space and Continuous Action Space

Background review of the two standard RL paradigms:

**Discrete action spaces** — Q-learning and its deep variants:

$$
Q(s, a) = \mathbb{E}_{r_t, s_{t+1}}\left[r_t + \gamma \max_{a' \in A} Q(s_{t+1}, a') \;\middle|\; s_t = s,\, a_t = a\right] \tag{1}
$$

- DQN approximates $Q$ with a deep neural network. Variants include asynchronous DQN, double DQN, and dueling DQN.

**Stochastic policy gradient** — optimizes a parameterized stochastic policy $\pi_\theta$:

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s,a}\left[\nabla_\theta \log \pi_\theta(a \mid s)\, Q^{\pi_\theta}(s, a)\right] \tag{2}
$$

- The advantage variant replaces $Q$ with the advantage function $A^{\pi_\theta}(s, a)$ to reduce variance:

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s,a}\left[\nabla_\theta \log \pi_\theta(a \mid s)\, A^{\pi_\theta}(s, a)\right] \tag{3}
$$

- *Context*:
	- The advantage function $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ measures how much better action $a$ is compared to the average action under policy $\pi$ at state $s$. Using $A$ instead of $Q$ as the baseline in the policy gradient reduces variance without introducing bias, because $V(s)$ does not depend on the action.

**Deterministic policy gradient** — for continuous action spaces:

$$
\nabla_\theta J(\mu_\theta) = \mathbb{E}_s\left[\nabla_\theta \mu_\theta(s)\, \nabla_a Q^{\mu_\theta}(s, a)\big|_{a = \mu_\theta(s)}\right] \tag{4}
$$

- DDPG extends this with deep networks, experience replay, and target networks. TRPO and PPO improve optimization stability through constrained or clipped updates.

## 2.2 RL Methods for Parameterized Action Space

The paper identifies three prior approaches and their shortcomings:

1. **Discretization** (Sherstov and Stone, 2005): Discretize the continuous parameters into a finite grid. This destroys fine-grained control and produces exponentially large action spaces as parameter dimensionality grows.

2. **Continuous relaxation** (Hausknecht and Stone, 2016): Treat both discrete selection and continuous parameters as outputs of a single DDPG actor. The discrete action is chosen as the one with the maximum output value. This flattens the hierarchical structure and may significantly increase action space complexity.

3. **Q-PAMDP** (Masson et al., 2016): Alternately learn discrete action selection with Q-learning and update continuous parameter policies with policy search. Decouples the two sub-problems but does not enable joint optimization.

4. **Hierarchical conditioning** (Wei et al., 2018): Conditions the parameter policy on the discrete action policy, trained with TRPO and Stochastic Value Gradient. The authors themselves report instability from **joint-learning** between the two policies.

5. **P-DQN** (Xiong et al., 2018): A DQN-DDPG hybrid. One network selects continuous parameters for all discrete actions; another network takes the state and chosen parameters as input and outputs Q-values for each discrete action. The parameter network is updated to maximize the *sum* of Q-values across all actions, which may improve the sum while decreasing the Q-value of the actually-selected action.

- *Context*:
	- P-DQN's flaw is subtle but important. The parameter network $x_\theta(s)$ outputs parameters for *every* discrete action simultaneously, and the gradient update maximizes $\sum_a Q(s, a, x_a)$. This means the gradient pushes the parameters to be good *on average across all discrete actions*, not specifically good for the action that will actually be selected. If improving parameters for action $a_1$ degrades them for the currently-selected action $a_2$, the overall sum may still increase — but the agent's actual behavior worsens.

---

# 3. Methodologies

The paper introduces the formal framework of PAMDPs and the hybrid actor-critic architecture.

The **parameterized action space** is defined as follows. Discrete actions are drawn from a finite set $A_d = \{a_1, a_2, \ldots, a_k\}$. Each $a \in A_d$ has a continuous parameter space $\mathcal{X}_a \subseteq \mathbb{R}^{m_a}$. A complete action is a tuple $(a, x)$ where $a \in A_d$ and $x \in \mathcal{X}_a$:

$$
\mathcal{A} = \bigcup_{a \in A_d} \{(a, x) \mid x \in \mathcal{X}_a\} \tag{5}
$$

- *Context*:
	- Each discrete action $a_i$ may have a parameter space of *different dimensionality* $m_{a_i}$. A KICK action might require 2 parameters (power, direction), while a TURN action requires only 1 (angle). This heterogeneity is precisely what makes parameterized action spaces difficult — the agent must select from a union of continuous manifolds of different dimensions.

An MDP with this action space is a **Parameterized-action Markov Decision Process (PAMDP)**.

### Hybrid Actor-Critic Architecture

The proposed architecture (Figure 3 in the paper) consists of three components:

1. **State Encoding Network**: A shared front-end that extracts latent state representations from raw state input. Both sub-actor networks share this encoder.

2. **Discrete Actor Network**: Learns a stochastic policy $\pi_{\theta_d}$ to select the discrete action $a$.

3. **Continuous Actor Network**: Learns a stochastic policy $\pi_{\theta_c}$ to output continuous parameters $x_{a_1}, x_{a_2}, \ldots, x_{a_k}$ for all discrete actions. The complete action is $(a, x_a)$ — the selected discrete action paired with its corresponding parameter vector.

4. **Critic Network**: Estimates the **state-value function** $V(s)$, not the action-value function $Q(s, a, x)$.

- *Context*:
	- The choice of $V(s)$ over $Q(s, a, x)$ as the critic is the key architectural insight. The continuous actor outputs parameters for *all* discrete actions simultaneously (because parameter dimensions differ across actions, you cannot conditionally output parameters for only the selected action without knowing which action is selected first). If $Q$ were used as the critic, it would need to take all these parameters as input — leading to the over-parameterization problem.

### The Over-Parameterization Problem

If an action-value critic is used, it takes the form $Q(s, a, x_{a_1}, x_{a_2}, \ldots, x_{a_k})$. But the true Q-value depends only on the parameter of the selected action:

$$
Q(s, a, x_{a_1}, x_{a_2}, \ldots, x_{a_k}) = Q(s, a, x_a) \tag{6}
$$

- The critic must learn to *ignore* the parameters for all non-selected actions. In practice, these irrelevant inputs introduce noise into the value estimate, degrading learning stability.
- The state-value function $V(s)$ bypasses this entirely — it conditions only on the state and does not take any action or parameter as input.

- *Context*:
	- Equation 6 formalizes why concatenating all parameters into the critic input is wasteful and harmful. The network has $\sum_i m_{a_i}$ extra input dimensions that are pure noise for any given action selection. The network must learn an internal masking function conditioned on $a$ — an additional representational burden that competes with learning the actual value landscape. By using $V(s)$, the architecture eliminates this burden entirely.

### Advantage Estimation

The critic provides a **variance-reduced advantage function estimator** $\hat{A}_t$, computed from a trajectory segment of length $T$:

$$
\hat{A}_t = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t-1} r_{T-1} + \gamma^{T-t} V(s_T) \tag{7}
$$

where $t \in [0, T]$ is the timestep index within the segment, and $T$ is much smaller than the episode length.

- *Context*:
	- This is the $k$-step return advantage estimator from Mnih et al. (2016). It bootstraps after $T$ steps using $V(s_T)$ rather than waiting for the full episode return. The parameter $T$ controls the bias-variance tradeoff: small $T$ gives low variance but high bias (heavily reliant on $V(s_T)$'s accuracy); large $T$ gives lower bias but higher variance (accumulates stochastic reward signals). This same advantage estimate is shared by both the discrete and continuous sub-actors — the critic does not differentiate between them.

## 3.1 Hybrid Proximal Policy Optimization

**H-PPO** instantiates the hybrid actor-critic architecture using PPO as the optimization method for both sub-actors.

The PPO **clipped surrogate objective** constrains policy updates to prevent destructively large steps:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\,\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1 - \epsilon,\, 1 + \epsilon)\,\hat{A}_t\right)\right] \tag{8}
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio and $\epsilon$ is a clipping hyperparameter.

- *Context*:
	- The clipping mechanism serves as a soft trust region. If the ratio $r_t(\theta)$ drifts too far from 1 (meaning the new policy assigns very different probability to action $a_t$ than the old policy did), the objective is clipped so that the gradient vanishes. This prevents the policy from changing too rapidly in a single update, which is a common source of catastrophic performance collapse in policy gradient methods. **See MATHEMATICS §5** for the full derivation.

The **discrete sub-actor** produces a policy $\pi_{\theta_d}$ by outputting $k$ logits $f_{a_1}, \ldots, f_{a_k}$ and sampling from $\text{softmax}(f)$. The **continuous sub-actor** produces a policy $\pi_{\theta_c}$ by outputting the mean and variance of a Gaussian distribution for each parameter.

The two sub-actors are updated with **separate clipped objectives**:

**Discrete objective:**

$$
L^{\text{CLIP}}_d(\theta_d) = \hat{\mathbb{E}}_t\left[\min\left(r^d_t(\theta_d)\,\hat{A}_t,\; \text{clip}(r^d_t(\theta_d),\, 1 - \epsilon,\, 1 + \epsilon)\,\hat{A}_t\right)\right] \tag{9}
$$

**Continuous objective:**

$$
L^{\text{CLIP}}_c(\theta_c) = \hat{\mathbb{E}}_t\left[\min\left(r^c_t(\theta_c)\,\hat{A}_t,\; \text{clip}(r^c_t(\theta_c),\, 1 - \epsilon,\, 1 + \epsilon)\,\hat{A}_t\right)\right] \tag{10}
$$

where $r^d_t(\theta_d) = \frac{\pi_{\theta_d}(a \mid s_t)}{\pi_{\theta_{d(\text{old})}}(a \mid s_t)}$ and $r^c_t(\theta_c) = \frac{\pi_{\theta_c}(x_a \mid s_t)}{\pi_{\theta_{c(\text{old})}}(x_a \mid s_t)}$.

- The probability ratios consider only their respective policies: $r^d_t$ uses only the discrete policy, $r^c_t$ uses only the continuous policy.
- The two policies are treated as **separate distributions**, not a joint distribution, during optimization. Their objectives are not explicitly conditioned on each other.

- *Context*:
	- This independence assumption is the architectural gamble of H-PPO. By treating $\pi_{\theta_d}$ and $\pi_{\theta_c}$ as independent distributions during updates, the method avoids the joint-learning instability reported by Wei et al. (2018). The coordination between the two actors comes implicitly from two sources: (1) they share the state encoding network's front layers, so gradient updates from one actor affect the shared representation, and (2) they share the same advantage estimate $\hat{A}_t$ from the global critic, so both actors are pushed toward actions that improve the same value signal. The question is whether this *implicit* coordination suffices — the paper's experiments suggest it does for the tested environments, but the absence of explicit conditioning may fail in environments requiring tight action-parameter synchronization.

## 3.2 Hybrid Actor-Critic Architecture for General Hierarchical Action Space

The architecture extends to general tree-structured action spaces (Figure 4 in the paper):

- Each action-selection sub-problem at each node gets its own sub-actor network.
- Internal nodes correspond to **discrete actor networks**; leaf nodes may be either discrete or continuous.
- All sub-actors share the state encoding layers and the same global critic.
- Sub-actors are updated independently using the chosen optimization method (e.g., PPO).

- *Context*:
	- This generalization is theoretically significant but empirically untested in the paper. A deep tree with $L$ layers would require $O(B^L)$ sub-actor networks (where $B$ is the branching factor), though only the sub-actors along the selected path need to produce outputs at inference time. The scalability of this approach — both computationally and in terms of gradient interference through the shared state encoder — is an open question.

---

# 4. Experiments

## 4.1 Environments

Four environments with parameterized action spaces are used, each with a defined "winning state":

1. **Catching Point**: The agent catches a target point. Actions: MOVE($\text{direction}_M$) and CATCH. The agent has at most 10 CATCH attempts. Episode ends on catch success, exhaustion of attempts, or timeout.

2. **Moving**: The agent moves to a target area and stops in it. Actions: ACCEL($\text{power}_A$), TURN($\text{direction}_T$), BRAKE. Episode ends when the agent stops in the target, moves out of the field, or timeout.

3. **Chase and Attack**: The agent chases a rule-based runner and attacks it. Actions: RUSH($\text{direction}_R$), ATTACK($\text{direction}_A$). The runner has 3 lives. Episode ends when all lives are lost or timeout.

4. **Half Field Football**: The agent scores a goal in a half football field with no goalie. Actions: DASH($\text{power}_D$, $\text{direction}_D$), TURN($\text{direction}_T$), KICK($\text{power}_K$, $\text{direction}_K$). Episode ends on goal, ball out of bounds, or timeout.

- *Context*:
	- These environments span a range of complexity. Catching Point is relatively simple (binary action choice with one continuous parameter for movement). Half Field Football is the most complex, with three discrete actions carrying different numbers of continuous parameters (DASH: 2, TURN: 1, KICK: 2). This heterogeneity in parameter dimensionality across actions is precisely the scenario where over-parameterization hurts Q-based critics the most.

## 4.2 Experiment Settings and Results

**Network architecture:** All algorithms use networks with hidden layers of size $(256, 256, 128, 64)$. DDPG and DQN use replay buffers of size 10,000 with batch size 32. For DQN, the continuous spaces are discretized: 16 actions (Catching Point), 23 (Moving), 30 (Chase and Attack), 104 (Half Field Football — 8 direction values $\times$ 6 power values per parameter).

**DDPG exclusion:** DDPG's performance was far worse than reported in the original parameterized-action paper (Hausknecht and Stone, 2016), with extreme variance and failure to learn reasonable policies — consistent with issues reported by Wei et al. (2018). DDPG results are excluded from comparisons.

**Results (Table 1):**

| Environment | Algorithm | Success Rate | SD | Mean Reward |
|---|---|---|---|---|
| Catching Point | DQN | 6.13% | $\pm$8.21% | 0.796 |
| Catching Point | P-DQN | 82.52% | $\pm$11.60% | 4.977 |
| Catching Point | **H-PPO** | **96.32%** | $\pm$4.82% | 4.790 |
| Moving | DQN | 0.00% | $\pm$0.00% | -0.415 |
| Moving | P-DQN | 1.56% | $\pm$2.78% | 0.173 |
| Moving | **H-PPO** | **90.45%** | $\pm$6.75% | 8.955 |
| Chase and Attack | DQN | 99.91% | $\pm$0.74% | 5.664 |
| Chase and Attack | P-DQN | 99.85% | $\pm$0.84% | 5.589 |
| Chase and Attack | **H-PPO** | **99.98%** | $\pm$0.30% | 5.393 |
| Half Field Football | DQN | 0.00% | $\pm$0.00% | 0.000 |
| Half Field Football | P-DQN | 76.31% | $\pm$16.81% | 8.762 |
| Half Field Football | **H-PPO** | **95.39%** | $\pm$4.81% | 9.849 |

- H-PPO achieves the highest success rate on all four environments.
- H-PPO has the lowest standard deviation on all four environments, indicating the most stable learning.
- In **Moving**, P-DQN achieves only 1.56% success rate versus H-PPO's 90.45% — a near-total failure of the Q-based approach in this environment. This is the strongest evidence for the state-value critic's advantage.
- In **Catching Point**, H-PPO achieves higher success rate (96.32% vs 82.52%) but *lower* mean reward (4.790 vs 4.977) than P-DQN. This suggests H-PPO's policy is more conservative — it reliably completes the task but does not pursue high-reward trajectories that carry higher failure risk.

- *Context*:
	- The Catching Point success-rate vs. reward discrepancy is informative. P-DQN occasionally finds high-reward strategies (fewer moves, catching early) but fails more often. H-PPO converges to a robust policy that catches the target reliably, even if it takes more steps (lower reward per success). This is a hallmark of the PPO objective's conservatism — the clipping mechanism prevents the policy from chasing high-reward but brittle strategies.
	- The Moving environment result is the paper's strongest empirical argument. P-DQN's 1.56% success rate suggests the over-parameterization problem is catastrophic in environments requiring precise coordination between ACCEL power and TURN direction. The Q-critic cannot disentangle which parameters matter for the selected action, leading to noisy value estimates that prevent effective learning.

- The learning curves (Figure 6) show H-PPO converges faster and with less variance than P-DQN in all environments.
- Figure 7 illustrates a Half Field Football episode: the H-PPO agent executes TURN (to face the ball), then DASH (to approach it), then KICK (toward the goal) — demonstrating coordinated discrete-action selection and continuous-parameter specification.

---

# 5. Conclusion and Future Work

The paper concludes that the hybrid actor-critic architecture with parallel sub-actors and a global state-value critic is an effective framework for parameterized action spaces. H-PPO achieves stable learning and outperforms prior methods across all tested environments.

- The generalization to multi-layer hierarchical action spaces is presented as a natural extension but requires further empirical validation.
- Future work focuses on testing the architecture in general hierarchical action spaces with deeper tree structures.

- *Context*:
	- The paper's conclusion is modest relative to the architectural contribution. The hybrid actor-critic framework is not just a technique for parameterized actions — it is a design principle for decomposing any structured action space into parallel, independently-optimized components unified by a shared critic signal. The unresolved question is whether this decomposition scales: as the tree deepens, the number of sub-actors grows, and the shared state encoder must serve increasingly diverse representational needs.

---

# MATHEMATICS

The mathematical framework of this paper proceeds from standard RL value estimation through the formal definition of parameterized action spaces to the construction of the H-PPO algorithm. The derivation chain establishes why conventional action-value critics fail in hybrid spaces and how the state-value critic, combined with independent clipped objectives for each sub-actor, resolves this structural problem.

### 1. Bellman Equation for Discrete Action Spaces

The foundational recursive characterization of the action-value function under Q-learning:

**Bellman Equation ($Q$):**

$$
Q(s, a) = \mathbb{E}_{r_t, s_{t+1}}\left[r_t + \gamma \max_{a' \in A} Q(s_{t+1}, a') \;\middle|\; s_t = s,\, a_t = a\right] \tag{1}
$$

where $s$ is the current state, $a$ is the action taken, $r_t$ is the immediate reward, $\gamma \in [0,1)$ is the discount factor, and the expectation is over the stochastic reward and transition dynamics. The $\max$ operator selects the action with the highest estimated value at the next state, making this the *optimality* form of the Bellman equation.

### 2. Stochastic Policy Gradient

For stochastic policies $\pi_\theta$ parameterized by $\theta$, the policy gradient theorem provides the gradient of the expected cumulative reward $J(\pi_\theta)$:

**Stochastic Policy Gradient ($\nabla_\theta J$):**

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s,a}\left[\nabla_\theta \log \pi_\theta(a \mid s)\, Q^{\pi_\theta}(s, a)\right] \tag{2}
$$

where $\pi_\theta(a \mid s)$ is the probability of taking action $a$ in state $s$, and $Q^{\pi_\theta}(s, a)$ is the action-value function under policy $\pi_\theta$. The score function $\nabla_\theta \log \pi_\theta(a \mid s)$ acts as a direction in parameter space, weighted by the value of the action taken.

### 3. Advantage-Based Policy Gradient

The variance of the estimator in §2 can be reduced by replacing $Q^{\pi_\theta}$ with the advantage function $A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)$:

**Advantage Policy Gradient ($\nabla_\theta J$ with advantage):**

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s,a}\left[\nabla_\theta \log \pi_\theta(a \mid s)\, A^{\pi_\theta}(s, a)\right] \tag{3}
$$

This substitution is valid because $V^{\pi_\theta}(s)$ is independent of $a$ and thus vanishes under the expectation $\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a \mid s)\, V(s)] = V(s)\,\nabla_\theta \sum_a \pi_\theta(a \mid s) = 0$. The advantage function centers the gradient signal: positive advantages reinforce actions, negative advantages suppress them.

### 4. Deterministic Policy Gradient

For continuous action spaces, the deterministic policy gradient theorem provides the gradient for a deterministic policy $\mu_\theta$:

**Deterministic Policy Gradient ($\nabla_\theta J$ for $\mu_\theta$):**

$$
\nabla_\theta J(\mu_\theta) = \mathbb{E}_s\left[\nabla_\theta \mu_\theta(s)\, \nabla_a Q^{\mu_\theta}(s, a)\big|_{a = \mu_\theta(s)}\right] \tag{4}
$$

where $\mu_\theta(s)$ is the deterministic action and $\nabla_a Q$ is the gradient of the Q-function with respect to the action, evaluated at the policy's output. This is the basis for DDPG.

### 5. Parameterized Action Space Definition

The formal definition of the structured action manifold:

**Parameterized Action Space ($\mathcal{A}$):**

$$
\mathcal{A} = \bigcup_{a \in A_d} \{(a, x) \mid x \in \mathcal{X}_a\} \tag{5}
$$

where $A_d = \{a_1, a_2, \ldots, a_k\}$ is a finite set of $k$ discrete actions, $\mathcal{X}_a \subseteq \mathbb{R}^{m_a}$ is the continuous parameter space for action $a$, and $m_a$ is the dimensionality of the parameter space for action $a$ (which may differ across actions). A complete action is a tuple $(a, x)$ with $a \in A_d$ and $x \in \mathcal{X}_a$.

### 6. Over-Parameterization Identity

The structural identity that exposes the flaw in action-value critics for PAMDPs:

**Over-Parameterization Identity:**

$$
Q(s, a, x_{a_1}, x_{a_2}, \ldots, x_{a_k}) = Q(s, a, x_a) \tag{6}
$$

The true Q-value of the executed action $(a, x_a)$ depends only on the parameter $x_a$ for the selected action $a$. Parameters $x_{a'}$ for all $a' \neq a$ are irrelevant — the executed action is $(a, x_a)$ regardless of what parameters were generated for other actions. Yet the critic network must accept all parameters as input (because parameter dimensionalities differ, preventing conditional input). The network must therefore learn to mask out $k - 1$ irrelevant parameter vectors — an additional representational burden that injects noise into the value estimate.

### 7. Variance-Reduced Advantage Estimator

The advantage function is estimated using a $k$-step return computed from a trajectory segment of length $T$:

**Advantage Estimator ($\hat{A}_t$):**

$$
\hat{A}_t = -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{T-t-1} r_{T-1} + \gamma^{T-t} V(s_T) \tag{7}
$$

where $V(s_t)$ is the critic's value estimate at the current timestep, $r_t, r_{t+1}, \ldots, r_{T-1}$ are the observed rewards along the trajectory segment, $\gamma$ is the discount factor, and $V(s_T)$ is the bootstrap value estimate at the end of the segment. The estimator can be written compactly as $\hat{A}_t = \left(\sum_{i=0}^{T-t-1} \gamma^i r_{t+i}\right) + \gamma^{T-t} V(s_T) - V(s_t)$, which is the $k$-step return minus the current value baseline.

### 8. Clipped Surrogate Objective (PPO)

The core PPO objective used by H-PPO:

**Clipped Surrogate Objective ($L^{\text{CLIP}}$):**

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\,\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1 - \epsilon,\, 1 + \epsilon)\,\hat{A}_t\right)\right] \tag{8}
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio between the new and old policies, $\hat{A}_t$ is the advantage estimator from §7, and $\epsilon$ is the clipping hyperparameter. The $\min$ operator ensures that the objective is pessimistic: when the advantage is positive and the ratio exceeds $1 + \epsilon$, clipping prevents the objective from further increasing; when the advantage is negative and the ratio falls below $1 - \epsilon$, clipping similarly caps the penalty. This implements a soft trust region without the computational overhead of TRPO's KL constraint.

### 9. Discrete Clipped Objective

The clipped objective applied to the discrete sub-actor of H-PPO:

**Discrete Clipped Objective ($L^{\text{CLIP}}_d$):**

$$
L^{\text{CLIP}}_d(\theta_d) = \hat{\mathbb{E}}_t\left[\min\left(r^d_t(\theta_d)\,\hat{A}_t,\; \text{clip}(r^d_t(\theta_d),\, 1 - \epsilon,\, 1 + \epsilon)\,\hat{A}_t\right)\right] \tag{9}
$$

where $r^d_t(\theta_d) = \frac{\pi_{\theta_d}(a \mid s_t)}{\pi_{\theta_{d(\text{old})}}(a \mid s_t)}$ is the probability ratio using only the discrete policy. The advantage $\hat{A}_t$ is the same estimator computed by the shared critic (§7). The discrete policy $\pi_{\theta_d}$ is produced by applying softmax to the logit outputs of the discrete actor network.

### 10. Continuous Clipped Objective

The clipped objective applied to the continuous sub-actor of H-PPO:

**Continuous Clipped Objective ($L^{\text{CLIP}}_c$):**

$$
L^{\text{CLIP}}_c(\theta_c) = \hat{\mathbb{E}}_t\left[\min\left(r^c_t(\theta_c)\,\hat{A}_t,\; \text{clip}(r^c_t(\theta_c),\, 1 - \epsilon,\, 1 + \epsilon)\,\hat{A}_t\right)\right] \tag{10}
$$

where $r^c_t(\theta_c) = \frac{\pi_{\theta_c}(x_a \mid s_t)}{\pi_{\theta_{c(\text{old})}}(x_a \mid s_t)}$ is the probability ratio using only the continuous policy. The continuous policy $\pi_{\theta_c}$ is a Gaussian distribution whose mean and variance are output by the continuous actor network.

Equations 9 and 10 are structurally identical to Equation 8 but applied to different components of the joint action. The probability ratios $r^d_t$ and $r^c_t$ are computed *independently* — neither conditions on the other. This decoupling means the gradient of $L^{\text{CLIP}}_d$ does not flow through $\pi_{\theta_c}$, and vice versa. Coordination between the two policies emerges only through (1) the shared state encoding network and (2) the shared advantage signal $\hat{A}_t$.

---

### **1. Problem:**

The paper addresses the challenge of applying reinforcement learning to **parameterized action spaces** — action spaces where the agent must simultaneously select a discrete meta-action and specify continuous parameters for its execution. Standard RL algorithms handle either discrete or continuous action spaces, not their structured union. Prior approaches either flatten the structure (losing fine-grained control or inflating complexity) or suffer from instability in joint optimization of discrete and continuous policies. The specific gap is the absence of a stable, general-purpose actor-critic framework that respects the hierarchical structure of parameterized action spaces without requiring explicit conditional dependencies between the discrete and continuous components.

### **2. Setup:**

The method assumes a **Parameterized-action MDP (PAMDP)** with a finite set of $k$ discrete meta-actions $A_d = \{a_1, \ldots, a_k\}$, each associated with a real-valued continuous parameter space $\mathcal{X}_a \subseteq \mathbb{R}^{m_a}$. The environment is Markovian. The method uses an actor-critic architecture with PPO-style clipped surrogate updates, parallel sub-actor networks sharing a state encoding front-end, and a single state-value critic $V(s)$. The approach is on-policy: data is collected by running the current policy for $T$ timesteps before each update.

### **3. Key Idea:**

The core contribution is the **hybrid actor-critic architecture** that decomposes the parameterized action space into parallel sub-actors (discrete and continuous) supervised by a single state-value critic. By using $V(s)$ instead of $Q(s, a, x)$, the architecture eliminates the over-parameterization problem where irrelevant continuous parameters for non-selected actions inject noise into value estimation. The sub-actors are updated independently using PPO's clipped objective, with coordination emerging implicitly through shared state representations and shared advantage signals.

### **4. Assumptions:**

**Explicit:**
- The discrete action set $A_d$ is finite and pre-defined.
- Each discrete action's parameter space $\mathcal{X}_a$ is continuous and real-valued.
- The environment is a PAMDP (Markovian dynamics, parameterized action structure).
- Sub-actors are updatable by any advantage-based actor-critic method.

**Implicit:**
- The state-value function $V(s)$ is a sufficient critic for coordinating heterogeneous sub-actors — no action-conditioned value signal is needed.
- Independent optimization of discrete and continuous policies (treating them as separate distributions) reliably converges to coordinated behavior.
- The shared state encoding network does not suffer from gradient conflict between the discrete and continuous actor heads during backpropagation.
- The same $\epsilon$ clipping hyperparameter is appropriate for both discrete (categorical) and continuous (Gaussian) policy distributions.
- The action-parameter structure is known a priori and fixed — the method does not discover or adapt the action hierarchy.

### **5. Limitation:**

- **Hierarchical depth:** Experiments are confined to two-layer parameterized spaces. The theoretical extension to deeper hierarchies is projected but untested, leaving open questions about scalability and gradient interference.
- **Coordination capacity:** The lack of explicit conditioning between the discrete and continuous objectives may fail in environments requiring tight synchronization between action selection and parameterization — for instance, when the optimal parameter range differs drastically across discrete actions.
- **Shared representation:** Reliance on a single state encoding network assumes a universal representation serves both discrete and continuous policy heads equally well. In complex environments, these may require fundamentally different features.
- **DDPG instability:** DDPG could not serve as a consistent baseline due to extreme variance, limiting the comparative scope. This is not a limitation of H-PPO per se, but it constrains the empirical evaluation.
- **On-policy sample efficiency:** PPO is on-policy, requiring fresh trajectory data for each update. In environments where data collection is expensive, the sample efficiency cost relative to off-policy methods like P-DQN is unaddressed.
- **Scale of discrete action set:** All test environments have small discrete action sets ($k \le 3$). Behavior at scale — $k$ in the hundreds or thousands — is unknown.

### **6. Relevance & Open Questions:**

The hybrid actor-critic architecture is directly relevant to any RL application where actions have natural hierarchical structure — robotics (meta-actions with continuous parameters), strategy games (action type + target + parameters), and financial portfolio management (discrete allocation decisions with continuous position sizes). The decomposition principle — parallel sub-actors, shared critic, independent updates — is a reusable design pattern.

**Open questions:**
1. How does the architecture scale as the action tree depth reaches 3+ layers, and what is the computational overhead of maintaining $O(B^L)$ sub-actor networks?
2. Should the clipping parameter $\epsilon$ be decoupled between discrete and continuous objectives, given their different distributional families?
3. Can the architecture handle leaf nodes that are themselves a mixture of discrete and continuous selections at the same hierarchical level?
4. How does performance degrade when the discrete action set scales to hundreds or thousands of entries?
5. What explicit coordination mechanisms (e.g., conditioning the continuous policy on the selected discrete action) could be added without reintroducing joint-learning instability?
6. How does this framework interact with off-policy learning — can the parallel sub-actor decomposition be combined with experience replay?

---

### Integration:

* **Problem:** The hybrid actor-critic architecture addresses a fundamental structural challenge in RL: navigating action spaces that mix discrete and continuous decisions. For research in structured-action RL — including portfolio management (discrete asset selection + continuous allocation), robotic control (discrete skill selection + continuous parameterization), and hierarchical game-playing agents — this paper provides a clean architectural template. The key principle is that *decomposing the action space into parallel, independently-optimized sub-actors* unified by a *state-value critic* avoids the over-parameterization noise that degrades action-value methods. The state-value critic sidesteps the need to process irrelevant parameters, while the shared advantage signal provides sufficient coordination without explicit conditional dependencies.

* **Limitation:** The most significant limitation for practical application is the untested scalability of the approach. Two-layer parameterized spaces with 2-3 discrete actions are far simpler than the hierarchical taxonomies found in real-world domains (thousands of discrete actions, deep tree structures, heterogeneous parameter spaces). The implicit coordination mechanism — shared state encoder + shared advantage — may be insufficient when the optimal continuous parameters are highly sensitive to the discrete action choice. For portfolio management applications, where the discrete decision (buy/sell/hold) fundamentally changes the semantics of the continuous parameter (position size), explicit conditioning between action type and parameter selection may be necessary. The on-policy sample efficiency cost of PPO also limits applicability in domains where environment interaction is expensive.
