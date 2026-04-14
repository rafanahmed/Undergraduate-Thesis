[[Deep Reinforcement Learning in Parameterized Action Space.pdf]]

# **Abstract:**

The paper extends Deep Deterministic Policy Gradients (DDPG) into **parameterized action spaces** — hybrid spaces where the agent must select a discrete action type and simultaneously specify continuous parameters for that action. No prior work had succeeded at applying deep neural networks to such structured action spaces. The domain is simulated RoboCup 2D soccer, which features four discrete action types (Dash, Turn, Tackle, Kick), each parameterized by one or two continuous variables.

- The paper documents a critical modification to DDPG: **bounding action-space gradients** via a technique called **Inverting Gradients**. Without this modification, the critic's gradients push continuous parameters toward ever-larger values, causing overflow beyond the environment's physical limits.
- The best learned agent scores goals with 100% reliability — surpassing the 96.2% reliability of the 2012 RoboCup world champion **Helios-Agent2D**, a hand-coded expert policy.
- The work represents a successful extension of deep reinforcement learning to the class of **Parameterized Action Space MDPs (PAMDPs)**.

- *Context*:
	- The significance of this paper lies in bridging two previously separate branches of DRL: methods for high-dimensional discrete spaces (DQN for Atari) and methods for low-dimensional continuous control (DDPG for locomotion). Real-world robotic and strategic domains often require *both* — a categorical decision ("what to do") coupled with continuous precision ("how to do it"). The parameterized action formulation captures this structure natively, avoiding the combinatorial explosion of discretizing continuous parameters or the loss of structure from flattening into a purely continuous space.

---

# 1. Introduction

The paper motivates the need for deep RL in parameterized action spaces by identifying a structural gap: standard DRL architectures cannot natively manage the hybrid discrete-continuous structure required by many real-world tasks. The RoboCup soccer domain exemplifies this — an agent must choose among qualitatively different action types and then specify exact physical parameters for execution.

- The core contribution is an extension of DDPG (Lillicrap et al., 2015) to PAMDPs.
- A key practical finding: **bounding action space gradients** is necessary for stable learning in bounded continuous action spaces. The proposed **Inverting Gradients** technique is not domain-specific and applies to any bounded-continuous action space.
- The agents learn entirely from scratch using a single reward function, acquiring a complete behavioural sequence: locate the ball, approach it, dribble toward the goal, and score.
- The learned agent is more reliable (though slower) than the hand-coded 2012 RoboCup champion.

- *Context*:
	- The phrase "learning from scratch" is important: the agent has no hand-coded subroutines for dribbling or shooting. The entire policy — from low-level motor commands to high-level tactical decisions — is a single monolithic neural network trained end-to-end. This contrasts with prior RoboCup agents like Brainstormers (Riedmiller et al., 2009) that used separate learned modules for individual skills.

---

# 2. Half Field Offense Domain

**Half Field Offense (HFO)** is a research abstraction of RoboCup 2D soccer that isolates the core decision-making challenge: scoring and defending goals. Full RoboCup games are lengthy, high-variance, and encumbered by rules (free kicks, offsides). HFO strips away these complications.

- HFO is naturally characterized as an **episodic multi-agent POMDP**: agents receive partial observations, act independently, and episodes have well-defined termination conditions.
- Episode initialization: the agent and ball are positioned randomly on the offensive half of the field.
- Episode termination: a goal is scored, the ball leaves the field, or 500 timesteps elapse.

- *Context*:
	- Although HFO is a POMDP, the paper treats the agent's 58-feature state representation as sufficient for approximately Markovian decision-making. This is a common approximation in applied RL — the state features are rich enough to capture most decision-relevant information, even though the full state of the simulator is not directly observed.

## 2.1 State Space

The agent uses a **low-level, egocentric viewpoint** encoded as 58 continuously-valued features. These features are derived from the **Helios-Agent2D** world model (Akiyama, 2010) and provide distances and angles to on-field objects.

- The most relevant features include:
	- Agent's position, velocity, orientation, and **stamina**
	- **Indicator if the agent is able to kick** ($I_{\text{kick}}$)
	- Angles and distances to: ball, goal, field corners, penalty-box corners, teammates, opponents

- *Context*:
	- The "able to kick" indicator is a discrete feature embedded in an otherwise continuous state space. It signals proximity to the ball — the agent can only execute a Kick action when close enough. This binary feature creates a natural phase transition in the policy: the agent must first learn approach behaviour (making $I_{\text{kick}} = 1$), then learn kicking behaviour.
	- The agent does not learn from raw pixel data. It relies on the Helios world model for feature extraction, which means the perceptual problem is solved externally. The learning challenge is purely in the decision-making domain.

## 2.2 Action Space

HFO features a **low-level, parameterized action space** with four mutually exclusive discrete actions, each accompanied by 1–2 continuous parameters:

- **Dash**$(p_1^{\text{dash}}, p_2^{\text{dash}})$: Moves in the indicated direction with scalar power in $[0, 100]$. Movement is faster forward than sideways or backwards.
- **Turn**$(p_3^{\text{turn}})$: Turns to the indicated direction.
- **Tackle**$(p_4^{\text{tackle}})$: Contests the ball by moving in the indicated direction. Only useful against opponents.
- **Kick**$(p_5^{\text{kick}}, p_6^{\text{kick}})$: Kicks the ball in the indicated direction with scalar power in $[0, 100]$.
- All direction parameters are bounded in $[-180, 180]$ degrees.

- *Context*:
	- The action space has latent structure that purely continuous formulations would ignore. Not all six continuous parameters are relevant at each timestep — only the parameters associated with the selected discrete action affect the environment. The agent must learn this association implicitly through the critic's gradients, since the paper does not explicitly indicate to the critic which discrete action was applied. This makes the learning problem harder but avoids architectural complexity.

## 2.3 Reward Signal

True rewards in HFO come from winning full games, but this signal is far too sparse for learning traction. The paper introduces a **hand-crafted, multi-component reward signal**:

$$
r_t = d_{t-1}(a, b) - d_t(a, b) + I_t^{\text{kick}} + 3\left(d_{t-1}(b, g) - d_t(b, g)\right) + 5 \cdot I_t^{\text{goal}}
$$

- **Move To Ball Reward**: Proportional to the reduction in distance between the agent $a$ and ball $b$. An additional reward of 1 is given the first time the agent is close enough to kick ($I_t^{\text{kick}}$).
- **Kick To Goal Reward**: Proportional to the reduction in distance between the ball $b$ and goal center $g$. Weighted by a factor of 3 to overcome the negative move-to-ball signal generated when the ball moves away from the agent after a kick.
- **Goal Reward**: A terminal bonus of 5 for scoring ($I_t^{\text{goal}}$).

- *Context*:
	- The reward engineering is a pragmatic necessity: acting randomly in the parameterized action space is exceedingly unlikely to produce even a single goal. Without the dense intermediate rewards, the agent would receive no gradient signal to guide early exploration. The weighting scheme (1:3:5) reflects a curriculum of priorities — approach first, kick toward goal second, score third.
	- The authors explicitly acknowledge this as a limitation. The dependence on hand-crafted reward components limits the framework's generality and introduces sensitivity to the specific weight choices. The paper flags exploration of large state spaces as an important future direction, citing Stadie et al. (2015) on exploration bonuses derived from dynamics models.

---

# 3. Background: Deep Reinforcement Learning

The paper reviews deep RL in continuous action spaces, following the notation of Lillicrap et al. (2015). The section establishes the **Actor/Critic architecture** that forms the foundation for the parameterized action space extension.

- **DQN** (Mnih et al., 2015) handles discrete action spaces by outputting Q-values for each discrete action. It does not extend to continuous actions because its output nodes are trained to estimate Q-values, not to produce continuous actions.
- The **Actor/Critic architecture** (Sutton & Barto, 1998) decouples value estimation from action selection using two networks:
	- **Actor** $\mu(s|\theta^\mu)$: Takes state $s$ as input, outputs continuous action $a$.
	- **Critic** $Q(s, a|\theta^Q)$: Takes state $s$ and action $a$ as input, outputs scalar Q-value.

- *Context*:
	- The separation of actor and critic is essential for continuous spaces. In discrete spaces, the Q-network can enumerate all actions and select the maximum. In continuous spaces, this enumeration is impossible — the actor serves as the implicit $\arg\max$ operator, proposing the action that the critic then evaluates.

The standard temporal difference update adapted to the neural network setting yields a critic loss:

$$
L_Q(s, a|\theta^Q) = \left(Q(s, a|\theta^Q) - \left(r + \gamma Q(s', \mu(s'|\theta^\mu)|\theta^Q)\right)\right)^2
$$

The actor's next-state action $a' = \mu(s'|\theta^\mu)$ replaces the intractable $\max_{a'} Q(s', a'|\theta^Q)$ in continuous spaces.

The actor is updated using the critic's **gradients with respect to actions** $\nabla_a Q(s, a|\theta^Q)$, which indicate directions in action space that increase the estimated Q-value:

$$
\nabla_{\theta^\mu} \mu(s) = \nabla_a Q(s, a|\theta^Q) \nabla_{\theta^\mu} \mu(s|\theta^\mu)
$$

- *Context*:
	- These gradients are not the standard gradients with respect to network parameters. They are gradients with respect to *inputs* — specifically, the action inputs of the critic. This technique was first used by NFQCA (Hafner & Riedmiller, 2011). The forward pass evaluates the actor's proposed action; the backward pass through the critic produces $\nabla_a Q$, which then flows backward through the actor, updating only the actor's parameters $\theta^\mu$. The critic's parameters $\theta^Q$ are held fixed during the actor update.

## 3.1 Stable Updates

The interdependence of actor and critic creates a fragile training loop: errors in the critic's Q-estimates produce bad actor updates, which in turn produce worse training data for the critic. Three stabilization techniques are employed:

- **Target-Q-Network** $Q'$: A replica of the critic that changes on a slower timescale, used to generate next-state targets. Introduced by Mnih et al. (2015).
- **Target-Actor-Network** $\mu'$: A slow-moving replica of the actor, preventing rapid policy shifts from destabilizing the critic's targets.
- **Replay Memory** $D$: A FIFO queue of the agent's latest experiences (typically one million transitions). Mini-batch updates from uniformly sampled experience reduce bias compared to sequential updates.

With these techniques, the critic loss and actor update become (**See MATHEMATICS §3**):

$$
L_Q(\theta^Q) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \sim D}\left[\left(Q(s_t, a_t) - \left(r_t + \gamma Q'(s_{t+1}, \mu'(s_{t+1}))\right)\right)^2\right]
$$

$$
\nabla_{\theta^\mu} \mu = \mathbb{E}_{s_t \sim D}\left[\nabla_a Q(s_t, a|\theta^Q) \nabla_{\theta^\mu} \mu(s_t)\big|_{a = \mu(s_t)}\right]
$$

The target networks are updated via **soft tracking** with factor $\tau \ll 1$:

$$
\theta^{Q'} = \tau\theta^Q + (1 - \tau)\theta^{Q'}, \quad \theta^{\mu'} = \tau\theta^\mu + (1 - \tau)\theta^{\mu'}
$$

- *Context*:
	- The soft update rule replaces the periodic hard copy used in DQN. Instead of copying all weights every $C$ steps (which introduces discontinuities in the target), the target network smoothly tracks the online network at each update. The parameter $\tau$ controls the tracking speed — the paper uses $\tau = 10^{-4}$, meaning the target network moves very slowly. This provides a more stable training signal at the cost of slower adaptation to policy improvements.

## 3.2 Network Architecture

Both the actor and critic use identical architectures:

- **Input**: 58 state features (the critic additionally receives 4 discrete actions + 6 continuous parameters)
- **Hidden layers**: Four fully connected layers of 1024-512-256-128 units
- **Activation**: ReLU with **negative slope** $10^{-2}$ (Leaky ReLU)
- **Weight initialization**: Gaussian with standard deviation $10^{-2}$
- **Output layers**: Two linear output layers — one for 4 discrete actions, one for 6 continuous parameters
- **Critic output**: Single scalar Q-value
- **Optimizer**: ADAM with learning rate $10^{-3}$ for both actor and critic
- **Target network tracking**: $\tau = 10^{-4}$

- *Context*:
	- The use of Leaky ReLU (negative slope $10^{-2}$) rather than standard ReLU prevents dead neurons — units that output zero for all inputs and can never recover. In a domain where the gradient signal from the critic is the sole driver of actor updates, dead neurons represent a permanent loss of representational capacity. The four-layer architecture is relatively deep for 2016-era RL and reflects the complexity of the mapping from 58-dimensional state to the 10-dimensional action output.

---

# 4. Parameterized Action Space Architecture

Following the formalism of Masson & Konidaris (2015), a **Parameterized Action Space MDP (PAMDP)** defines a set of discrete actions $A_d = \{a_1, a_2, \ldots, a_k\}$, where each discrete action $a \in A_d$ has $m_a$ continuous parameters $\{p_1^a, \ldots, p_{m_a}^a\} \in \mathbb{R}^{m_a}$. Actions are represented as tuples $(a, p_1^a, \ldots, p_{m_a}^a)$ and the overall action space is:

$$
A = \bigcup_{a \in A_d} (a, p_1^a, \ldots, p_{m_a}^a)
$$

For HFO, this expands to:

$$
A = (\text{Dash}, p_1^{\text{dash}}, p_2^{\text{dash}}) \cup (\text{Turn}, p_3^{\text{turn}}) \cup (\text{Tackle}, p_4^{\text{tackle}}) \cup (\text{Kick}, p_5^{\text{kick}}, p_6^{\text{kick}})
$$

The actor network **factors** the action space into two output layers: one for the four discrete actions and another for all six continuous parameters.

- *Context*:
	- The factored output is a design choice with important implications. All six continuous parameters are always produced by the network, regardless of which discrete action is selected. This means parameters for unselected actions still receive gradient updates from the critic. The paper acknowledges this design and shows empirically that the critic learns to provide meaningful gradients to the *correct* parameters for each discrete action — a non-obvious result, since the critic is never explicitly told which discrete action was applied.
	- This factored approach contrasts with Masson & Konidaris (2015), who used separate policies for discrete selection and continuous parameterization, alternating between optimizing each. The monolithic approach here is simpler architecturally but requires the network to implicitly learn the action-parameter associations.

## 4.1 Action Selection and Exploration

**Deterministic action selection**: At each timestep, the actor outputs values for all four discrete actions and all six parameters. The discrete action is the maximally valued output $a = \max(\text{Dash}, \text{Turn}, \text{Tackle}, \text{Kick})$, paired with its associated parameters.

**Exploration**: $\epsilon$-greedy exploration adapted to parameterized action space:
- With probability $\epsilon$, a random discrete action is selected and its continuous parameters are sampled uniformly.
- $\epsilon$ is annealed from 1.0 to 0.1 over the first 10,000 updates.

- *Context*:
	- The use of $\epsilon$-greedy rather than Ornstein-Uhlenbeck noise (as in the original DDPG) is notable. OU noise adds temporally correlated perturbations to continuous actions, which can be beneficial for smooth physical control. However, in a parameterized action space, exploration must also cover the *discrete* dimension — randomly selecting different action types is essential for discovering that kicking is useful after approaching the ball. Uniform random action selection achieves this naturally.

---

# 5. Bounded Parameter Space Learning

The HFO domain bounds the range of each continuous parameter: directions in $[-180, 180]$ degrees and power in $[0, 100]$. Without enforcing these bounds, continuous parameters routinely exceed their ranges within a few hundred updates, quickly trending toward astronomically large values.

- The root cause: the critic provides gradients that encourage the actor to *continue increasing* a parameter that already exceeds bounds. This is rational from the critic's perspective (more power is generally better) but physically invalid.

The paper evaluates three approaches to gradient bounding:

### Zeroing Gradients

The simplest approach: zero the critic's gradient for any parameter that has reached or exceeded its bound (**See MATHEMATICS §5**):

$$
\nabla p = \begin{cases} \nabla p & \text{if } p_{\min} < p < p_{\max} \\ 0 & \text{otherwise} \end{cases}
$$

- **Failure mode 1 — Indirect overflow**: Zeroing the gradient for parameter $p$ does not prevent gradients applied to *other* parameters $p_i \neq p$ from inadvertently causing $p$ to overflow. The paper observed learned networks attempting to dash with power 120 (maximum is 100).
- **Failure mode 2 — Instability**: One of two zeroing-gradient agents exhibited unstable Q-values and divergent critic losses.

### Squashing Gradients

A squashing function (e.g., $\tanh$) bounds each parameter's activation, which is then rescaled into the intended range.

- **Failure mode — Saturation**: The critic's gradients push parameters toward extreme values, saturating the squashing function. Once saturated, the agent takes the same action with fixed maximum/minimum parameters at every timestep, and requires many updates to recover — which in practice never happens.

### Inverting Gradients

The paper's core algorithmic contribution. Gradients are **downscaled** as the parameter approaches its boundary and **inverted** if the parameter exceeds the range (**See MATHEMATICS §6**):

$$
\nabla p = \nabla p \cdot \begin{cases} (p_{\max} - p) / (p_{\max} - p_{\min}) & \text{if } \nabla p \text{ suggests increasing } p \\ (p - p_{\min}) / (p_{\max} - p_{\min}) & \text{otherwise} \end{cases}
$$

- If the critic continually recommends increasing a parameter, the parameter converges to its upper bound smoothly.
- If the critic then reverses direction, the parameter responds *immediately* — no recovery from saturation is needed.
- The observed dash power reaches 98.8 out of 100 — close to the boundary but not exceeding it.

- *Context*:
	- The Inverting Gradients technique is the paper's most transferable contribution. It solves a problem that arises in *any* domain with bounded continuous actions, not just parameterized action spaces. The key insight is that the gradient scaling factor $(p_{\max} - p) / (p_{\max} - p_{\min})$ approaches zero as $p \to p_{\max}$, naturally damping updates near the boundary. But unlike $\tanh$ squashing, the gradient does not get "trapped" — when the desired direction of change reverses, the opposite scaling factor $(p - p_{\min}) / (p_{\max} - p_{\min})$ is at its maximum, providing full gradient magnitude for the return.
	- This is not specific to HFO or parameterized action spaces. Any continuous, bounded action space will face the same gradient overflow problem if left unchecked. The technique has proven useful for subsequent work in continuous control.

---

# 6. Results

Two agents are independently trained for each of the three gradient bounding approaches (six agents total), each for 3 million iterations (approximately 20,000 episodes). Training took three days per agent on an NVidia Titan-X GPU.

- **Inverting Gradients**: Both agents learned to reliably approach the ball and score goals. The learning curves show three distinct phases:
	1. Approaching the ball (around episode 1,500)
	2. Kicking toward the goal (episodes 2,000–8,000)
	3. Scoring goals (around episode 10,000)
- **Squashing Gradients**: Neither agent learned. Parameters stayed within bounds but squashing functions became saturated. Agents took the same action with fixed parameters at every timestep. Q-values were stable but reflected the sub-optimal fixed policy — critic loss stayed near zero because no reward was discovered.
- **Zeroing Gradients**: Neither agent learned reliably. One agent exhibited catastrophic instability with astronomically high Q-values and exploding critic loss. The other was stable but showed no learning progress.

- *Context*:
	- The three-phase learning curve for Inverting Gradients is a natural consequence of the reward structure. The move-to-ball component dominates early learning, the kick-to-goal component emerges once the agent can reliably reach the ball, and goal scoring emerges once the approach-kick sequence is consistently executed. This emergent curriculum arises from the reward design without explicit staging.
	- The analysis of failure modes is particularly valuable. The squashing failure reveals that the critic's gradient signal is fundamentally *biased toward extremes* in bounded spaces — it tends to push parameters toward their limits. Any bounding strategy must account for this tendency without losing sensitivity to future gradient reversals. Only Inverting Gradients achieves this balance.

---

# 7. Soccer Evaluation

The Inverting Gradients agents are compared against two baselines:

- **Helios-Agent2D**: The 2012 RoboCup 2D world champion, a hand-coded expert policy (Akiyama, 2010). An extremely competent player and a high performance bar.
- **SARSA baseline**: A model-free on-policy RL agent using high-level discrete actions (move, dribble, shoot) with tile-coded state features.

To demonstrate reliability, five additional Inverting Gradients agents are trained, for a total of seven (DDPG$_1$–DDPG$_7$). All seven learn to score goals.

| Agent | Scoring % | Avg. Steps to Goal |
|---|---|---|
| Helios' Champion | 0.962 | 72.0 |
| SARSA | 0.81 | 70.7 |
| DDPG$_1$ | **1.00** | 108.0 |
| DDPG$_2$ | 0.99 | 107.1 |
| DDPG$_3$ | 0.98 | 104.8 |
| DDPG$_4$ | 0.96 | 112.3 |
| DDPG$_5$ | 0.94 | 119.1 |
| DDPG$_6$ | 0.84 | 113.2 |
| DDPG$_7$ | 0.80 | 118.2 |

- Six of seven DDPG agents outperform the SARSA baseline, and three exceed Helios' scoring reliability.
- DDPG$_1$ achieves **100% scoring reliability** over 100 evaluation episodes.
- However, DDPG agents are significantly slower: 104.8–119.1 average steps vs. Helios' 72.0 steps.

- *Context*:
	- The speed-reliability tradeoff is illuminating. The DDPG agents face no temporal pressure from the reward signal — there is no penalty for taking longer. They learn to take extra time to line up shots carefully, becoming more accurate at the cost of speed. Helios, being hand-coded with tactical awareness, executes rapid sequences of approach-dribble-shoot that a learned agent has no incentive to discover.
	- Occasional Helios failures result from noise in the action space, which can cause missed kicks. The DDPG agents, by contrast, learn policies that are robust to this noise through their extended approach sequences. This suggests that an explicit time penalty in the reward function could produce agents that are both fast and reliable.
	- The fact that all seven independently trained DDPG agents learn to score demonstrates the *reliability* of the learning process itself — the algorithm is not dependent on lucky initialization.

---

# 8. Related Work

- **Andre & Teller (1999)**: Used Genetic Programming to evolve RoboCup 2D policies with a sequence of reward functions. Like the current work, the policies are entirely trained with no hand-coded components, but they used evolution rather than gradient-based learning.
- **Masson & Konidaris (2015)**: Formalized the PAMDP framework and proposed alternating optimization — fixing discrete selection while optimizing parameters, then vice versa. They used linear function approximation rather than deep networks and tested on a simplified domain (co-located agent and ball, kick-only actions).
- **Brainstormers (Riedmiller et al., 2009)**: Competitively integrated neural network decision-making across skills, but used constrained, focused training environments for each skill. The current work learns all skills within a single monolithic policy.
- **Lillicrap et al. (2015)**: Provided the continuous DDPG framework extended here.
- **Mnih et al. (2015)**: The seminal DQN work that proved deep networks as value function approximators.

- *Context*:
	- The critical distinction from Masson & Konidaris (2015) is architectural: the current paper uses a single monolithic network that simultaneously produces discrete actions and continuous parameters, relying on the critic to learn the appropriate associations. Masson & Konidaris' alternating approach has convergence guarantees (to local optima) but requires separate optimization phases. The deep network approach offers no convergence guarantees but empirically learns richer policies.

---

# 9. Future Work

- **Scoring on a goalie**: The current experiments use an empty goal. Adding a defender would test the agent's ability to learn adaptive strategies.
- **Multi-agent collaboration**: Both adhoc teamwork (a single learner collaborating with unknown teammates) and true multi-agent settings (multiple learners collaborating).
- **Exploiting critic state gradients**: The critic's gradients with respect to *state* inputs $\nabla_s Q(s, a|\theta^Q)$ indicate directions of improvement in state space. An agent with a forward model could exploit these to transition into states the critic finds more favourable. Recent work on model-based deep RL (Oh et al., 2015) shows that detailed next-state models are possible.

- *Context*:
	- The state-gradient idea is particularly forward-looking. It suggests a hybrid model-based/model-free approach: the model-free critic identifies *which* states are valuable, and the model-based component plans *how to reach* them. This connects to later work on world models and model-based RL planners, though the paper predates that literature by several years.

---

# 10. Conclusion

The paper demonstrates that deep reinforcement learning can master parameterized action spaces from scratch, producing agents that outperform hand-coded expert policies in reliability. The key enabler is the **Inverting Gradients** technique for bounding action-space gradients, which is a general-purpose contribution applicable to any bounded-continuous action space.

- The agents learn the full behavioural pipeline — approach, dribble, score — within a single monolithic policy, without explicit hierarchical decomposition or modular skill learning.
- The extension of DDPG to PAMDPs is straightforward architecturally (factored output layers) but requires the gradient bounding innovation to achieve stable learning.

---

# MATHEMATICS

The mathematical framework proceeds from the definition of parameterized action spaces, through the DDPG actor-critic machinery, to the gradient bounding strategies that enable stable learning. The derivation chain establishes why standard continuous-action RL fails in bounded parameterized spaces and how Inverting Gradients resolves the failure.

### 1. Parameterized Action Space

A **Parameterized Action Space MDP (PAMDP)** augments a standard MDP with structured actions. The discrete action set $A_d = \{a_1, a_2, \ldots, a_k\}$ contains $k$ action types. Each action $a \in A_d$ has $m_a$ continuous parameters.

**Parameterized Action Space ($A$):**

$$
A = \bigcup_{a \in A_d} (a, p_1^a, \ldots, p_{m_a}^a) \tag{1}
$$

where $A_d$ is the set of discrete action types, $p_j^a \in \mathbb{R}$ is the $j$-th continuous parameter of action $a$, and $m_a$ is the number of parameters for action $a$. Each action is represented as a tuple combining the discrete selection with its continuous parameterization.

### 2. HFO Reward Signal

The reward signal provides dense feedback to guide learning through three behavioural phases: approaching the ball, kicking toward the goal, and scoring.

**Reward Signal ($r_t$):**

$$
r_t = d_{t-1}(a, b) - d_t(a, b) + I_t^{\text{kick}} + 3\left(d_{t-1}(b, g) - d_t(b, g)\right) + 5 \cdot I_t^{\text{goal}} \tag{2}
$$

where $d_t(a, b)$ is the distance between agent $a$ and ball $b$ at timestep $t$, $d_t(b, g)$ is the distance between ball $b$ and goal center $g$, $I_t^{\text{kick}} \in \{0, 1\}$ is an indicator that fires once per episode when the agent first reaches kicking range, and $I_t^{\text{goal}} \in \{0, 1\}$ is an indicator for scoring. The coefficient 3 on the kick-to-goal component compensates for the negative move-to-ball reward generated when the ball moves away after a kick.

### 3. Critic Loss Function

The critic network $Q(s, a|\theta^Q)$ is trained to minimize the temporal difference error. In continuous action spaces, the intractable $\max_{a'} Q(s', a')$ is replaced by the actor's next-state action $\mu(s'|\theta^\mu)$.

**Critic Loss ($L_Q$):**

$$
L_Q(s, a|\theta^Q) = \left(Q(s, a|\theta^Q) - \left(r + \gamma Q(s', \mu(s'|\theta^\mu)|\theta^Q)\right)\right)^2 \tag{3}
$$

where $Q(s, a|\theta^Q)$ is the critic's estimate of the action-value, $r$ is the observed reward, $\gamma$ is the discount factor, and $\mu(s'|\theta^\mu)$ is the actor's proposed next-state action.

### 4. Actor Policy Gradient

The actor $\mu(s|\theta^\mu)$ is updated using the critic's gradients with respect to the action inputs. These gradients indicate directions of change in action space that increase the estimated Q-value.

**Actor Policy Gradient ($\nabla_{\theta^\mu}\mu$):**

$$
\nabla_{\theta^\mu} \mu(s) = \nabla_a Q(s, a|\theta^Q) \nabla_{\theta^\mu} \mu(s|\theta^\mu) \tag{4}
$$

where $\nabla_a Q(s, a|\theta^Q)$ is the gradient of the critic with respect to the action input (obtained via a backward pass through the critic), and $\nabla_{\theta^\mu} \mu(s|\theta^\mu)$ is the Jacobian of the actor with respect to its parameters. This is a chain rule application: action-space gradients from the critic are backpropagated through the actor to update $\theta^\mu$.

### 5. Stabilized Updates

Employing target networks $Q'$, $\mu'$ and replay memory $D$ produces stable versions of the critic loss and actor update.

**Stabilized Critic Loss:**

$$
L_Q(\theta^Q) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1}) \sim D}\left[\left(Q(s_t, a_t) - \left(r_t + \gamma Q'(s_{t+1}, \mu'(s_{t+1}))\right)\right)^2\right] \tag{5}
$$

**Stabilized Actor Update:**

$$
\nabla_{\theta^\mu} \mu = \mathbb{E}_{s_t \sim D}\left[\nabla_a Q(s_t, a|\theta^Q) \nabla_{\theta^\mu} \mu(s_t)\big|_{a = \mu(s_t)}\right] \tag{6}
$$

**Target Network Soft Updates:**

$$
\theta^{Q'} = \tau\theta^Q + (1 - \tau)\theta^{Q'}, \quad \theta^{\mu'} = \tau\theta^\mu + (1 - \tau)\theta^{\mu'} \tag{7}
$$

where $D$ is the replay memory, $Q'$ and $\mu'$ are the target critic and target actor networks, and $\tau \ll 1$ is the soft update rate. The target networks provide slowly-changing regression targets, preventing the destructive feedback loop between actor and critic.

### 6. Zeroing Gradients

The simplest bounding strategy: suppress any gradient that would push a parameter beyond its bounds.

**Zeroing Gradients ($\nabla p$):**

$$
\nabla p = \begin{cases} \nabla p & \text{if } p_{\min} < p < p_{\max} \\ 0 & \text{otherwise} \end{cases} \tag{8}
$$

where $\nabla p$ denotes the critic's gradient with respect to parameter $p$ (i.e., $\nabla_p Q(s_t, a|\theta^Q)$), and $p_{\min}$, $p_{\max}$ are the parameter's bounds. This approach fails because gradients applied to other parameters $p_i \neq p$ can indirectly cause $p$ to overflow, and the sudden zeroing creates instability in the optimization landscape.

### 7. Inverting Gradients

The paper's central algorithmic contribution: a continuous, non-saturating gradient scaling that smoothly enforces parameter bounds.

**Inverting Gradients ($\nabla p$):**

$$
\nabla p = \nabla p \cdot \begin{cases} \dfrac{p_{\max} - p}{p_{\max} - p_{\min}} & \text{if } \nabla p \text{ suggests increasing } p \\[8pt] \dfrac{p - p_{\min}}{p_{\max} - p_{\min}} & \text{otherwise} \end{cases} \tag{9}
$$

where all symbols are defined as in §6. The scaling factor is a linear function of the parameter's current position within its range. When $p$ approaches $p_{\max}$ and the gradient suggests increasing $p$, the factor approaches zero — damping the update. If the gradient then reverses (suggests decreasing $p$), the opposite factor $(p - p_{\min})/(p_{\max} - p_{\min})$ is near its maximum, providing full gradient magnitude. If $p$ exceeds $p_{\max}$, the factor becomes negative, *inverting* the gradient and actively pushing $p$ back within bounds.

This achieves three properties simultaneously: (1) parameters remain within bounds, (2) no saturation occurs — the gradient signal is always responsive to direction changes, and (3) the scaling is smooth and differentiable within the valid range.

---

### **1. Problem:**

The paper addresses the inability of existing DRL architectures to handle **parameterized action spaces** — hybrid spaces where a discrete action type must be paired with continuous parameters. Prior deep RL methods operated either in purely discrete spaces (DQN) or purely continuous spaces (DDPG), with no framework for the structured combination. The gap is a method that can learn monolithic policies in PAMDPs, where the discrete-continuous coupling introduces latent structure that purely continuous formulations ignore.

### **2. Setup:**

The method operates in a **Partially Observable MDP** (approximated as an MDP via 58 egocentric features) with a parameterized action space consisting of four discrete actions and six total continuous parameters. The environment is episodic (500-timestep limit), the state space is continuous and low-dimensional (58 features derived from a pre-existing world model), and continuous parameters are bounded in physically meaningful ranges ($[0, 100]$ for power, $[-180, 180]$ for direction). The framework uses an Actor/Critic architecture with a factored output — one layer for discrete action selection, another for all continuous parameters.

### **3. Key Idea:**

The core contribution is the **Inverting Gradients** technique for bounding action-space gradients in continuous, bounded action spaces. By scaling the critic's gradient by a factor proportional to the parameter's distance from its nearest boundary — and inverting the gradient when the parameter exceeds bounds — the method avoids both the saturation of squashing functions and the indirect overflow of gradient zeroing. This enables a single DDPG network to learn complete policies in parameterized action spaces from scratch.

### **4. Assumptions:**

**Explicit:**
- The reward signal must be hand-crafted with dense intermediate components; a sparse goal-only reward is insufficient for exploration
- The agent has access to pre-processed features from the Helios-Agent2D world model, not raw observations
- Episodes are bounded at 500 timesteps
- Parameter bounds are known a priori and fixed

**Implicit:**
- The 58-feature egocentric representation is sufficient for approximately Markovian decision-making, despite the environment being a POMDP
- A single monolithic policy can learn diverse sub-tasks (approaching, dribbling, kicking, scoring) without hierarchical decomposition
- The critic learns to provide meaningful gradients to the correct parameters for each discrete action, despite receiving all parameters regardless of the selected action
- The training environment (empty goal, no defenders) is representative enough for the learned skills to generalize

### **5. Limitation:**

- **Environment simplicity**: Experiments use an empty goal with no goalie or defenders. The agent never needs to learn adversarial strategy, deceptive movements, or passing.
- **Reward engineering dependence**: The learning process is highly sensitive to the specific reward components and their weights. The three-component reward with fixed coefficients (1:3:5) was manually tuned.
- **Speed deficit**: DDPG agents take 46–65% more steps than Helios to score, reflecting the absence of temporal pressure in the reward.
- **Feature dependency**: The agent relies on high-level features from an external world model. Whether Inverting Gradients remains stable when combined with end-to-end feature learning from raw visual input is untested.
- **No convergence guarantees**: The deep network parameterization offers no theoretical convergence guarantees, relying entirely on empirical success.

### **6. Relevance & Open Questions:**

This paper demonstrates that DRL can handle the structured action spaces common in robotics and strategic planning, where actions have both categorical and continuous components. Key open questions:

- **Multi-agent scaling**: Can monolithic DDPG policies be extended to cooperative or adversarial multi-agent settings, where the parameterized action space grows and agents must coordinate?
- **Hierarchical decomposition**: Would an explicit skill hierarchy (approach → dribble → shoot) learn faster or generalize better than the monolithic approach, and how does Inverting Gradients interact with hierarchical RL?
- **Model-based synergy**: The paper's suggestion to exploit $\nabla_s Q(s, a|\theta^Q)$ with a forward model anticipates later work on world models. Can the state-gradient approach improve sample efficiency in parameterized action spaces?
- **End-to-end learning**: Extending the framework to learn from raw observations (pixels) simultaneously with the parameterized action policy is an important step toward real-world applicability.
- **Reward-free exploration**: The paper's reliance on hand-crafted rewards is a significant practical limitation. Can intrinsic motivation or curiosity-driven exploration provide sufficient signal in parameterized action spaces?

---

### Integration:

* **Problem:** This paper fills a foundational gap in the DRL toolkit by providing a working method for parameterized action spaces — the natural formalism for many real-world control problems where qualitative action choice and quantitative parameterization are inseparable. For portfolio management applications, this resonates with the problem of combining discrete allocation decisions (which assets to hold) with continuous sizing decisions (how much of each). The Inverting Gradients technique, while demonstrated in soccer, is a general-purpose contribution to any bounded-continuous action domain.

* **Limitation:** The most significant limitations for broader applications are the **environment simplicity** (no adversarial dynamics, no partial observability beyond the feature approximation) and the **reward engineering dependency**. Real-world domains rarely offer the luxury of manually tunable, dense reward signals, and the absence of defenders means the learned policies have never been tested against adaptive opponents. The speed-reliability tradeoff also suggests that the reward design implicitly shapes the *style* of the learned policy in ways that may not align with deployment requirements. For financial applications specifically, the analogue would be an agent that achieves high Sharpe ratio but with unacceptable latency in execution — correctness without timeliness.
