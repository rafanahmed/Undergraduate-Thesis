[[A Distributional Perspective on Reinforcement Learning.pdf]]

# **Abstract:**

The paper argues for the fundamental importance of the **value distribution**: the full probability distribution of the random return received by a reinforcement learning agent. This stands in contrast to the common approach which models only the *expectation* of this return (the value $Q$).

- The value distribution has an established literature, but it has always been subordinated to specific purposes — most notably implementing *risk-aware* behaviour.
- This paper repositions the value distribution as a central object in RL, not merely a tool for risk-sensitivity.

The paper contributes along four axes:

1. **Policy evaluation theory:** The distributional Bellman operator $\mathcal{T}^\pi$ is a $\gamma$-contraction in the maximal Wasserstein metric $\bar{d}_p$, guaranteeing convergence to the true value distribution $Z^\pi$.
2. **Control theory:** The distributional Bellman *optimality* operator is **not** a contraction in any distributional metric — exposing a significant instability in the control setting.
3. **Algorithmic contribution:** A new algorithm (**Categorical DQN / C51**) applies the Bellman equation to learn approximate value distributions using a discrete parametric distribution.
4. **Empirical results:** State-of-the-art performance on the Arcade Learning Environment (57 Atari 2600 games), with 701% mean and 178% median human-normalized scores.

- *Context*:
	- The value distribution $Z^\pi(x,a)$ is defined as the *full* probability distribution over possible returns from state-action pair $(x,a)$ under policy $\pi$. Its expectation is the standard value function: $Q^\pi(x,a) = \mathbb{E}[Z^\pi(x,a)]$. Where traditional RL compresses all possible future outcomes into a single scalar average, the value distribution preserves the entire shape — multimodality, variance, skewness, tail behaviour.
	- The distributional Bellman equation $Z(x,a) \stackrel{D}{=} R(x,a) + \gamma Z(X', A')$ characterizes the value distribution through the interaction of three random variables: the immediate reward $R$, the next state-action pair $(X', A')$, and the random return at that next state $Z(X', A')$.
	- The paper's key philosophical claim is that modelling the full distribution is not merely useful for risk — it is *intrinsically beneficial* for approximate RL, producing more stable learning, better gradient signals, and richer representations.

---

# 1. Introduction

The paper opens by establishing Bellman's equation as the foundation of value-based RL:

$$
Q(x, a) = \mathbb{E}\left[R(x,a)\right] + \gamma\,\mathbb{E}\left[Q(X', A')\right]
$$

- This describes the value in terms of expected reward and expected outcome of the random transition $(x,a) \to (X', A')$.

The distributional counterpart replaces expectations with distributional equality:

$$
Z(x, a) \stackrel{D}{=} R(x, a) + \gamma Z(X', A')
$$

- *Context*:
	- The notation $\stackrel{D}{=}$ means "equal in distribution" — the random variable on the left has the same probability law as the expression on the right, but they are not necessarily the same random variable. This is fundamentally different from the expectation-based Bellman equation, which is an equality between *scalars*.
	- This distributional equation says: the return distribution at $(x,a)$ is whatever distribution you get by adding the random reward $R(x,a)$ to a discounted copy of the return distribution at the random next state-action pair $(X', A')$. Three independent sources of randomness interact to form a compound distribution.

The paper identifies prior uses of the value distribution:
- **Parametric uncertainty modelling:** Dearden et al. (1998) used Gaussian approximations with Normal-Gamma priors.
- **Risk-sensitive algorithms:** Morimura et al. (2010a,b) studied distributional Bellman equations from the CDF perspective.
- **Theoretical analysis:** Azar et al. (2012), Lattimore & Hutter (2012).

The introduction previews four contributions:
1. **Contraction of T^π:** The policy evaluation operator is a contraction in the Wasserstein metric — but *not* in total variation, KL divergence, or Kolmogorov distance.
2. **Instability of T:** The optimality operator is *not* a contraction in any distributional metric, though it still contracts in expectation.
3. **Better approximations:** Distributional learning preserves multimodality, mitigates nonstationarity effects, and leads to more stable approximate RL.
4. **Empirical demonstration:** Dramatic improvements in the Arcade Learning Environment.

- *Context*:
	- The claim that learning a *distribution* helps even when the agent's policy is greedy with respect to *expected* value is counterintuitive. If the agent only cares about the mean, why would learning the full distribution matter? The paper's answer is nuanced: it is not that the distribution helps at decision time (the agent still picks actions by expected value), but that learning the distribution produces better *learning dynamics* — more stable gradients, richer error signals, and preservation of multimodal structure that would otherwise be collapsed and lost.
	- The connection to Veness et al. (2015) is significant: that work showed very fast learning by predicting Monte Carlo returns directly, raising the question of whether a Bellman-based distributional algorithm could be equally effective. This paper answers yes.

---

# 2. Setting

The paper models the agent-environment interaction as a time-homogeneous **Markov Decision Process** $(\mathcal{X}, \mathcal{A}, R, P, \gamma)$:
- $\mathcal{X}$: state space
- $\mathcal{A}$: action space
- $P$: transition kernel $P(\cdot \mid x, a)$
- $\gamma \in [0, 1]$: discount factor
- $R$: reward function, **explicitly treated as a random variable**

A stationary policy $\pi$ maps each state $x \in \mathcal{X}$ to a probability distribution over $\mathcal{A}$.

- *Context*:
	- Most RL formulations treat the reward as a deterministic function of $(x, a)$ or $(x, a, x')$. Treating $R$ as a *random variable* is essential here because the distributional framework needs to account for reward stochasticity as one of the three sources of randomness that shape the return distribution.

## 2.1 Bellman's Equations

The return $Z^\pi$ is the sum of discounted rewards along a trajectory. The value function is its expectation:

$$
Q^\pi(x, a) := \mathbb{E}\left[Z^\pi(x, a)\right] = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(x_t, a_t)\right] \tag{1}
$$

where $x_t \sim P(\cdot \mid x_{t-1}, a_{t-1})$, $a_t \sim \pi(\cdot \mid x_t)$, $x_0 = x$, $a_0 = a$.

The **Bellman operator** $\mathcal{T}^\pi$ and **optimality operator** $\mathcal{T}$ are defined as:

$$
\mathcal{T}^\pi Q(x, a) := \mathbb{E}\left[R(x,a)\right] + \gamma\,\mathbb{E}_{P,\pi}\left[Q(x', a')\right] \tag{2}
$$

$$
\mathcal{T}\,Q(x, a) := \mathbb{E}\left[R(x,a)\right] + \gamma\,\mathbb{E}_P\left[\max_{a' \in \mathcal{A}} Q(x', a')\right] \tag{3}
$$

- Both are **contraction mappings** in the expected-value setting.
- Repeated application of $\mathcal{T}^\pi$ converges exponentially to $Q^\pi$; repeated application of $\mathcal{T}$ converges to $Q^*$, the optimal value function.

- *Context*:
	- These operators are the formal backbone of algorithms like SARSA ($\mathcal{T}^\pi$) and Q-Learning ($\mathcal{T}$). The contraction property guarantees convergence: at each iteration, the distance to the fixed point shrinks by a factor of $\gamma$. The paper will show that this clean convergence story breaks down in interesting ways when we move from expectations to distributions.

---

# 3. The Distributional Bellman Operators

The paper now strips away the expectations and considers the **full distribution** of $Z^\pi$. The value distribution is viewed as a mapping from state-action pairs to distributions over returns.

- *Context*:
	- This section is the core theoretical contribution. The key results are: (1) T^π is a contraction for policy evaluation (good), and (2) T is *not* a contraction for control (problematic but informative). The reader interested only in the algorithm can skip to Section 4, but the theoretical results provide the intellectual justification for the entire approach.

## 3.1 Distributional Equations

The paper introduces the probability space $(\Omega, \mathcal{F}, \Pr)$:
- $\Omega$ is the space of all possible outcomes
- The $L_p$ norm of a random vector $U: \Omega \to \mathbb{R}^{\mathcal{X}}$ is $\|U\|_p := \left[\mathbb{E}\left[\|U(\omega)\|_p^p\right]\right]^{1/p}$
- For $p = \infty$: $\|U\|_\infty = \text{ess sup}\,\|U(\omega)\|_\infty$
- CDF notation: $F_U(y) := \Pr\{U \le y\}$
- Inverse CDF: $F_U^{-1}(q) := \inf\{y : F_U(y) \ge q\}$

A **distributional equation** $U \stackrel{D}{:=} V$ means $U$ has the same law as $V$ — the two sides relate the distributions of (conceptually) independent random variables.

## 3.2 The Wasserstein Metric

The primary analytical tool is the **Wasserstein metric** $d_p$ (also called Kantorovich or Mallows metric):

$$
d_p(F, G) := \inf_{U, V} \|U - V\|_p
$$

where the infimum is over all pairs $(U, V)$ with CDFs $F$ and $G$ respectively. The infimum is attained by the **inverse CDF transform**:

$$
d_p(F, G) = \left(\int_0^1 \left|F^{-1}(u) - G^{-1}(u)\right|^p du\right)^{1/p}
$$

- *Context*:
	- The Wasserstein metric measures the minimum "cost" of transporting probability mass from one distribution to another. Unlike KL divergence or total variation, it is sensitive to the *geometry* of the outcome space — moving mass a little costs a little. This geometric sensitivity is exactly why it works for value distributions: a return of 10 is "close" to a return of 11, and the metric respects this.
	- The inverse CDF transform provides a constructive way to compute $d_p$: line up both distributions by their quantile functions and measure the point-wise difference. This coupling is optimal.

**Key properties** for scalar $a$ and random variable $A$ independent of $U, V$:
- **(P1)** $d_p(aU, aV) \le |a|\,d_p(U, V)$ — scalar scaling
- **(P2)** $d_p(A + U, A + V) \le d_p(U, V)$ — shift by an independent random variable does not increase distance
- **(P3)** $d_p(AU, AV) \le \|A\|_p\,d_p(U, V)$ — scaling by a random variable

### Partition Lemma (Lemma 1)

Let $A_1, A_2, \ldots$ partition $\Omega$ (each $A_i(\omega) \in \{0, 1\}$, exactly one is 1 for each $\omega$). Then:

$$
d_p(U, V) \le \sum_i d_p(A_i U, A_i V)
$$

- *Context*:
	- This lemma is a crucial technical tool. It says: if you decompose the sample space into disjoint events and measure the Wasserstein distance separately within each event, the sum of these piece-wise distances is an upper bound on the total distance. The paper uses this to separate "solved" states (where the greedy policy is already optimal) from "unsolved" states in the proof of Theorem 1.

### Maximal Wasserstein Metric (d̄_p)

For value distributions $Z_1, Z_2 \in \mathcal{Z}$ (the space of value distributions with bounded moments):

$$
\bar{d}_p(Z_1, Z_2) := \sup_{x, a} d_p(Z_1(x, a), Z_2(x, a))
$$

**Lemma 2:** $\bar{d}_p$ is a metric over value distributions (the only nontrivial property is the triangle inequality, proved via the triangle inequality for $d_p$).

## 3.3 Policy Evaluation

In policy evaluation, the goal is to characterize the value distribution $Z^\pi$ for a fixed policy $\pi$.

The **transition operator** $P^\pi : \mathcal{Z} \to \mathcal{Z}$:

$$
P^\pi Z(x, a) \stackrel{D}{:=} Z(X', A') \tag{4}
$$

where $X' \sim P(\cdot \mid x, a)$, $A' \sim \pi(\cdot \mid X')$.

The **distributional Bellman operator** $\mathcal{T}^\pi : \mathcal{Z} \to \mathcal{Z}$:

$$
\mathcal{T}^\pi Z(x, a) \stackrel{D}{:=} R(x, a) + \gamma\,P^\pi Z(x, a) \tag{5}
$$

- *Context*:
	- Equation (5) looks superficially similar to the standard Bellman operator (2), but it is fundamentally different. It is an equation between *distributions*, not scalars. Three independent sources of randomness define the compound distribution $\mathcal{T}^\pi Z$:
		- (a) The randomness in the reward $R$
		- (b) The randomness in the transition $P^\pi$ (which state-action pair comes next)
		- (c) The next-state value distribution $Z(X', A')$ (which itself is a random distribution because $(X', A')$ is random)
	- The usual independence assumption applies: reward, transition, and next-state return are mutually independent.

### 3.3.1 Contraction in d̄_p

**Lemma 3:** $\mathcal{T}^\pi : \mathcal{Z} \to \mathcal{Z}$ is a $\gamma$-contraction in $\bar{d}_p$.

The proof proceeds by:
1. Using property (P2) to remove the shared reward $R(x,a)$ from both sides.
2. Using property (P1) to pull out the discount factor $\gamma$.
3. Bounding $d_p(P^\pi Z_1(x,a), P^\pi Z_2(x,a)) \le \sup_{x',a'} d_p(Z_1(x',a'), Z_2(x',a'))$ from the definition of $P^\pi$.
4. Taking the supremum over $(x,a)$ yields $\bar{d}_p(\mathcal{T}^\pi Z_1, \mathcal{T}^\pi Z_2) \le \gamma\,\bar{d}_p(Z_1, Z_2)$.

By **Banach's fixed point theorem**, $\mathcal{T}^\pi$ has a unique fixed point, which must be $Z^\pi$ as defined by equation (1). The sequence $\{Z_k\}$ generated by $Z_{k+1} := \mathcal{T}^\pi Z_k$ converges to $Z^\pi$ in $\bar{d}_p$ for all $1 \le p \le \infty$.

- *Context*:
	- This is the headline positive result. It says that iterating the distributional Bellman operator converges to the true value distribution at the same exponential rate $\gamma^k$ as the standard Bellman operator converges to $Q^\pi$ — but in a *much stronger* sense. Convergence in the Wasserstein metric implies convergence of all moments (mean, variance, skewness, etc.), not just the mean.
	- Critically, the same operator is **not** a contraction in total variation distance (shown by Chung & Sobel, 1987), KL divergence, or Kolmogorov distance. The choice of metric is not a technicality — it is essential to the result.

### 3.3.2 Contraction in Centered Moments

The relationship between $d_2$ and variance is:

$$
d_2^2(U, V) \le \mathbb{E}[(U - V)^2] = \mathbb{V}(C) + (\mathbb{E}[C])^2
$$

where $C(\omega) := U(\omega) - V(\omega)$ is the coupling.

- $\mathcal{T}^\pi$ is a $\gamma^2$-contraction in **variance** (Sobel, 1982):

$$
\|\mathbb{V}(\mathcal{T}^\pi Z_1) - \mathbb{V}(\mathcal{T}^\pi Z_2)\|_\infty \le \gamma^2 \|\mathbb{V}(Z_1) - \mathbb{V}(Z_2)\|_\infty
$$

- $\mathcal{T}^\pi$ is **not** a contraction in the $p$-th centered moment for $p > 2$, but the centered moments of the iterates $\{Z_k\}$ still converge exponentially to those of $Z^\pi$ (extending Rösler, 1992).

- *Context*:
	- The variance contraction at rate $\gamma^2$ (rather than $\gamma$) is intuitive: variance scales with the square of the random variable, so discounting by $\gamma$ at each step produces a $\gamma^2$ contraction in variance. This means variance estimates converge faster than mean estimates, which is encouraging for distributional learning.

## 3.4 Control

The paper now turns to the control setting — seeking a policy $\pi$ that maximizes value. This is where the distributional framework encounters fundamental difficulties.

### Definition 1 (Optimal Value Distribution)

An **optimal value distribution** is the value distribution of an optimal policy:

$$
\mathcal{Z}^* := \{Z^{\pi^*} : \pi^* \in \Pi^*\}
$$

- Not all value distributions with expectation $Q^*$ are optimal — the full distributional structure must match the return under some $\pi^* \in \Pi^*$.

### Definition 2 (Greedy Policy)

$$
\mathcal{G}_Z := \left\{\pi : \sum_a \pi(a \mid x)\,\mathbb{E}[Z(x,a)] = \max_{a'} \mathbb{E}[Z(x,a')]\right\}
$$

### Distributional Bellman Optimality Operator

A distributional Bellman optimality operator is any $\mathcal{T}$ that implements a greedy selection rule:

$$
\mathcal{T}\,Z = \mathcal{T}^\pi Z \quad \text{for some } \pi \in \mathcal{G}_Z
$$

- *Context*:
	- In the expected-value case, the maximization at the next state is "invisible" — $\max_{a'} Q(x', a')$ implicitly selects the greedy action but produces a unique scalar. In the distributional case, the greedy action is explicit because we need to apply the full transition operator $\mathcal{T}^{\pi}$, and different greedy policies may yield different *distributions* even if they agree on expected value.

### Lemma 4 (Expectations Still Converge)

$$
\|\mathbb{E}[\mathcal{T}\,Z_1] - \mathbb{E}[\mathcal{T}\,Z_2]\|_\infty \le \gamma\|\mathbb{E}[Z_1] - \mathbb{E}[Z_2]\|_\infty
$$

In particular, $\mathbb{E}[Z_k] \to Q^*$ exponentially quickly.

- *Context*:
	- This says the expected-value dynamics are unchanged: even when operating on distributions, the means converge to $Q^*$ at the usual rate. The trouble is entirely in the distributional structure *around* the mean.

### Definition 3 (Nonstationary Optimal Value Distribution)

A **nonstationary optimal value distribution** $Z^{**}$ is the value distribution corresponding to a *sequence* of optimal policies (which may differ at each time step). The set of all such distributions is $\mathcal{Z}^{**}$.

### Theorem 1 (Convergence in the Control Setting)

Let $\mathcal{X}$ be measurable and $\mathcal{A}$ finite. Then:

$$
\lim_{k \to \infty} \inf_{Z^{**} \in \mathcal{Z}^{**}} d_p(Z_k(x, a), Z^{**}(x, a)) = 0 \quad \forall\, x, a
$$

- If $\mathcal{X}$ is finite, convergence is uniform.
- If there is a total ordering on $\Pi^*$ that the greedy selection respects, then $\mathcal{T}$ has a unique fixed point $Z^* \in \mathcal{Z}^*$.

- *Context*:
	- Compare this to Lemma 4: while the *mean* converges exponentially to $Q^*$, the *distribution* merely converges (in the limit, not exponentially) to the *set* $\mathcal{Z}^{**}$ — which is larger than $\mathcal{Z}^*$. The iterates may never settle on a single distribution; they can cycle among different nonstationary optimal value distributions. This is a dramatic weakening compared to the policy evaluation case.
	- The total-ordering condition for uniqueness is a technical fix: if the greedy selection always picks the "smallest" optimal policy (by some consistent ordering), then the operator reduces to a single $\mathcal{T}^{\pi^*}$, which has a unique fixed point by Lemma 3.

### Negative Results (Propositions 1–3)

**Proposition 1:** $\mathcal{T}$ is **not** a contraction. Demonstrated via a 2-state MDP where $\bar{d}_1(\mathcal{T}\,Z, \mathcal{T}\,Z^*) > \bar{d}_1(Z, Z^*)$ for suitable $Z$.

- The MDP has states $x_1, x_2$; from $x_2$, action $a_1$ yields 0 reward while optimal action $a_2$ yields $\varepsilon \pm 1$ with equal probability. When $\mathcal{T}$ changes the greedy action at $x_1$ from $a_2$ to $a_1$, the distributional distance *increases*.

**Proposition 2:** Not all optimality operators have a fixed point $Z^* = \mathcal{T}\,Z^*$. A specific tie-breaking rule can cause the iterates to alternate between two distributions.

**Proposition 3:** Even if $\mathcal{T}$ has a fixed point, the iterates $\{Z_k\}$ need not converge to it.

- *Context*:
	- These negative results paint a sobering picture: the distributional optimality operator is fundamentally more volatile than its expected-value counterpart. The root cause is that greedy action selection introduces discontinuities — a small change in the distribution can flip the greedy action, causing a large distributional shift. This is the distributional analogue of what Gordon (1995) called **chattering** in function approximation.
	- Despite this bleak theory, the practical algorithm works extremely well. The paper suggests this is because gradient-based learning with cross-entropy loss effectively *averages* over the distributional instability, similar to conservative policy iteration.

---

# 4. Approximate Distributional Learning

This section presents the practical algorithm. The key challenges are: (1) choosing a parametric family to represent distributions, and (2) defining a tractable learning objective.

## 4.1 Parametric Distribution

The value distribution is modelled as a **discrete distribution** over $N$ fixed atoms:

$$
z_i = V_{\text{MIN}} + i\,\Delta z, \quad 0 \le i < N, \quad \Delta z = \frac{V_{\text{MAX}} - V_{\text{MIN}}}{N - 1}
$$

The atom probabilities are given by a softmax over learned parameters $\theta : \mathcal{X} \times \mathcal{A} \to \mathbb{R}^N$:

$$
Z_\theta(x, a) = z_i \quad \text{w.p.} \quad p_i(x, a) := \frac{e^{\theta_i(x,a)}}{\sum_j e^{\theta_j(x,a)}}
$$

- *Context*:
	- The discrete distribution has two major advantages: (1) it is highly *expressive* — with enough atoms, it can approximate any distribution to arbitrary precision, and (2) it is *computationally friendly* — the softmax output head integrates naturally into deep network architectures.
	- The support $[V_{\text{MIN}}, V_{\text{MAX}}]$ is a hyperparameter. In experiments, $V_{\text{MAX}} = -V_{\text{MIN}} = 10$ was chosen from preliminary experiments on 5 training games. This clipping treats all returns beyond the range as equivalent — a form of inductive bias that surprisingly helps rather than hurts.

## 4.2 Projected Bellman Update

The central algorithmic problem: the Bellman update $\mathcal{T}\,Z_\theta$ and the parametrization $Z_\theta$ almost always have **disjoint supports**. The updated atoms $\hat{\mathcal{T}}z_j := r + \gamma z_j$ generally do not land on the original support $\{z_i\}$.

Two considerations guide the solution:
1. Minimizing the Wasserstein metric would be natural (from Section 3's analysis), but...
2. The Wasserstein loss cannot be minimized from sample transitions — its sample gradient is **biased** (Proposition 5 / Lemma 7 in the appendix).

- *Context*:
	- Proposition 5 shows that $d_p(P, Q) \le \mathbb{E}_{i \sim I}\,d_p(P_i, Q)$, with strict inequality in general. This means that replacing the true next-state distribution with a sample and computing the Wasserstein distance gives a biased estimate of the true loss, and — critically — the *gradients* are also biased: $\nabla_Q d_p(P_I, Q) \ne \mathbb{E}_{i \sim I}\,\nabla_Q d_p(P_i, Q)$. This rules out straightforward stochastic gradient descent on the Wasserstein loss.

### The Projection Φ

Instead, the paper **projects** the sample Bellman update onto the support of $Z_\theta$. Given a sample transition $(x, a, r, x')$ and greedy action $\pi(x') = \arg\max_a \mathbb{E}[Z_\theta(x', a)]$:

1. Compute the Bellman update for each atom: $\hat{\mathcal{T}}z_j := [r + \gamma z_j]_{V_{\text{MIN}}}^{V_{\text{MAX}}}$
2. Distribute probability $p_j(x', \pi(x'))$ to the two nearest atoms in the original support:

$$
(\Phi\,\hat{\mathcal{T}}\,Z_\theta(x, a))_i = \sum_{j=0}^{N-1} \left[1 - \frac{|[\hat{\mathcal{T}}z_j]_{V_{\text{MIN}}}^{V_{\text{MAX}}} - z_i|}{\Delta z}\right]_0^1 p_j(x', \pi(x')) \tag{7}
$$

The **loss function** is the cross-entropy term of the KL divergence:

$$
L_{x,a}(\theta) = D_{\text{KL}}\left(\Phi\,\hat{\mathcal{T}}\,Z_{\tilde{\theta}}(x, a)\;\|\;Z_\theta(x, a)\right)
$$

where $\tilde{\theta}$ is the target network parameter (held fixed, as in DQN).

- *Context*:
	- The projection reduces the distributional Bellman update to **multiclass classification**: the target is a categorical distribution over $N$ atoms (produced by the projection), and the loss is cross-entropy against the model's predicted distribution. This is elegant because cross-entropy between categorical distributions is one of the best-understood and most well-behaved losses in deep learning.
	- The projection itself is computable in $O(N)$ time: for each target atom $\hat{\mathcal{T}}z_j$, compute its fractional index $b_j = (\hat{\mathcal{T}}z_j - V_{\text{MIN}})/\Delta z$, then spread its probability to the floor and ceiling atoms proportionally to distance.

### Categorical Algorithm (Algorithm 1)

The full pseudocode:
- **Input:** Transition $(x_t, a_t, r_t, x_{t+1}, \gamma_t)$
- Compute expected values: $Q(x_{t+1}, a) = \sum_i z_i\,p_i(x_{t+1}, a)$
- Select greedy action: $a^* \gets \arg\max_a Q(x_{t+1}, a)$
- For each atom $j$:
	- Compute $\hat{\mathcal{T}}z_j = [r_t + \gamma_t z_j]_{V_{\text{MIN}}}^{V_{\text{MAX}}}$
	- Compute fractional index $b_j = (\hat{\mathcal{T}}z_j - V_{\text{MIN}})/\Delta z$
	- Distribute probability to neighbours: $m_{\lfloor b_j \rfloor}$ and $m_{\lceil b_j \rceil}$
- **Output:** Cross-entropy loss $-\sum_i m_i \log p_i(x_t, a_t)$

The paper also defines the **Bernoulli algorithm** for $N=2$: a single-parameter version where $\Phi\,\hat{\mathcal{T}}\,Z_\theta(x,a) := [\mathbb{E}[\hat{\mathcal{T}}\,Z_\theta(x,a)] - V_{\text{MIN}})/\Delta z]_0^1$.

---

# 5. Evaluation on Atari 2600 Games

The algorithm is evaluated on the **Arcade Learning Environment** (ALE; Bellemare et al., 2013) — 57 Atari 2600 games. The ALE is deterministic, but stochasticity arises from:
1. **State aliasing** — frame stacking produces partial observability
2. **Nonstationary policy** — the agent's policy changes during training
3. **Approximation errors** — the neural network introduces noise

The architecture is DQN (Mnih et al., 2015) modified to output atom probabilities $p_i(x, a)$ instead of scalar Q-values. The squared Bellman error loss is replaced by the cross-entropy loss $L_{x,a}(\theta)$. Action selection uses $\varepsilon$-greedy over expected values $\mathbb{E}[Z_\theta(x, a)]$.

- Hyperparameters: $V_{\text{MAX}} = -V_{\text{MIN}} = 10$, $N = 51$, Adam optimizer with step size $\alpha = 0.00025$ and $\varepsilon_{\text{adam}} = 0.01/L$ where $L=32$ is minibatch size.
- The resulting architecture is called **Categorical DQN** or **C51** (for the 51-atom version).

### Qualitative Observations (Figure 4)

In Space Invaders, the learned distributions clearly separate:
- **Safe actions** (Left, Right): high-value, concentrated distributions
- **Dangerous actions** (involving laser): bimodal distributions with significant probability at 0 (terminal/losing)

- *Context*:
	- This visualization is powerful evidence for the distributional approach. The expected-value agent would assign similar Q-values to "high expected value with some catastrophic risk" and "moderate expected value with no risk." The distributional agent can *see* the catastrophic mode and, while it still picks actions by expected value, the distributional learning target keeps the low-value and high-value events separated rather than averaging them into one unrealizable expectation.

In Pong, the agent detects **intrinsic stochasticity**: over consecutive frames, the value distribution shows bimodality reflecting uncertainty about reward timing. Since the agent's state does not include past rewards, it cannot extinguish this uncertainty.

## 5.1 Varying the Number of Atoms

Performance was studied on 5 training games (Asterix, Q\*Bert, Breakout, Pong, Seaquest) with $N \in \{2, 5, 11, 21, 51\}$ and $\varepsilon = 0.05$.

Key findings:
- **More atoms monotonically increases performance** — no evidence of network capacity saturation
- The **51-atom version outperforms DQN in all 5 games**
- Seaquest achieves **state-of-the-art performance** with 51 atoms
- The **Bernoulli algorithm** ($N=2$) outperforms DQN in 3 of 5 games, and is notably more robust in Asterix

## 5.2 State-of-the-Art Results

Full evaluation: $N=51$, $\varepsilon=0.01$ (training), $\varepsilon=0.001$ (evaluation), 57 Atari games.

| Agent | Mean | Median | > Human | > DQN |
|---|---|---|---|---|
| DQN | 228% | 79% | 24 | 0 |
| DDQN | 307% | 118% | 33 | 43 |
| Dueling | 373% | 151% | 37 | 50 |
| Prioritized | 434% | 124% | 39 | 48 |
| Prior. Duel. | 592% | 172% | 39 | 44 |
| **C51** | **701%** | **178%** | **40** | **50** |

Notable results:
- **Seaquest:** 266,434 (C51) vs. 5,861 (DQN) — a 45× improvement
- **Private Eye:** 15,095 (C51) vs. 147 (DQN)
- **Venture:** 1,520 (C51) vs. 163 (DQN)
- **Sparse reward games** benefit disproportionately from distributional learning

Training efficiency: within 50 million frames (25% of the full training budget), C51 outperforms fully-trained DQN on **45 of 57 games**.

Under **stochastic execution** (action rejected with $p = 0.25$): C51 obtains 126% mean and 21.5% median improvement over DQN on a DQN-normalized scale.

- *Context*:
	- The magnitude of improvement is remarkable: C51 surpasses DQN, Double DQN, Dueling, and Prioritized Replay *without incorporating any of those orthogonal algorithmic improvements*. It uses only the basic DQN training regime with a different output head and loss function.
	- The strong performance on sparse-reward games (Venture, Private Eye) suggests that distributional learning is better at propagating rarely occurring events — the distribution maintains separate modes for rare high-value and common low-value outcomes, whereas the mean collapses them.
	- **Erratum:** The originally reported mean of 1010% was corrected to 701% due to uncapped episode length in Atlantis (which inflated that game's score from 841,075 to 3.7 million). The median remains unchanged at 178%.

---

# 6. Discussion

## 6.1 Why Does Learning a Distribution Matter?

The paper proposes five mechanisms explaining why distributional learning improves performance even with a greedy-in-expectation policy:

### Reduced Chattering
- The theoretical instability of the optimality operator (Section 3.4) can prevent policy convergence under function approximation — **chattering** (Gordon, 1995).
- The gradient-based categorical algorithm effectively *averages* over different distributions, similar to **conservative policy iteration** (Kakade & Langford, 2002). The chattering persists but is integrated into the approximate solution.

### State Aliasing
- Even in deterministic environments, state aliasing (partial observability) produces effective stochasticity.
- By explicitly modelling the resulting distribution, the algorithm provides a more stable learning target.
- The Pong example (Section 5) demonstrates this: the agent cannot predict exact reward timing, but the distributional representation handles the bimodality naturally.

### A Richer Set of Predictions
- The distributional approach provides a rich set of **auxiliary predictions**: the probability that the return takes on each particular value.
- Unlike previously proposed auxiliary tasks (Caruana 1997; Sutton et al. 2011; Jaderberg et al. 2017), these predictions are *tightly coupled* with the agent's performance — they are not arbitrary side objectives but direct components of the value estimate.

### Framework for Inductive Bias
- The distributional perspective allows imposing assumptions about the domain.
- The bounded support $[V_{\text{MIN}}, V_{\text{MAX}}]$ treats all extremal returns as equivalent — a form of value clipping that *improves* performance for distributional agents but *degrades* DQN.
- Interpreting the discount factor $\gamma$ as a proper probability (rather than a geometric weight) leads to yet another algorithm.

### Well-Behaved Optimization
- KL divergence between categorical distributions is a well-understood, smooth loss.
- Early experiments with KL divergence between continuous densities "were not fruitful" — partly because KL is insensitive to the *values* of outcomes (it only cares about probability mass, not where it sits on the number line).
- The authors conjecture that a closer minimization of the Wasserstein metric should yield even better results.

- *Context*:
	- The "richer predictions" argument connects to a deep theme in RL and representation learning: networks trained on multiple related prediction tasks develop better internal representations. The distributional outputs effectively provide $N$ auxiliary prediction targets at every state-action pair, all of which are semantically meaningful and mutually consistent. This may explain why more atoms monotonically help without saturating network capacity — each additional atom provides additional gradient signal that shapes the learned representation.

---

# MATHEMATICS

The mathematical framework of this paper establishes when distributional Bellman operators converge (policy evaluation) and when they do not (control), then constructs a practical algorithm that sidesteps the convergence difficulties. The derivation chain proceeds from the Wasserstein metric and its properties, through contraction results for $\mathcal{T}^\pi$, through the negative results for $\mathcal{T}$, to the projected categorical algorithm.

### 1. The Wasserstein Metric and Its Properties

The **Wasserstein metric** $d_p$ measures distance between cumulative distribution functions $F$, $G$ over $\mathbb{R}$:

**Wasserstein metric ($d_p$):**

$$
d_p(F, G) := \inf_{U, V} \|U - V\|_p = \left(\int_0^1 |F^{-1}(u) - G^{-1}(u)|^p\,du\right)^{1/p} \tag{W1}
$$

where $U$, $V$ are random variables with CDFs $F$, $G$ and the infimum is over all joint distributions (couplings). The infimum is attained by the **inverse CDF coupling**: $U = F^{-1}(W)$, $V = G^{-1}(W)$ for $W \sim \text{Uniform}[0,1]$.

Three properties are fundamental. For scalar $a$, random variable $A$ independent of $U$, $V$:

**Scalar scaling (P1):**

$$
d_p(aU, aV) \le |a|\,d_p(U, V) \tag{P1}
$$

**Shift invariance (P2):**

$$
d_p(A + U, A + V) \le d_p(U, V) \tag{P2}
$$

**Random scaling (P3):**

$$
d_p(AU, AV) \le \|A\|_p\,d_p(U, V) \tag{P3}
$$

The **Partition Lemma** (Lemma 1) provides a decomposition tool. For a partition $\{A_i\}$ of $\Omega$ (i.e. $A_i(\omega) \in \{0, 1\}$ and exactly one $A_i$ equals 1 for each $\omega$):

**Partition Lemma:**

$$
d_p(U, V) \le \sum_i d_p(A_i U, A_i V) \tag{PL}
$$

The proof constructs the bound by decomposing the sample space: $d_p^p(U, V) = \inf_{U,V} \sum_i \Pr\{A_i = 1\}\,\mathbb{E}[|A_i U - A_i V|^p \mid A_i = 1]$, then noting that the right-hand side of (PL) allows independent optimization within each partition element, which can only increase the total.

### 2. Maximal Wasserstein Metric Over Value Distributions

Let $\mathcal{Z}$ denote the space of value distributions with bounded moments. For $Z_1, Z_2 \in \mathcal{Z}$:

**Maximal Wasserstein metric ($\bar{d}_p$):**

$$
\bar{d}_p(Z_1, Z_2) := \sup_{x, a} d_p(Z_1(x, a), Z_2(x, a)) \tag{W2}
$$

**Lemma 2** establishes that $\bar{d}_p$ is a metric. The triangle inequality follows from applying $d_p$'s triangle inequality pointwise and then taking the supremum.

### 3. Distributional Bellman Operator

The **transition operator** $P^\pi : \mathcal{Z} \to \mathcal{Z}$ maps value distributions forward by one step:

$$
P^\pi Z(x, a) \stackrel{D}{:=} Z(X', A'), \quad X' \sim P(\cdot \mid x, a),\; A' \sim \pi(\cdot \mid X') \tag{T1}
$$

The **distributional Bellman operator** $\mathcal{T}^\pi : \mathcal{Z} \to \mathcal{Z}$ combines reward and discounted transition:

$$
\mathcal{T}^\pi Z(x, a) \stackrel{D}{:=} R(x, a) + \gamma\,P^\pi Z(x, a) \tag{T2}
$$

### 4. Contraction of $\mathcal{T}^\pi$ (Lemma 3)

**$\gamma$-contraction in $\bar{d}_p$:**

$$
\bar{d}_p(\mathcal{T}^\pi Z_1, \mathcal{T}^\pi Z_2) \le \gamma\,\bar{d}_p(Z_1, Z_2) \tag{C1}
$$

**Proof sketch:** For any $(x, a)$:

$$
d_p(\mathcal{T}^\pi Z_1(x,a), \mathcal{T}^\pi Z_2(x,a)) = d_p(R + \gamma P^\pi Z_1, R + \gamma P^\pi Z_2)
$$

Applying (P2) removes the shared $R(x,a)$. Applying (P1) extracts $\gamma$. The transition operator $P^\pi$ mixes over next states, bounded by the supremum:

$$
d_p(P^\pi Z_1(x,a), P^\pi Z_2(x,a)) \le \sup_{x', a'} d_p(Z_1(x', a'), Z_2(x', a'))
$$

Taking $\sup_{x,a}$ of both sides yields (C1).

By **Banach's fixed point theorem**, $\mathcal{T}^\pi$ has a unique fixed point $Z^\pi \in \mathcal{Z}$, and $Z_{k+1} := \mathcal{T}^\pi Z_k$ converges to $Z^\pi$ at rate $\gamma^k$.

### 5. Variance Contraction

The distributional Bellman operator also contracts in variance at rate $\gamma^2$:

**Variance contraction:**

$$
\|\mathbb{V}(\mathcal{T}^\pi Z_1) - \mathbb{V}(\mathcal{T}^\pi Z_2)\|_\infty \le \gamma^2 \|\mathbb{V}(Z_1) - \mathbb{V}(Z_2)\|_\infty \tag{V1}
$$

This follows from independence of $R$ and $P^\pi Z$:

$$
\mathbb{V}(\mathcal{T}^\pi Z_i(x,a)) = \mathbb{V}(R(x,a)) + \gamma^2 \mathbb{V}(P^\pi Z_i(x,a))
$$

The reward variance cancels in the difference, leaving the $\gamma^2$ factor on the transition-variance term.

### 6. Control Setting — Expectations Converge (Lemma 4)

The distributional optimality operator $\mathcal{T}$ contracts expectations:

$$
\|\mathbb{E}[\mathcal{T}\,Z_1] - \mathbb{E}[\mathcal{T}\,Z_2]\|_\infty \le \gamma\,\|\mathbb{E}[Z_1] - \mathbb{E}[Z_2]\|_\infty \tag{C2}
$$

This follows by linearity of expectation: $\mathbb{E}[\mathcal{T}_D Z] = \mathcal{T}_E \mathbb{E}[Z]$, where $\mathcal{T}_D$ is the distributional operator and $\mathcal{T}_E$ is the standard optimality operator, which is a known $\gamma$-contraction.

### 7. Control Setting — Distributional Convergence (Theorem 1)

Convergence is only to the set of nonstationary optimal value distributions $\mathcal{Z}^{**}$:

$$
\lim_{k \to \infty} \inf_{Z^{**} \in \mathcal{Z}^{**}} d_p(Z_k(x, a), Z^{**}(x, a)) = 0 \quad \forall\, x, a \tag{CT}
$$

The proof constructs a hierarchy of "solved" state sets $\mathcal{X}_{k,i}$:

$$
\mathcal{X}_{k,0} := \mathcal{X}_k = \left\{x : Q^*(x, \pi^*(x)) - \max_{a \ne \pi^*(x)} Q^*(x, a) > 2\varepsilon_k\right\}
$$

where $\varepsilon_k := \gamma^k B$ and $B := 2\sup_{Z \in \mathcal{Z}} \|Z\|_\infty < \infty$. Recursively:

$$
\mathcal{X}_{k,i} := \{x \in \mathcal{X}_k : P(\mathcal{X}_{k-1,i-1} \mid x, \pi^*(x)) \ge 1 - \delta\}
$$

**Lemma 5** shows every state eventually enters $\mathcal{X}_k$ (because $\mathcal{A}$ is finite, the action gap $\Delta(x) > 0$). **Lemma 6** shows by induction that every state eventually enters $\mathcal{X}_{k,i}$ for any $i$.

The proof then uses the Partition Lemma to separate solved and unsolved states:

$$
d_p(W_{k+1}(x), W^*(x)) \le \gamma\,d_p(S_i^k W_k(X'), S_i^k W^*(X')) + \gamma\delta B
$$

By induction on $i$:

$$
d_p(W_{k+i}(x), W^*(x)) \le \gamma^i B + \frac{\delta B}{1 - \gamma}
$$

Taking $\delta$ small, $i$ large, then $k$ large makes this arbitrarily small.

### 8. Non-Contraction of $\mathcal{T}$ (Proposition 1)

$\mathcal{T}$ is demonstrated to *not* be a contraction via a 2-state MDP. States $x_1$ (transitions to $x_2$) and $x_2$ (terminal, two actions). Under action $a_2$: reward $\varepsilon \pm 1$ with equal probability. Construct $Z$ such that:

$$
\bar{d}_1(Z, Z^*) = 2\varepsilon \quad \text{but} \quad \bar{d}_1(\mathcal{T}\,Z, \mathcal{T}\,Z^*) > 1 > 2\varepsilon
$$

for small $\varepsilon$, because $\mathcal{T}$ flips the greedy action at $x_1$ from $a_2$ to $a_1$.

### 9. Parametric Distribution and Projection

The discrete distribution has $N$ atoms $\{z_i\}_{i=0}^{N-1}$ on support $[V_{\text{MIN}}, V_{\text{MAX}}]$:

**Atom locations:**

$$
z_i = V_{\text{MIN}} + i\,\Delta z, \quad \Delta z = \frac{V_{\text{MAX}} - V_{\text{MIN}}}{N - 1} \tag{A1}
$$

**Atom probabilities (softmax):**

$$
p_i(x, a) = \frac{e^{\theta_i(x,a)}}{\sum_{j=0}^{N-1} e^{\theta_j(x,a)}} \tag{A2}
$$

**Projected Bellman update:** Given sample $(x, a, r, x')$, target atoms $\hat{\mathcal{T}}z_j = [r + \gamma z_j]_{V_{\text{MIN}}}^{V_{\text{MAX}}}$:

$$
(\Phi\,\hat{\mathcal{T}}\,Z_\theta(x, a))_i = \sum_{j=0}^{N-1} \left[1 - \frac{|\hat{\mathcal{T}}z_j - z_i|}{\Delta z}\right]_0^1 p_j(x', \pi(x')) \tag{P}
$$

**Cross-entropy loss:**

$$
L_{x,a}(\theta) = D_{\text{KL}}\!\left(\Phi\,\hat{\mathcal{T}}\,Z_{\tilde{\theta}}(x,a) \;\big\|\; Z_\theta(x,a)\right) = -\sum_{i=0}^{N-1} m_i \log p_i(x, a) + \text{const.} \tag{L}
$$

where $m_i = (\Phi\,\hat{\mathcal{T}}\,Z_{\tilde{\theta}}(x,a))_i$ are the projected target probabilities and $\tilde{\theta}$ is the frozen target network parameter.

### 10. Sample Wasserstein Loss is Biased (Lemma 7 / Proposition 5)

For a mixture $P = P_I$ where $I$ is a random index independent of $\{P_i\}$ and $Q$:

**Sample Wasserstein bound:**

$$
d_p(P, Q) \le \mathbb{E}_{i \sim I}\,d_p(P_i, Q) \tag{SW}
$$

with strict inequality in general, and:

$$
\nabla_Q d_p(P_I, Q) \ne \mathbb{E}_{i \sim I}\,\nabla_Q d_p(P_i, Q)
$$

This means stochastic gradient descent on the sample Wasserstein loss does not minimize the true Wasserstein loss, justifying the use of the categorical projection with cross-entropy instead.

---

### **1. Problem:**

The paper addresses the question of whether the full return distribution — rather than just its expected value — should be a central object in reinforcement learning. Prior work used the value distribution only for specialized purposes (risk-sensitivity, uncertainty modelling, theoretical analysis). The gap is the absence of both theoretical understanding and practical algorithms that leverage the value distribution to improve *general* approximate RL performance.

### **2. Setup:**

The paper assumes a time-homogeneous MDP $(\mathcal{X}, \mathcal{A}, R, P, \gamma)$ with random rewards, finite action space (for theoretical results), and possibly infinite state space. The algorithmic setup uses the DQN architecture with convolutional networks processing raw Atari frames, $\varepsilon$-greedy exploration, experience replay, and target networks. The value distribution is parametrized as a discrete categorical distribution over $N$ equally-spaced atoms on a bounded support $[V_{\text{MIN}}, V_{\text{MAX}}]$.

### **3. Key Idea:**

The core contribution is twofold: (1) a theoretical analysis showing that the distributional Bellman operator is a $\gamma$-contraction in the Wasserstein metric for policy evaluation but *not* a contraction for control, and (2) a practical algorithm (C51) that parametrizes value distributions as discrete categoricals, projects Bellman updates onto this fixed support, and minimizes cross-entropy loss. The insight is that learning distributions — even when acting greedily on expectations — produces dramatically better approximate RL through improved optimization, multimodality preservation, and richer gradient signals.

### **4. Assumptions:**

**Explicit:**
- Time-homogeneous MDP with stationary policies
- Independence of reward, transition, and next-state return
- Bounded moments for all value distributions
- Finite action space (for Theorem 1); finite state space (for uniform convergence)
- Fixed bounded support $[V_{\text{MIN}}, V_{\text{MAX}}]$ for the discrete distribution

**Implicit:**
- The DQN architecture has sufficient capacity for $N \times |\mathcal{A}|$ atom probabilities
- The projection $\Phi$ does not introduce pathological distortions (no formal error analysis)
- Cross-entropy is a reasonable surrogate for Wasserstein minimization (theory uses Wasserstein, practice uses KL)
- $\varepsilon$-greedy exploration suffices to observe distributional structure
- Experience replay and target networks interact benignly with distributional objectives
- Fixed support from 5 training games generalizes across all 57 Atari games
- Tabular theoretical results transfer to the function approximation setting

### **5. Limitation:**

The theory-practice gap is the most significant limitation. The contraction analysis is conducted in the Wasserstein metric, but the algorithm minimizes KL divergence — no formal bridge connects the two. The projection operator $\Phi$ has no accompanying error analysis. The non-contraction of the optimality operator is left as a theoretical warning with no algorithmic mitigation. Practically, the method requires hyperparameters ($N$, $V_{\text{MIN}}$, $V_{\text{MAX}}$) that are tuned on only 5 games and applied uniformly. Scalability to continuous action spaces is unaddressed. The method does not leverage the distributional information for exploration — action selection remains $\varepsilon$-greedy over expected values, leaving a major potential benefit untapped.

### **6. Relevance & Open Questions:**

This paper is foundational for *distributional reinforcement learning* and directly relevant to risk-sensitive RL. The value distribution is precisely the object needed to reason about risk: CVaR, variance constraints, and other risk measures are all functionals of the return distribution. The paper does not pursue risk-sensitive policies but provides the infrastructure for doing so. Key open questions for risk-sensitive extensions:
- Can the distributional instability in the control setting be tamed for risk-sensitive objectives (where the optimality criterion is itself distributional)?
- How should $V_{\text{MIN}}$ and $V_{\text{MAX}}$ be chosen for risk-sensitive applications where tail behaviour matters most?
- Can distributional information drive *risk-aware exploration* rather than just risk-aware evaluation?
- The quantile-based successors (QR-DQN, IQN) avoid the fixed-support limitation — how do they compare for risk-sensitive control?
- What is the relationship between distributional RL and the log-utility / Kelly criterion framework, where the objective is itself a nonlinear functional of the return distribution?

---

### Integration:

* **Problem:** This paper reframes the value distribution from a specialized tool for risk-sensitivity into a foundational object for approximate RL. For research in risk-sensitive RL, this is doubly important: it provides both the theoretical grounding (contraction in Wasserstein, instability characterization in control) and the practical machinery (categorical parametrization, projected Bellman updates) needed to build risk-sensitive agents that learn value distributions end-to-end. The distributional perspective is the natural *language* in which risk-sensitive RL should be formulated — risk measures like CVaR, entropic risk, and variance constraints are all defined over the return distribution that this paper learns to approximate.

* **Limitation:** The most relevant limitation for risk-sensitive applications is the fixed, uniformly-spaced support $[V_{\text{MIN}}, V_{\text{MAX}}]$. Risk-sensitive objectives often depend critically on tail behaviour — precisely the region where a uniform discretization with limited atoms is least accurate. The theory-practice gap (Wasserstein theory, KL practice) also matters for risk: the Wasserstein metric is geometry-aware and respects the *magnitude* of returns, while KL divergence does not — yet the algorithm uses KL. For risk-sensitive extensions, the quantile-based successors (QR-DQN, IQN) may be more natural since quantile regression directly targets the inverse CDF, which is the natural parametrization for CVaR and related risk measures.
