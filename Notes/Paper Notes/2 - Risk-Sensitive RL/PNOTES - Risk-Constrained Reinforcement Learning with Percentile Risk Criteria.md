[[Risk-Constrained Reinforcement Learning with Percentile Risk Criteria.pdf]]

# **Abstract:**

The paper presents reinforcement learning algorithms for **risk-constrained Markov decision processes (MDPs)**, where risk is represented via a **chance constraint** or a constraint on the **Conditional Value-at-Risk (CVaR)** of the cumulative cost. These are collectively termed **percentile risk-constrained MDPs**.

- The approach derives a formula for computing the gradient of the **Lagrangian function** for percentile risk-constrained MDPs.
- Two classes of algorithms are devised:
	- **Policy gradient (PG)** algorithms that update parameters after observing entire trajectories.
	- **Actor-critic (AC)** algorithms that update parameters incrementally at each time step.
- Both algorithm classes:
	1. Estimate the gradient of the Lagrangian.
	2. Update the policy in the descent direction.
	3. Update the Lagrange multiplier in the ascent direction.
- Convergence to **locally optimal policies** is formally proved for all algorithms.
- Effectiveness is demonstrated on an **optimal stopping problem** and a **personalized ad-recommendation** application.

- *Context*:
	- This paper fills a gap in the risk-sensitive RL literature by addressing *constrained* formulations rather than unconstrained risk-sensitive objectives. Where prior work (e.g., exponential utility, mean-variance) optimized a single risk-sensitive criterion, this paper treats risk as a *constraint* — minimize expected cost subject to a bound on tail risk. This constrained formulation is more natural in applications where a practitioner has a clear risk budget (e.g., "CVaR must not exceed $\beta$") rather than a preference for a specific risk-return tradeoff. The constrained approach also decouples the performance objective from the risk measure, allowing each to be specified independently. See [[PNOTES - Policy Gradient for Coherent Risk Measures]] for the unconstrained coherent risk framework that this paper's CVaR constraint draws upon.

---

# 1. Introduction

The paper motivates the shift from risk-neutral to risk-sensitive RL by identifying a fundamental limitation: the standard expected cumulative cost criterion ignores events of small probability and high consequence. In domains where catastrophic tail events matter — finance, robotics, logistics — risk-awareness is essential.

## Risk-Sensitive MDPs

- The **exponential risk metric** $(1/\gamma)\mathbb{E}[\exp(\gamma Z)]$ (Howard and Matheson, 1972) was among the earliest, but the selection of $\gamma$ is often challenging.
- Alternative approaches include:
	- Maximization of strictly concave functionals of terminal state distributions (Collins, 1997).
	- Percentile performance maximization (Wu and Lin, 1999; Boda et al., 2004; Filar et al., 1995).
	- Variance-related metrics (Sobel, 1982; Filar et al., 1989).
- More recently, **Value-at-Risk (VaR)** and **Conditional Value-at-Risk (CVaR)** have emerged as promising alternatives that directly quantify tail-risk.

- *Context*:
	- VaR and CVaR address different aspects of tail risk. VaR answers "what is the maximum cost at a given confidence level?" — useful when there is a well-defined failure threshold. CVaR answers "what is the expected cost in the worst $\alpha$-fraction of outcomes?" — useful when the *magnitude* of tail losses matters, not just whether they occur. CVaR is a **coherent risk measure** (Artzner et al., 1999) while VaR is not, which gives CVaR superior mathematical properties (convexity, in particular).

## Risk-Sensitive RL

- RL methods (policy gradient, actor-critic) are necessary for large-scale risk-sensitive MDPs where exact solutions are computationally infeasible.
- Prior risk-sensitive RL work:
	- Exponential utility (Borkar, 2001, 2002)
	- Variance-related measures (Tamar et al., 2012; Prashanth and Ghavamzadeh, 2013)
	- CVaR-based formulations (Morimura et al., 2010; Tamar et al., 2015; Petrik and Subramanian, 2012)
	- Nested CVaR formulations (Tallec, 2007; Shapiro et al., 2013)

## Risk-Constrained RL and Contributions

- Despite the large literature on risk-sensitive MDPs and RL, **risk-constrained** formulations have been largely unaddressed, with few exceptions (Chow and Pavone, 2013; Borkar and Jain, 2014).
- Constrained formulations naturally arise in engineering, finance, and logistics, and provide a principled approach to multi-objective problems.

- *Context*:
	- The distinction between risk-sensitive and risk-constrained is important. Risk-sensitive RL optimizes a single objective that blends expected cost with a risk measure (e.g., minimize $\mathbb{E}[Z] + \lambda \cdot \text{CVaR}(Z)$ with a fixed $\lambda$). Risk-constrained RL separates the two: minimize expected cost *subject to* a bound on risk. The constrained formulation is often preferred in practice because the constraint threshold $\beta$ has a direct operational interpretation (e.g., "the 95th percentile cost must not exceed $\beta$"), whereas the tradeoff parameter $\lambda$ in the unconstrained formulation has no such direct meaning.

The paper's **four contributions**:

1. **Formulation**: Two risk-constrained MDP problems (CVaR-constrained and chance-constrained) are formulated. A Lagrangian relaxation leads to a Bellman optimality condition on an **augmented MDP** $\bar{M}$ whose state tracks both the original MDP state and the cumulative constraint cost.
2. **Trajectory-based PG algorithm**: A policy gradient algorithm with unbiased gradient estimation under Monte Carlo sampling, with convergence to local optima via ODE analysis.
3. **Actor-critic algorithms**: Incremental algorithms using TD-learning on the augmented MDP, with SPSA for the VaR parameter and multi-timescale convergence proofs.
4. **Empirical validation**: Experiments on an optimal stopping problem and a personalized ad-recommendation system.

---

# 2. Preliminaries

## 2.1 Notation

The paper considers a **finite MDP** $M = (\mathcal{X}, \mathcal{A}, C, D, P, P_0)$ where:

- $\mathcal{X} = \{1, \ldots, n, x_{\text{Tar}}\}$: state space with a recurrent **target state** $x_{\text{Tar}}$.
- $\mathcal{A} = \{1, \ldots, m\}$: action space.
- $C(x, a)$: cost function with $|C(x, a)| \leq C_{\max}$.
- $D(x, a)$: **constraint cost function** with $|D(x, a)| \leq D_{\max}$.
- $P(\cdot|x, a)$: transition probability distribution.
- $P_0(\cdot)$: initial state distribution, assumed to be $P_0 = \mathbf{1}\{x = x_0\}$ for a given $x_0$.
- Policies are stationary and parameterized: $\mu(\cdot|x; \theta)$ with $\theta \in \Theta \subseteq \mathbb{R}^\kappa$.

- *Context*:
	- The MDP has *two* cost functions: $C(x, a)$ for the primary objective and $D(x, a)$ for the constraint. This dual-cost structure is the hallmark of **Constrained MDPs (CMDPs)** (Altman, 1999). The agent minimizes expected cumulative $C$ while bounding the risk of cumulative $D$. In many applications, $C = D$ (the same cost is both optimized and constrained), but the general formulation allows them to differ.

The **$\gamma$-discounted occupation measure** is defined as:

$$
d_\gamma^\mu(x|x_0) = (1 - \gamma) \sum_{k=0}^{\infty} \gamma^k P(x_k = x | x_0; \mu)
$$

and the state-action occupation measure as $\pi_\gamma^\mu(x, a|x_0) = d_\gamma^\mu(x|x_0)\mu(a|x)$. This is a proper probability distribution when state and action spaces are finite (Theorem 3.1 of Altman, 1999).

When $\gamma = 1$, the occupation measures are $d^\mu(x|x_0) = \sum_{t=0}^{\infty} P(x_t = x|x_0; \mu)$ and $\pi^\mu(x, a|x_0) = d^\mu(x|x_0)\mu(a|x)$, representing total visiting probabilities (not necessarily normalized).

### Transient MDP

**Definition 1:** An MDP is **transient** if:
1. $\sum_{k=0}^{\infty} P(x_k = x | x_0, \mu) < \infty$ for every $x \in \mathcal{X}' = \mathcal{X} \setminus \{x_{\text{Tar}}\}$ and every stationary policy $\mu$.
2. $P(x_{\text{Tar}} | x_{\text{Tar}}, a) = 1$ for every $a \in \mathcal{A}$ (the target state is absorbing).

**Assumption 2 (Bounded First-Hitting Time):** The first-hitting time $T_{\mu,x}$ of $x_{\text{Tar}}$ is bounded almost surely over all stationary policies $\mu$ and all initial states $x \in \mathcal{X}$, by $T$: $T_{\mu,x} \leq T$ a.s.

- *Context*:
	- Transience ensures well-defined occupation measures. The bounded first-hitting time (Assumption 2) is stronger than simple transience — it guarantees that *every* trajectory terminates within $T$ steps regardless of the policy. This is justified by the fact that practical RL algorithms use a finite time-out. The bounded stopping time can be reconciled with time-stationary transitions by augmenting the state space with a time counter.

### Cost and Value Functions

The cumulative discounted **cost** and **constraint cost** random variables are:

$$
G^\theta(x) = \sum_{k=0}^{T-1} \gamma^k C(x_k, a_k) \bigg| x_0 = x, \mu(\cdot|\cdot, \theta)
$$

$$
J^\theta(x) = \sum_{k=0}^{T-1} \gamma^k D(x_k, a_k) \bigg| x_0 = x, \mu(\cdot|\cdot, \theta)
$$

The value and action-value functions are $V^\theta(x) = \mathbb{E}[G^\theta(x)]$ and $Q^\theta(x, a) = \mathbb{E}[G^\theta(x, a)]$.

### VaR and CVaR

**Value-at-Risk** at confidence level $\alpha \in (0, 1)$:

$$
\text{VaR}_\alpha(Z) = \min\{z \mid F_Z(z) \geq \alpha\}
$$

**Conditional Value-at-Risk** (Rockafellar and Uryasev, 2000):

$$
\text{CVaR}_\alpha(Z) := \min_{\nu \in \mathbb{R}} \left\{\nu + \frac{1}{1 - \alpha}\mathbb{E}[(Z - \nu)^+]\right\} \tag{1}
$$

- *Context*:
	- The CVaR definition via minimization over $\nu$ is the key to making CVaR constraints tractable. The optimal $\nu$ in the minimization equals $\text{VaR}_\alpha(Z)$ (Rockafellar and Uryasev, 2000, 2002), so CVaR can be computed by jointly optimizing over both the policy parameter $\theta$ and the auxiliary VaR parameter $\nu$. This avoids having to explicitly compute quantiles — a significant computational advantage that the entire algorithmic framework exploits.

## 2.2 Problem Statement

### CVaR-Constrained Optimization

For $\gamma \in (0, 1)$, confidence level $\alpha \in (0, 1)$, and cost tolerance $\beta \in \mathbb{R}$:

$$
\min_\theta V^\theta(x_0) \quad \text{subject to} \quad \text{CVaR}_\alpha(J^\theta(x_0)) \leq \beta \tag{2}
$$

Using the auxiliary function $H_\alpha(Z, \nu) := \nu + \frac{1}{1-\alpha}\mathbb{E}[(Z - \nu)^+]$, this is reformulated as:

$$
\min_{\theta, \nu} V^\theta(x_0) \quad \text{subject to} \quad H_\alpha(J^\theta(x_0), \nu) \leq \beta \tag{3}
$$

- *Context*:
	- The equivalence between (2) and (3) is shown by a feasibility argument: any feasible $\theta$ for (2) yields a feasible $(\theta, \nu)$ for (3) by setting $\nu = \text{VaR}_\alpha(J^\theta(x_0))$, and conversely, feasibility of (3) implies feasibility of (2) since $\text{CVaR}_\alpha \leq H_\alpha$ for all $\nu$. The reformulation introduces $\nu$ as an additional optimization variable, but this is a small price for making the constraint smooth and differentiable almost everywhere.

### Chance-Constrained Optimization

For $\gamma = 1$, confidence level $\beta \in (0, 1)$, and cost tolerance $\alpha \in \mathbb{R}$:

$$
\min_\theta V^\theta(x_0) \quad \text{subject to} \quad P(J^\theta(x_0) \geq \alpha) \leq \beta \tag{4}
$$

- *Context*:
	- The chance-constrained formulation uses $\gamma = 1$ (undiscounted) because in engineering applications where chance constraints ensure safety, future threats are as important as current ones. The CVaR-constrained formulation uses $\gamma \in (0, 1)$ because in financial applications the emphasis is on near-term costs.

### Technical Assumptions

**Assumption 3 (Differentiability):** $\mu(a|x; \theta)$ is continuously differentiable in $\theta$ and $\nabla_\theta \mu(a|x; \theta)$ is Lipschitz in $\theta$.

**Assumption 4 (Strict Feasibility / Slater's Condition):** There exists a transient policy $\mu(\cdot|x; \theta)$ such that $H_\alpha(J^\theta(x_0), \nu) < \beta$ (CVaR case) or $P(J^\theta(x_0) \geq \alpha) < \beta$ (chance-constrained case).

- *Context*:
	- Strict feasibility (Slater's condition) is essential for guaranteeing the existence of a local saddle point for the Lagrangian. Without it, the Lagrange multiplier might grow unboundedly, and the duality gap might not close. Operationally, it means there exists at least one policy that strictly satisfies the risk constraint — a reasonable requirement, since if no policy can satisfy the constraint, the problem is infeasible.

## 2.3 Lagrangian Approach and Reformulation

The constrained problem (3) is transformed via **Lagrangian relaxation** into:

$$
\max_{\lambda \geq 0} \min_{\theta, \nu} \left(L(\nu, \theta, \lambda) := V^\theta(x_0) + \lambda\left(H_\alpha(J^\theta(x_0), \nu) - \beta\right)\right) \tag{5}
$$

where $\lambda$ is the Lagrange multiplier.

- $L(\nu, \theta, \lambda)$ is linear in $\lambda$ and $H_\alpha(J^\theta(x_0), \nu)$ is continuous in $\nu$.
- The **saddle point theorem** (Bertsekas, 1999) guarantees that a local saddle point $(\nu^*, \theta^*, \lambda^*)$ yields a locally optimal policy $\theta^*$ for the original constrained problem.

**Definition 5 (Local Saddle Point):** A point $(\nu^*, \theta^*, \lambda^*)$ is a local saddle point of $L$ if for some $r > 0$:

$$
L(\nu, \theta, \lambda^*) \geq L(\nu^*, \theta^*, \lambda^*) \geq L(\nu^*, \theta^*, \lambda)
$$

for all $(\theta, \nu) \in \Theta \times [-D_{\max}/(1-\gamma), D_{\max}/(1-\gamma)] \cap B_{(\theta^*, \nu^*)}(r)$ and all $\lambda \geq 0$.

- *Context*:
	- The Lagrangian reformulation is the methodological backbone of the paper. By converting the constrained optimization into an unconstrained saddle-point problem, the authors can apply gradient-based methods: descend in $(\theta, \nu)$ to minimize cost, ascend in $\lambda$ to penalize constraint violations. The local nature of the saddle point is unavoidable since the Lagrangian is non-convex in $\theta$ (due to the policy parameterization), so global optimality is not guaranteed.

It has been shown (Ott, 2010; Bäuerle and Ott, 2011) that the optimal policy for CVaR-constrained optimization is **deterministic and history-dependent**, but the dependence is only on the current time step $k$, current state $x_k$, and accumulated discounted constraint cost $\sum_{i=0}^{k} \gamma^i D(x_i, a_i)$. This motivates the augmented state space used in the actor-critic algorithms.

---

# 3. A Trajectory-based Policy Gradient Algorithm

The PG algorithm operates by descending in $(\theta, \nu)$ and ascending in $\lambda$ using the gradients of $L(\nu, \theta, \lambda)$:

### Gradient w.r.t. $\theta$ (Eq. 7)

$$
\nabla_\theta L(\nu, \theta, \lambda) = \nabla_\theta V^\theta(x_0) + \frac{\lambda}{1 - \alpha} \nabla_\theta \mathbb{E}[(J^\theta(x_0) - \nu)^+] \tag{7}
$$

### Sub-gradient w.r.t. $\nu$ (Eq. 8)

$$
\partial_\nu L(\nu, \theta, \lambda) \ni \lambda\left(1 - \frac{1}{1 - \alpha} P(J^\theta(x_0) \geq \nu)\right) \tag{8}
$$

### Gradient w.r.t. $\lambda$ (Eq. 9)

$$
\nabla_\lambda L(\nu, \theta, \lambda) = \nu + \frac{1}{1 - \alpha}\mathbb{E}[(J^\theta(x_0) - \nu)^+] - \beta \tag{9}
$$

- *Context*:
	- The sub-gradient in (8) involves an indicator function $\mathbf{1}\{J^\theta(x_0) \geq \nu\}$, which is non-differentiable. This is inherent to the CVaR formulation: the $(Z - \nu)^+$ function has a kink at $Z = \nu$. The paper handles this by working with sub-differentials and differential inclusions in the convergence analysis. The $\ni$ notation in (8) emphasizes that this is a *member* of the sub-gradient set, not the unique gradient.

### Algorithm 1: Trajectory-based PG

At each iteration, the algorithm generates $N$ trajectories $\{\xi_{j,k}\}_{j=1}^N$ from the current policy $\theta_k$. Let $\xi = \{x_0, a_0, c_0, \ldots, x_{T-1}, a_{T-1}, c_{T-1}, x_T\}$ be a trajectory with:
- Cost: $G(\xi) = \sum_{k=0}^{T-1} \gamma^k C(x_k, a_k)$
- Constraint cost: $J(\xi) = \sum_{k=0}^{T-1} \gamma^k D(x_k, a_k)$
- Log-probability gradient: $\nabla_\theta \log P_\theta(\xi) = \sum_{k=0}^{T-1} \nabla_\theta \log \mu(a_k|x_k; \theta)$

The update rules are:

**$\nu$ Update:**

$$
\nu_{k+1} = \Gamma_{\mathcal{N}}\left[\nu_k - \zeta_3(k)\left(\lambda_k - \frac{\lambda_k}{(1-\alpha)N}\sum_{j=1}^{N} \mathbf{1}\{J(\xi_{j,k}) \geq \nu_k\}\right)\right]
$$

**$\theta$ Update:**

$$
\theta_{k+1} = \Gamma_\Theta\left[\theta_k - \zeta_2(k)\left(\frac{1}{N}\sum_{j=1}^{N} \nabla_\theta \log P_\theta(\xi_{j,k}) G(\xi_{j,k}) + \frac{\lambda_k}{(1-\alpha)N}\sum_{j=1}^{N} \nabla_\theta \log P_\theta(\xi_{j,k})(J(\xi_{j,k}) - \nu_k)\mathbf{1}\{J(\xi_{j,k}) \geq \nu_k\}\right)\right]
$$

**$\lambda$ Update:**

$$
\lambda_{k+1} = \Gamma_\Lambda\left[\lambda_k + \zeta_1(k)\left(\nu_k - \beta + \frac{1}{(1-\alpha)N}\sum_{j=1}^{N}(J(\xi_{j,k}) - \nu_k)\mathbf{1}\{J(\xi_{j,k}) \geq \nu_k\}\right)\right]
$$

where $\Gamma_\Theta$, $\Gamma_{\mathcal{N}}$, $\Gamma_\Lambda$ are projection operators onto the compact constraint sets $\Theta$, $[-D_{\max}/(1-\gamma), D_{\max}/(1-\gamma)]$, and $[0, \lambda_{\max}]$ respectively.

- *Context*:
	- The $\theta$ update has a natural interpretation. The first term is the standard policy gradient (minimize expected cost). The second term is the *risk gradient*: it uses the same likelihood-ratio structure but weights only trajectories whose constraint cost $J(\xi)$ exceeds $\nu$ (the VaR estimate), and only the excess $(J(\xi) - \nu)$. This means the risk gradient is driven entirely by the *tail* of the constraint cost distribution — trajectories that violate the current VaR threshold contribute to the gradient, while safe trajectories do not. The Lagrange multiplier $\lambda$ controls how strongly the risk gradient pulls the policy away from the expected-cost descent direction.

### Step-Size Conditions (Assumption 6)

The step sizes $\{\zeta_1(k)\}$, $\{\zeta_2(k)\}$, $\{\zeta_3(k)\}$ satisfy:

$$
\sum_k \zeta_i(k) = \infty, \quad \sum_k \zeta_i(k)^2 < \infty \quad (i = 1, 2, 3)
$$

$$
\zeta_1(k) = o(\zeta_2(k)), \quad \zeta_2(k) = o(\zeta_3(k))
$$

This creates a **three-timescale** stochastic approximation:
- **Fastest** ($\zeta_3$): $\nu$ update (VaR parameter).
- **Intermediate** ($\zeta_2$): $\theta$ update (policy).
- **Slowest** ($\zeta_1$): $\lambda$ update (Lagrange multiplier).

- *Context*:
	- Multi-timescale stochastic approximation is the standard technique for simultaneous optimization of interdependent parameters. The idea is that from the perspective of the slow variable, the fast variables have already converged to their quasi-static equilibrium. This decouples the convergence analysis: analyze $\nu$ convergence with $(\theta, \lambda)$ frozen, then $\theta$ convergence with $\lambda$ frozen and $\nu$ at its equilibrium, then $\lambda$ convergence with $(\theta, \nu)$ at their equilibria. The theory is due to Borkar (2008).

### Convergence (Theorem 7)

Under Assumptions 2–6, the sequence of policy updates in Algorithm 1 converges almost surely to a locally optimal policy $\theta^*$ for the CVaR-constrained optimization problem.

**Proof overview:**

1. **Multi-timescale convergence:** Each discrete update $(\nu_k, \theta_k, \lambda_k)$ converges a.s. to a stationary point $(\nu^*, \theta^*, \lambda^*)$ of the corresponding continuous-time ODE system, at the respective timescale rates.
2. **Lyapunov stability:** Using $L(\nu, \theta, \lambda)$ itself as a Lyapunov function, the continuous-time system is shown to be locally asymptotically stable at $(\nu^*, \theta^*, \lambda^*)$.
3. **Saddle point:** Since $L$ serves as a Lyapunov function for both the descent ($\theta, \nu$) and ascent ($\lambda$) directions, the stationary point is a local saddle point. The saddle point theorem then implies $\theta^*$ is locally optimal for the original constrained problem.

- *Context*:
	- The proof technique is standard in the stochastic approximation literature (Bhatnagar et al., 2009; Borkar, 2008). The key insight is that the Lagrangian simultaneously serves as both the objective (for the primal minimization) and the Lyapunov function (for the stability analysis). This dual role is what makes Lagrangian relaxation so powerful for constrained optimization in the stochastic approximation framework. A subtlety addressed in the paper is the handling of **spurious fixed points**: if $\lambda^*$ converges to $\lambda_{\max}$ (the boundary of the projection set), the algorithm may not find a saddle point. This is resolved by incrementally doubling $\lambda_{\max}$ whenever convergence to the boundary is detected.

---

# 4. Actor-Critic Algorithms

The PG algorithm's unit of observation is an entire trajectory, which can produce high-variance gradient estimates. The actor-critic algorithms address this by:
- Approximating the value function with linear function approximation.
- Updating parameters incrementally after each state-action transition.

Two AC variants are presented:
1. **SPSA-based:** Fully incremental — updates $v$, $\theta$, $\nu$, $\lambda$ at each time step.
2. **Semi-trajectory-based:** Updates $v$, $\theta$, $\lambda$ at each time step, but updates $\nu$ only at the end of each trajectory.

### The Augmented MDP $\bar{M}$

Given the original MDP $M$ and Lagrange multiplier $\lambda$, the **augmented MDP** $\bar{M} = (\bar{\mathcal{X}}, \bar{\mathcal{A}}, \bar{C}_\lambda, \bar{P}, \bar{P}_0)$ is defined as:

- $\bar{\mathcal{X}} = \mathcal{X} \times \mathcal{S}$: augmented state space.
- $\bar{\mathcal{A}} = \mathcal{A}$: same action space.
- Initial state: $\bar{P}_0(x, s) = P_0(x)\mathbf{1}\{s_0 = s\}$ with $s_0 = \nu$.

**Augmented cost:**

$$
\bar{C}_\lambda(x, s, a) = \begin{cases} \frac{\lambda(-s)^+}{1 - \alpha} & \text{if } x = x_{\text{Tar}} \\ C(x, a) & \text{otherwise} \end{cases}
$$

**Augmented transition:**

$$
\bar{P}(x', s' | x, s, a) = \begin{cases} P(x'|x, a)\mathbf{1}\{s' = (s - D(x, a))/\gamma\} & \text{if } x \in \mathcal{X}' \\ \mathbf{1}\{x' = x_{\text{Tar}}, s' = 0\} & \text{if } x = x_{\text{Tar}} \end{cases}
$$

- *Context*:
	- The augmented state $s$ tracks the "remaining constraint budget": it starts at $s_0 = \nu$ (the current VaR estimate) and decreases by $D(x_k, a_k)/\gamma^k$ at each step. At termination, $s_{\text{Tar}} < 0$ means the trajectory's constraint cost exceeded $\nu$, contributing a penalty $\lambda(-s_{\text{Tar}})^+/(1-\alpha)$ at the target state. This construction transforms the CVaR Lagrangian problem into a standard (unconstrained) MDP problem on $\bar{M}$, enabling the use of standard TD-learning and policy gradient machinery. The key identity is:

$$
\sum_{k=0}^{T} \gamma^k \bar{C}_\lambda(x_k, s_k, a_k) = G^\theta(x) + \frac{\lambda}{1-\alpha}(J^\theta(x) - s)^+ \tag{25}
$$

This means $V^\theta(x_0, \nu) = \nabla_\theta L(\nu, \theta, \lambda)$, which connects the augmented MDP's value function directly to the Lagrangian gradient.

## 4.1 Gradient w.r.t. Policy Parameters $\theta$

By the **policy gradient theorem** applied to the augmented MDP $\bar{M}$:

$$
\nabla_\theta L(\nu, \theta, \lambda) = \nabla_\theta V^\theta(x_0, \nu) = \frac{1}{1 - \gamma} \sum_{x, s, a} \pi_\gamma^\theta(x, s, a | x_0, \nu) \nabla \log \mu(a|x, s; \theta) Q^\theta(x, s, a) \tag{26}
$$

The critic uses **linear approximation**: $V^\theta(x, s) \approx v^\top \phi(x, s) = \tilde{V}^{\theta, v}(x, s)$, where $\phi(\cdot) \in \mathbb{R}^{\kappa_1}$ is a feature vector.

The **TD error** in the augmented MDP:

$$
\delta_k(v_k) = \bar{C}_{\lambda_k}(x_k, s_k, a_k) + \gamma v_k^\top \phi(x_{k+1}, s_{k+1}) - v_k^\top \phi(x_k, s_k) \tag{18}
$$

provides an unbiased estimate of the policy gradient: $\frac{1}{1-\gamma}\nabla \log \mu(a_k|x_k, s_k; \theta) \cdot \delta_k$ is an unbiased estimator of $\nabla_\theta L(\nu, \theta, \lambda)$.

**Assumption 9 (Independent Basis Functions):** The basis functions $\{\phi^{(i)}\}_{i=1}^{\kappa_1}$ are linearly independent, $\kappa_1 \leq n$, $\Phi$ is full column rank, and $\Phi v \neq \mathbf{e}$ for every $v$.

### Critic Convergence (Theorem 10)

The **Bellman operator** on the augmented MDP:

$$
B^\theta[V](x, s) = \sum_a \mu(a|x, s; \theta)\left[\bar{C}_\lambda(x, s, a) + \sum_{x', s'} \gamma \bar{P}(x', s'|x, s, a)V(x', s')\right]
$$

The critic converges to $v^*$, the minimizer of the Bellman residual $\|B^\theta[\Phi v] - \Phi v\|_{d_\gamma^\theta}^2$, where $\tilde{V}^*(x, s) = (v^*)^\top \phi(x, s)$ is the projected Bellman fixed point.

## 4.2 Gradient w.r.t. Lagrangian Parameter $\lambda$

**Lemma 11:** The gradient of $V^\theta(x_0, \nu)$ w.r.t. $\lambda$:

$$
\nabla_\lambda V^\theta(x_0, \nu) = \frac{1}{1-\gamma} \sum_{x, s, a} \pi_\gamma^\theta(x, s, a | x_0, \nu) \frac{1}{1-\alpha}\mathbf{1}\{x = x_{\text{Tar}}\}(-s)^+ \tag{28}
$$

An unbiased estimate is $\nu - \beta + \frac{1}{(1-\gamma)(1-\alpha)}\mathbf{1}\{x = x_{\text{Tar}}\}(-s)^+$.

- *Context*:
	- An issue with this estimator is that its value is fixed to $\nu_k - \beta$ throughout a trajectory and only changes at the target state. This affects the incremental nature of the algorithm, though the paper notes that a previously proposed approach using a second value function approximation for the constraint (Chow and Ghavamzadeh, 2014) increases approximation error and impedes convergence speed.

## 4.3 Sub-gradient w.r.t. VaR Parameter $\nu$

The sub-gradient can be written as:

$$
\partial_\nu L(\nu, \theta, \lambda) \ni \lambda\left(1 - \frac{1}{1 - \alpha}P(s_{\text{Tar}} \leq 0 | x_0, s_0 = \nu; \theta)\right) \tag{30}
$$

An unbiased estimate $\lambda - \lambda\mathbf{1}\{s_{\text{Tar}} \leq 0\}/(1-\alpha)$ can only be computed at the end of a trajectory.

For the **SPSA-based** incremental estimator, the sub-gradient is approximated via:

$$
g(\nu) \approx \lambda + \frac{v^\top[\phi(x_0, \nu + \Delta) - \phi(x_0, \nu - \Delta)]}{2\Delta}
$$

where $\Delta > 0$ is a positive perturbation that vanishes asymptotically.

- *Context*:
	- SPSA (Spall, 1992) is a general-purpose technique for estimating gradients using function evaluations at perturbed points, without requiring explicit derivative computation. Here it approximates the sub-gradient of $V^\theta(x_0, \nu)$ w.r.t. $\nu$ using the critic's value function approximation at $\nu + \Delta$ and $\nu - \Delta$. The perturbation $\Delta_k \to 0$ ensures asymptotic accuracy, while the condition $\sum_k(\zeta_2(k)/\Delta_k)^2 < \infty$ (Assumption 8) ensures the SPSA noise does not overwhelm the signal.

## 4.4 Convergence of Actor-Critic Methods

**Step-Size Conditions (Assumption 8):** Four timescales:

$$
\zeta_1(k) = o(\zeta_2(k)), \quad \zeta_2(k) = o(\zeta_3(k)), \quad \zeta_3(k) = o(\zeta_4(k))
$$

- **Fastest** ($\zeta_4$): Critic ($v$) update.
- **Fast** ($\zeta_3$): VaR ($\nu$) update.
- **Intermediate** ($\zeta_2$): Policy ($\theta$) update.
- **Slowest** ($\zeta_1$): Lagrange multiplier ($\lambda$) update.

**Theorem 12:** Suppose $\epsilon_{\theta_k}(v_k) \to 0$ (the Bellman residual vanishes), samples are from $\pi_\gamma^\theta$, and the SPSA technical conditions hold. Then under Assumptions 2–4 and 8–9, the actor-critic algorithm converges a.s. to a locally optimal policy for the CVaR-constrained problem.

- *Context*:
	- The four-timescale structure is necessitated by the interdependence of the parameters. The critic must converge first (so the value function estimate is accurate), then $\nu$ (which depends on the value function), then $\theta$ (which depends on both), and finally $\lambda$ (which depends on all three). Each faster variable sees the slower variables as quasi-static, enabling a sequential convergence analysis. The condition $\epsilon_{\theta_k}(v_k) \to 0$ is the standard requirement that function approximation error vanishes — in practice, this depends on the richness of the feature representation.

---

# 5. Extension to Chance-Constrained Optimization

The chance-constrained problem (4) is handled by Lagrangian relaxation:

$$
\max_\lambda \min_{\theta, \alpha} \left(L(\theta, \lambda) := G^\theta(x_0) + \lambda\left(P(J^\theta(x_0) \geq \alpha) - \beta\right)\right) \tag{32}
$$

## 5.1 Policy Gradient Method

Since no $\nu$ parameter is needed, the algorithm simplifies to a **two-timescale** stochastic approximation:

$$
\theta_{k+1} = \Gamma_\Theta\left[\theta_k - \frac{\zeta_2(k)}{N}\left(\sum_{j=1}^{N}\nabla_\theta \log P(\xi_{j,k})G(\xi_{j,k}) + \lambda_k \nabla_\theta \log P(\xi_{j,k})\mathbf{1}\{J(\xi_{j,k}) \geq \alpha\}\right)\right]
$$

$$
\lambda_{k+1} = \Gamma_\Lambda\left[\lambda_k + \zeta_1(k)\left(-\beta + \frac{1}{N}\sum_{j=1}^{N}\mathbf{1}\{J(\xi_{j,k}) \geq \alpha\}\right)\right]
$$

**Theorem 13:** Under Assumptions 2–6, convergence to a locally optimal policy for the chance-constrained problem holds a.s.

## 5.2 Actor-Critic Method

The augmented MDP for chance constraints is similar to the CVaR case, but with:
- $\bar{P}_0(x, s) = P_0(x)\mathbf{1}\{s = \alpha\}$
- $\bar{C}_\lambda(x, s, a) = \lambda\mathbf{1}\{s \leq 0\}$ if $x = x_{\text{Tar}}$, and $C(x, a)$ otherwise.

The total cost of a trajectory in the augmented MDP:

$$
\sum_{k=0}^{T} \bar{C}_\lambda(x_k, s_k, a_k) = G^\theta(x) + \lambda P(J^\theta(x) \geq \beta) \tag{33}
$$

Updates are episodic (after each trajectory ends). **Theorem 14** establishes critic convergence and **Theorem 15** establishes convergence to a locally optimal policy under Assumptions 2–9.

- *Context*:
	- The chance-constrained AC is episodic rather than fully incremental because the constraint indicator $\mathbf{1}\{s_{\text{Tar}} \leq 0\}$ can only be evaluated at the end of a trajectory. This contrasts with the CVaR AC, where the SPSA method provides an incremental $\nu$ estimate. The episodic nature increases variance but avoids the SPSA bias.

---

# 6. Examples

## 6.1 The Optimal Stopping Problem

### Setup

An optimal stopping problem for purchasing goods: at each time step $k \leq T$, the state is $(c_k, k)$ where $c_k$ is the purchase cost. The cost evolves as a **binomial Markov chain**:
- With probability $p$: $c_{k+1} = f_u \cdot c_k$ (appreciation by factor $f_u > 1$).
- With probability $1 - p$: $c_{k+1} = f_d \cdot c_k$ (depreciation by factor $f_d < 1$).

The agent decides to accept the cost ($u_k = 1$) or wait ($u_k = 0$). Accepting yields cost $\max(K, c_k)$; waiting incurs holding cost $p_h$ per step.

**Parameters:** $x_0 = [1; 0]$, $p_h = 0.1$, $T = 20$, $K = 5$, $\gamma = 0.95$, $f_u = 2$, $f_d = 0.5$, $p = 0.65$, $\alpha = 0.95$, $\beta = 3$, $N = 500{,}000$, $\kappa_1 = 1024$ (RBF features), Boltzmann policies.

- *Context*:
	- This is the purchasing-cost analogue of American option pricing — a standard testbed for risk-sensitive algorithms. The state space grows exponentially with $T$, making exact dynamic programming infeasible and requiring function approximation.

### Algorithms Compared

**Trajectory-based:** PG (risk-neutral), PG-CVaR, PG-CC (chance-constrained).
**Incremental:** AC (risk-neutral), AC-CVaR, AC-CVaR-SPSA, AC-VaR (chance-constrained).

### Results

| Algorithm | $\mathbb{E}[G^\theta]$ | $\sigma(G^\theta)$ | CVaR | VaR |
|---|---|---|---|---|
| PG | 1.177 | 1.065 | 4.464 | 4.005 |
| PG-CVaR | 1.997 | 0.060 | 2.000 | 2.000 |
| PG-CC | 1.994 | 0.121 | 2.058 | 2.000 |
| AC | 1.113 | 0.607 | 3.331 | 3.220 |
| AC-CVaR-SPSA | 1.326 | 0.322 | 2.145 | 1.283 |
| AC-CVaR | 1.343 | 0.346 | 2.208 | 1.290 |
| AC-VaR | 1.817 | 0.753 | 4.006 | 2.300 |

- Risk-constrained algorithms yield higher expected cost but substantially lower worst-case variability.
- The cost distributions of risk-constrained policies have lower right-tail (worst-case) mass compared to risk-neutral counterparts.
- CVaR constraints are satisfied but **not tight** (constraint not matched exactly) — a consequence of local optimality.

- *Context*:
	- The non-tightness of the constraint is a general feature of local gradient methods for constrained optimization, not specific to this paper. Prashanth and Ghavamzadeh (2013) and Bhatnagar and Lakshmanan (2012) report the same phenomenon. The authors note that since both the expectation and CVaR are sub-additive and convex, one can construct a tighter policy by taking a convex combination of the risk-neutral optimal policy and the risk-averse policy.

## 6.2 A Personalized Ad-Recommendation System

### Setup

An Adobe ad-recommendation simulator trained on real data from a Fortune 50 company. The agent observes a 31-dimensional feature vector for each user, selects among 4 ad classes, and receives reward $+1$ (click) or $0$ (no click). The formulation is **reward maximization** with a CVaR constraint on worst-case return:

$$
\max_\theta \mathbb{E}[R^\theta(x_0)] \quad \text{subject to} \quad \text{CVaR}_{1-\alpha}(-R^\theta(x_0)) \leq \beta \tag{38}
$$

**Parameters:** $T = 15$, $\gamma = 0.98$, $\alpha = 0.05$, $\beta = -0.12$, $N = 1{,}000{,}000$, $\kappa_1 = 4096$ (3rd order Fourier basis), Boltzmann policies.

### Results

| Algorithm | $\mathbb{E}[R^\theta]$ | $\sigma(R^\theta)$ | CVaR | VaR |
|---|---|---|---|---|
| PG | 0.396 | 1.898 | 0.037 | 1.000 |
| PG-CVaR | 0.287 | 0.914 | 0.126 | 1.795 |
| AC | 0.581 | 2.778 | 0 | 0 |
| AC-CVaR | 0.253 | 0.634 | 0.137 | 1.890 |

- Risk-constrained algorithms yield lower expected reward but **higher left-tail** (worst-case) reward distributions.
- The CVaR-constrained policies successfully lower-bound the worst-case revenue to meet company yearly targets.

- *Context*:
	- The ad-recommendation experiment is significant because it uses a realistic simulator trained on real data, not a synthetic benchmark. The risk-constrained agent avoids ads with high potential but high bounce risk, favoring "safer" revenue streams. The trade-off between expected LTV and worst-case LTV is precisely the kind of operational decision that motivates the constrained formulation — a company might accept lower average revenue in exchange for a guarantee that worst-case revenue meets a target.

---

# 7. Conclusions and Future Work

The paper's contributions:
1. Novel PG and AC algorithms for CVaR-constrained and chance-constrained MDPs with convergence proofs.
2. Empirical demonstration that risk-constrained policies effectively reduce right-tail (worst-case) cost mass at the expense of higher expected cost.

### Future Directions

1. **Convergence under sampling distribution:** Provide convergence proofs for AC algorithms when samples are generated by following the policy, not from its discounted occupation measure $\pi_\gamma^\theta$.
2. **Importance sampling:** Use importance sampling methods (Bardou et al., 2009; Tamar et al., 2015) to improve gradient estimates in the right tail of the cost distribution — the rare but catastrophic events that matter most for risk-constrained optimization.
3. **Applications:** Apply the algorithms to operations research, robotics, and finance.

- *Context*:
	- The sampling distribution issue is a known gap in the discounted AC literature: the gradient estimates are unbiased only when sampling from $\pi_\gamma^\theta$, but in practice the agent follows the policy $\mu_\theta$, generating samples from a different (on-policy) distribution. This discrepancy does not affect the PG algorithms (which use full trajectories) but introduces bias in the AC algorithms. To the authors' knowledge, no rigorous convergence analysis exists for likelihood-ratio-based discounted AC algorithms under the on-policy sampling distribution.

---

# Appendix A: Convergence of Policy Gradient Methods

## A.1 Computing the Gradients

The gradients are derived by expanding expectations and applying the likelihood-ratio trick.

**Gradient w.r.t. $\theta$:** Expanding $L(\nu, \theta, \lambda) = \sum_\xi P_\theta(\xi)G(\xi) + \lambda\nu + \frac{\lambda}{1-\alpha}\sum_\xi P_\theta(\xi)(J(\xi) - \nu)^+ - \lambda\beta$ and differentiating:

$$
\nabla_\theta L = \sum_{\xi: P_\theta(\xi) \neq 0} P_\theta(\xi) \nabla_\theta \log P_\theta(\xi)\left(G(\xi) + \frac{\lambda}{1-\alpha}(J(\xi) - \nu)\mathbf{1}\{J(\xi) \geq \nu\}\right) \tag{39}
$$

**Sub-gradient w.r.t. $\nu$:** The set of sub-derivatives of $(J(\xi) - \nu)^+$ is:

$$
\partial_\nu(J(\xi) - \nu)^+ = \begin{cases} -1 & \text{if } \nu < J(\xi) \\ -q, \; q \in [0, 1] & \text{if } \nu = J(\xi) \\ 0 & \text{otherwise} \end{cases}
$$

**Gradient w.r.t. $\lambda$:** Since $L$ is linear in $\lambda$:

$$
\nabla_\lambda L = \nu - \beta + \frac{1}{1-\alpha}\sum_\xi P_\theta(\xi)(J(\xi) - \nu)\mathbf{1}\{J(\xi) \geq \nu\} \tag{41}
$$

## A.2 Proof of Convergence (Theorem 7)

The proof proceeds in five steps:

**Step 1 ($\nu$-convergence):** The $\nu$ update is a stochastic approximation of the differential inclusion $\dot{\nu} \in \Upsilon_\nu[-g(\nu)]$ for $g(\nu) \in \partial_\nu L$. Using the Lyapunov function $L_{\theta,\lambda}(\nu) = L(\nu, \theta, \lambda) - L(\nu^*, \theta, \lambda)$ and Corollary 4 in Chapter 5 of Borkar (2008), convergence to $\nu^* \in \mathcal{N}_c$ (the set of stationary points) is established.

**Step 2 ($\theta$-convergence):** With $\nu = \nu^*(\theta)$ and $\lambda$ fixed, the $\theta$ update is a stochastic approximation of $\dot{\theta} = \Upsilon_\theta[-\nabla_\theta L|_{\nu=\nu^*(\theta)}]$. The Lyapunov function $L_\lambda(\theta) = L(\nu^*(\theta), \theta, \lambda) - L(\nu^*(\theta^*), \theta^*, \lambda)$ satisfies $dL_\lambda/dt \leq 0$, establishing local asymptotic stability at $\theta^*$.

**Step 3 (Local minimum):** By contradiction, showing that the convergent point $(\theta^*, \nu^*)$ must be a local minimum of $L(\nu, \theta, \lambda)$ for fixed $\lambda$.

**Step 4 ($\lambda$-convergence):** The $\lambda$ update is a stochastic approximation of $\dot{\lambda} = \Upsilon_\lambda[\nabla_\lambda L|_{\theta=\theta^*(\lambda), \nu=\nu^*(\lambda)}]$. The **envelope theorem** (Milgrom and Segal, 2002) ensures that $\nabla_\lambda L^*(\lambda) = \nabla_\lambda L|_{\theta=\theta^*(\lambda), \nu=\nu^*(\lambda)}$, so the ODE is a gradient ascent of $L^*$.

**Step 5 (Local saddle point):** The converged $(\theta^*, \nu^*, \lambda^*)$ is shown to be a local saddle point by establishing: (a) CVaR constraint satisfaction (Eq. 62), and (b) complementary slackness (Eq. 63). The case $\lambda^* = \lambda_{\max}$ (spurious fixed point) is handled by incrementally increasing $\lambda_{\max}$.

- *Context*:
	- The envelope theorem (Theorem 16 in the paper) is the mathematical bridge between the ODE analysis and the Lagrangian structure. It shows that the derivative of the optimal Lagrangian value $L^*(\lambda) = L(\nu^*(\lambda), \theta^*(\lambda), \lambda)$ equals the partial derivative of $L$ w.r.t. $\lambda$ evaluated at the optimizers — even though $\theta^*(\lambda)$ and $\nu^*(\lambda)$ themselves depend on $\lambda$. This is crucial because it means the $\lambda$ ODE is truly performing gradient ascent on $L^*$, not some distorted version of it.

---

# Appendix B: Convergence of Actor-Critic Algorithms

## B.1 Gradient w.r.t. $\lambda$ (Proof of Lemma 11)

The proof unrolls the Bellman recursion for $\nabla_\lambda V^\theta(x_0, \nu)$ on the augmented MDP $\bar{M}$. Using $h(x, s) = \sum_a \mu(a|x, s; \theta)\nabla_\lambda \bar{C}(x, s, a)$ and expanding:

$$
\nabla_\lambda V^\theta(x_0, \nu) = \sum_{k=0}^{\infty} \gamma^k \sum_{x, s} P(x_k = x, s_k = s | x_0, s_0 = \nu; \theta) h(x, s)
$$

$$
= \frac{1}{1-\gamma}\sum_{x, s, a} \pi_\gamma^\theta(x, s, a | x_0, \nu) \frac{1}{1-\alpha}\mathbf{1}\{x = x_{\text{Tar}}\}(-s)^+
$$

## B.2 Proof of Convergence (Theorem 12)

The proof has five steps:

**Step 1 (Critic convergence):** The critic update $v_{k+1} = v_k + \zeta_4(k)\phi(x_k, s_k)\delta_k(v_k)$ is a stochastic approximation of the ODE $\dot{v} = b - Av$, where $A$ is defined via the occupation measure (Eq. 66) and $b$ via the expected cost (Eq. 67). **Lemma 20** establishes that all eigenvalues of $A$ have positive real part, ensuring the ODE has a unique globally asymptotically stable equilibrium $v^* = A^{-1}b$. Boundedness of iterates follows from Theorem 3.1 of Borkar (2008), and convergence from Theorem 2 in Chapter 2 of Borkar (2008).

**Step 2 (SPSA-based $\nu$-convergence):** The SPSA estimator is shown to be asymptotically unbiased under **Assumption 21** (bounded Lipschitz feature functions). The bias decomposes as: $\Lambda_{1,k+1}$ (zero-mean noise), $\Lambda_{2,k}$ (SPSA bias $\to 0$ as $\Delta_k \to 0$), and $\Lambda_{3,k}$ (value function approximation error $\to 0$ as $\epsilon_\theta(v_k) \to 0$). The condition $\epsilon_{\theta_k}(v_k) = o(\Delta_k)$ ensures the approximation error vanishes faster than the SPSA perturbation.

**Steps 3–5:** Follow the same structure as the PG convergence proof (Steps 2–5 of Theorem 7), replacing trajectory-based estimates with incremental TD-based estimates and accounting for the function approximation error.

- *Context*:
	- The critic convergence proof (Step 1) is a standard result in the TD-learning literature, adapted here to the augmented state space. The key technical result is Lemma 20 (from Bertsekas and Tsitsiklis, 1996): the matrix $A$ in the projected TD fixed-point equation has eigenvalues with positive real part, ensuring that the ODE $\dot{v} = b - Av$ converges to a unique equilibrium. This positive-definiteness property relies on the discount factor $\gamma < 1$, which makes the Bellman operator a contraction.

---

# MATHEMATICS

The mathematical framework proceeds from the definition of the CVaR-constrained optimization problem, through Lagrangian relaxation and augmented MDP construction, to the derivation of gradient formulas and multi-timescale convergence analysis. The key technical tools are: the Rockafellar-Uryasev CVaR reformulation (converting the risk constraint into a smooth optimization), state-space augmentation (restoring the Markov property for history-dependent risk), the policy gradient theorem on augmented MDPs, and ODE-based Lyapunov analysis (establishing convergence of multi-timescale stochastic approximation).

### 1. CVaR as a Minimization Problem

The **Conditional Value-at-Risk** at confidence level $\alpha \in (0, 1)$ for a cost random variable $Z$ with $\mathbb{E}[|Z|] < \infty$ is defined via:

**CVaR Definition ($\text{CVaR}_\alpha$):**

$$
\text{CVaR}_\alpha(Z) := \min_{\nu \in \mathbb{R}} \left\{\nu + \frac{1}{1-\alpha}\mathbb{E}[(Z - \nu)^+]\right\} \tag{1}
$$

where $(x)^+ = \max(x, 0)$. The minimizer $\nu^*$ equals $\text{VaR}_\alpha(Z) = \min\{z \mid F_Z(z) \geq \alpha\}$. The function $H_\alpha(Z, \nu) := \nu + \frac{1}{1-\alpha}\mathbb{E}[(Z - \nu)^+]$ is convex in $\nu$, which makes the joint minimization over $(\theta, \nu)$ tractable.

### 2. The Constrained Optimization Problem

For a policy $\mu(\cdot|x; \theta)$ with cumulative cost $G^\theta(x_0)$, cumulative constraint cost $J^\theta(x_0)$, and discount factor $\gamma \in (0, 1)$:

**CVaR-Constrained MDP:**

$$
\min_{\theta, \nu} V^\theta(x_0) \quad \text{subject to} \quad H_\alpha(J^\theta(x_0), \nu) \leq \beta \tag{3}
$$

where $V^\theta(x_0) = \mathbb{E}[G^\theta(x_0)]$.

### 3. The Lagrangian Function

**Lagrangian ($L$):**

$$
L(\nu, \theta, \lambda) := V^\theta(x_0) + \lambda\left(H_\alpha(J^\theta(x_0), \nu) - \beta\right) = V^\theta(x_0) + \lambda\left(\nu + \frac{1}{1-\alpha}\mathbb{E}[(J^\theta(x_0) - \nu)^+] - \beta\right) \tag{5}
$$

The constrained problem is equivalent to the saddle-point problem $\max_{\lambda \geq 0} \min_{\theta, \nu} L(\nu, \theta, \lambda)$. A local saddle point $(\nu^*, \theta^*, \lambda^*)$ satisfies $L(\nu, \theta, \lambda^*) \geq L(\nu^*, \theta^*, \lambda^*) \geq L(\nu^*, \theta^*, \lambda)$ locally, and by the saddle point theorem, $\theta^*$ is a locally optimal policy for the original constrained problem.

### 4. Lagrangian Gradients

The three descent/ascent directions are:

**Policy Gradient ($\nabla_\theta L$):**

$$
\nabla_\theta L = \sum_\xi P_\theta(\xi) \nabla_\theta \log P_\theta(\xi)\left(G(\xi) + \frac{\lambda}{1-\alpha}(J(\xi) - \nu)\mathbf{1}\{J(\xi) \geq \nu\}\right) \tag{39}
$$

where $\nabla_\theta \log P_\theta(\xi) = \sum_{k=0}^{T-1} \nabla_\theta \log \mu(a_k|x_k; \theta)$ is the score function (likelihood ratio).

**VaR Sub-gradient ($\partial_\nu L$):**

$$
\partial_\nu L \ni \lambda - \frac{\lambda}{1-\alpha}P(J^\theta(x_0) \geq \nu) \tag{40}
$$

This is a sub-gradient (not a gradient) because $(Z - \nu)^+$ is non-differentiable at $Z = \nu$.

**Multiplier Gradient ($\nabla_\lambda L$):**

$$
\nabla_\lambda L = \nu + \frac{1}{1-\alpha}\mathbb{E}[(J^\theta(x_0) - \nu)^+] - \beta \tag{41}
$$

Since $L$ is linear in $\lambda$, this gradient is exact and equals $H_\alpha(J^\theta(x_0), \nu) - \beta$.

### 5. The Augmented MDP

To enable incremental (actor-critic) algorithms, the paper constructs the **augmented MDP** $\bar{M} = (\bar{\mathcal{X}}, \bar{\mathcal{A}}, \bar{C}_\lambda, \bar{P}, \bar{P}_0)$ with $\bar{\mathcal{X}} = \mathcal{X} \times \mathcal{S}$, where the auxiliary state $s$ tracks the "remaining constraint budget." The key identity linking the augmented MDP to the Lagrangian is:

**Augmented Cost Identity:**

$$
\sum_{k=0}^{T} \gamma^k \bar{C}_\lambda(x_k, s_k, a_k) = G^\theta(x) + \frac{\lambda}{1-\alpha}(J^\theta(x) - s)^+ \tag{25}
$$

At the initial state $(x_0, s_0 = \nu)$, the value function of $\bar{M}$ equals the Lagrangian (up to the $\lambda\nu - \lambda\beta$ terms that do not depend on $\theta$):

$$
V^\theta(x_0, \nu) = \mathbb{E}\left[G^\theta(x_0)\right] + \frac{\lambda}{1-\alpha}\mathbb{E}\left[(J^\theta(x_0) - \nu)^+\right]
$$

so that $\nabla_\theta L(\nu, \theta, \lambda) = \nabla_\theta V^\theta(x_0, \nu)$.

### 6. Policy Gradient Theorem on $\bar{M}$

By the standard policy gradient theorem (Sutton et al., 2000) applied to $\bar{M}$:

**Augmented Policy Gradient:**

$$
\nabla_\theta V^\theta(x_0, \nu) = \frac{1}{1-\gamma}\sum_{x, s, a} \pi_\gamma^\theta(x, s, a | x_0, \nu) \nabla_\theta \log \mu(a|x, s; \theta) \, Q^\theta(x, s, a) \tag{26}
$$

where $\pi_\gamma^\theta$ is the $\gamma$-discounted occupation measure on $\bar{M}$ and $Q^\theta$ is the action-value function of $\bar{M}$. With linear approximation $V^\theta(x, s) \approx v^\top \phi(x, s)$, the TD error $\delta_k = \bar{C}_\lambda(x_k, s_k, a_k) + \gamma v^\top \phi(x_{k+1}, s_{k+1}) - v^\top \phi(x_k, s_k)$ provides an unbiased gradient estimate.

### 7. The Bellman Operator on $\bar{M}$

**Bellman Operator ($B^\theta$):**

$$
B^\theta[V](x, s) = \sum_a \mu(a|x, s; \theta)\left[\bar{C}_\lambda(x, s, a) + \gamma\sum_{x', s'}\bar{P}(x', s'|x, s, a)V(x', s')\right] \tag{T10}
$$

The fixed point $V^* = B^\theta[V^*]$ satisfies $V^*(x_0, \nu) = L(\nu, \theta, \lambda) - \lambda\nu + \lambda\beta$. The projected fixed point $\tilde{V}^* = \Pi B^\theta[\tilde{V}^*]$ is the critic's target, where $\Pi$ projects onto the span of feature vectors $\Phi$.

### 8. Multi-Timescale Convergence via ODE Analysis

The discrete updates $(\nu_k, \theta_k, \lambda_k)$ are stochastic approximations of the continuous-time ODEs:

**$\nu$-ODE (fastest):**

$$
\dot{\nu} \in \Upsilon_\nu[-g(\nu)], \quad g(\nu) \in \partial_\nu L(\nu, \theta, \lambda) \tag{43}
$$

**$\theta$-ODE (intermediate):**

$$
\dot{\theta} = \Upsilon_\theta[-\nabla_\theta L(\nu, \theta, \lambda)|_{\nu=\nu^*(\theta)}] \tag{44}
$$

**$\lambda$-ODE (slowest):**

$$
\dot{\lambda} = \Upsilon_\lambda[\nabla_\lambda L(\nu, \theta, \lambda)|_{\theta=\theta^*(\lambda), \nu=\nu^*(\lambda)}] \tag{46}
$$

where $\Upsilon_x[K(x)] := \lim_{0 < \eta \to 0} (\Gamma(x + \eta K(x)) - \Gamma(x))/\eta$ is the left directional derivative of the projection operator, ensuring the gradient points along the boundary when the iterate hits the constraint set.

The **Lyapunov analysis** uses $L(\nu, \theta, \lambda)$ itself:
- For $\nu$: $L_{\theta,\lambda}(\nu) = L(\nu, \theta, \lambda) - L(\nu^*, \theta, \lambda) \geq 0$ with $\max_{g(\nu)} D_t L \leq 0$.
- For $\theta$: $L_\lambda(\theta) = L(\nu^*(\theta), \theta, \lambda) - L(\nu^*(\theta^*), \theta^*, \lambda) \geq 0$ with $dL_\lambda/dt \leq 0$.
- For $\lambda$: $\mathcal{L}(\lambda) = -L(\theta^*(\lambda), \nu^*(\lambda), \lambda) + L(\theta^*(\lambda^*), \nu^*(\lambda^*), \lambda^*) \geq 0$ with $d\mathcal{L}/dt \leq 0$.

The **envelope theorem** (Theorem 16) establishes that $\nabla_\lambda L^*(\lambda) = \nabla_\lambda L|_{\theta^*, \nu^*}$, ensuring the $\lambda$-ODE performs gradient ascent on $L^*$.

### 9. Chance-Constrained Extension

For the chance-constrained problem, the Lagrangian simplifies to:

**Chance-Constrained Lagrangian:**

$$
L(\theta, \lambda) = G^\theta(x_0) + \lambda(P(J^\theta(x_0) \geq \alpha) - \beta) \tag{32}
$$

No $\nu$ parameter is needed. The augmented cost becomes $\bar{C}_\lambda(x, s, a) = \lambda\mathbf{1}\{s \leq 0\}$ at $x_{\text{Tar}}$, giving total cost $G^\theta(x) + \lambda P(J^\theta(x) \geq \beta)$. The convergence analysis reduces to a **two-timescale** (or three-timescale for AC) scheme.

---

### **1. Problem:**

The paper addresses the gap in **risk-constrained** reinforcement learning, where the objective is to minimize expected cumulative cost subject to a bound on tail risk (CVaR or chance constraint). While risk-sensitive RL had received substantial attention, the constrained formulation — which decouples the performance objective from the risk measure and provides operationally interpretable risk budgets — was largely unaddressed. The paper provides the first policy gradient and actor-critic algorithms for percentile risk-constrained MDPs with formal convergence guarantees.

### **2. Setup:**

The method assumes a **finite MDP** $(\mathcal{X}, \mathcal{A}, C, D, P, P_0)$ with bounded costs, a recurrent absorbing target state $x_{\text{Tar}}$, and a uniformly bounded first-hitting time $T$. Policies are stationary, $\theta$-parameterized, and continuously differentiable with Lipschitz gradients. The CVaR-constrained formulation uses $\gamma \in (0, 1)$ (discounted); the chance-constrained formulation uses $\gamma = 1$ (undiscounted). Strict feasibility (Slater's condition) is required for the risk constraint. An **augmented MDP** $\bar{M} = \mathcal{X} \times \mathcal{S}$ tracks cumulative constraint cost via an auxiliary state $s$. The actor-critic algorithms use linear function approximation with full-rank features on $\bar{M}$.

### **3. Key Idea:**

The core methodological contribution is the combination of **Lagrangian relaxation** with **state-space augmentation** to transform a CVaR-constrained MDP into an unconstrained MDP on an augmented state space, where the auxiliary state $s$ tracks the remaining constraint "budget." This augmented MDP admits a standard Bellman operator whose fixed point equals the Lagrangian optimum, enabling the application of standard policy gradient and TD-learning machinery. The resulting algorithms operate on multiple timescales — critic, VaR parameter, policy, Lagrange multiplier — with convergence to local saddle points guaranteed via ODE-based Lyapunov analysis.

### **4. Assumptions:**

**Explicit:**
- Bounded first-hitting time $T$ for all policies and initial states (Assumption 2).
- Continuous differentiability of the policy $\mu(a|x; \theta)$ in $\theta$ with Lipschitz gradients (Assumption 3).
- Strict feasibility (Slater's condition) for the risk constraint (Assumption 4).
- Multi-timescale step-size conditions (Assumptions 6, 8).
- Linearly independent basis functions with full-rank feature matrix (Assumption 9).
- For SPSA: bounded Lipschitz feature functions and $\epsilon_\theta(v_k) = o(\Delta_k)$ (Assumption 21).

**Implicit:**
- **Simulator access:** The algorithms require extensive trajectory sampling or incremental state transitions from the MDP — purely offline application is not addressed.
- **Finite state/action spaces:** The augmented MDP construction and convergence proofs rely on finiteness, though the state space can be very large (function approximation handles this).
- **Occupation measure sampling:** The AC gradient estimates are unbiased only when sampling from the $\gamma$-discounted occupation measure $\pi_\gamma^\theta$, not when following the policy — a known gap acknowledged in the paper.
- **Adequate function approximation:** The convergence guarantees require $\epsilon_\theta(v_k) \to 0$, which depends on the feature representation being rich enough to approximate the augmented value function.

### **5. Limitation:**

- **Local optimality only:** Convergence is guaranteed to local saddle points; the resulting policies may be overly conservative, and constraints are satisfied but not necessarily tight.
- **Dimensionality overhead:** The augmented state space $\mathcal{X} \times \mathcal{S}$ increases dimensionality, potentially slowing convergence and requiring richer function approximation than the original MDP.
- **Multi-timescale sensitivity:** Four-timescale stochastic approximation requires careful step-size tuning; the convergence rate is not characterized, and practical performance depends heavily on step-size schedules.
- **Sampling bias in AC:** The actor-critic gradient estimates are biased when following the on-policy distribution rather than sampling from $\pi_\gamma^\theta$ — a theoretical gap that the paper acknowledges but does not resolve.
- **No importance sampling:** Gradient estimation for tail-sensitive constraints (CVaR at high $\alpha$) suffers from high variance because the relevant tail events are rare. The paper identifies importance sampling as future work but provides no solution.
- **Discrete cost distribution handling:** For discrete cost distributions (Eq. 8), the sub-gradient of $L$ w.r.t. $\nu$ is a set-valued mapping, requiring differential inclusion analysis and potentially slower convergence than the continuous case.

### **6. Relevance & Open Questions:**

This paper provides foundational algorithms for risk-constrained RL that are directly relevant to financial portfolio management, where CVaR constraints are standard regulatory requirements (e.g., Basel III/IV). The constrained formulation aligns with practical risk management: specify a risk budget $\beta$ for CVaR and optimize expected returns subject to it.

Key open questions and connections:
- **CVaR constraint tightness:** Can the non-tightness of the constraint at local optima be addressed by combining risk-neutral and risk-averse policies (as the authors suggest), or by global optimization techniques?
- **Off-policy extension:** The paper's AC algorithms require on-policy sampling. Can off-policy methods (importance sampling, retrace, V-trace) be integrated to enable batch/offline risk-constrained RL?
- **Continuous state/action spaces:** The augmented MDP construction assumes finite spaces. Extension to continuous spaces requires neural network function approximation and likely different convergence machinery.
- **Connection to coherent risk framework:** [[PNOTES - Policy Gradient for Coherent Risk Measures]] provides a unified gradient formula for *all* coherent risk measures. Can the constrained formulation here be generalized to constraints on arbitrary coherent risk measures, not just CVaR and chance constraints?
- **Distributional RL synergy:** [[PNOTES - A Distributional Perspective on Reinforcement Learning]] learns the full return distribution, which is exactly what CVaR and VaR require. Can distributional RL provide more efficient CVaR gradient estimation by directly accessing the tail distribution rather than relying on indicator-based sampling?

---

### Integration:

* **Problem:** This paper addresses a critical practical need in risk-sensitive RL: the ability to impose *constraints* on tail risk rather than simply incorporating risk into the objective. For portfolio optimization and financial risk management, the constrained formulation maps directly onto regulatory frameworks (CVaR limits, probability-of-loss bounds) and business requirements (minimum revenue guarantees). The Lagrangian relaxation + augmented MDP approach provides a principled algorithmic pipeline that connects the abstract CMDP theory of Altman (1999) to implementable RL algorithms. Combined with the coherent risk gradient framework of [[PNOTES - Policy Gradient for Coherent Risk Measures]], this paper suggests a complete toolkit: use the coherent framework to choose the right risk measure, then use the constrained formulation to impose it as a hard bound.

* **Limitation:** The most pressing limitations for practical deployment are the **local optimality** (constraints satisfied but not tight, potentially over-conservative policies), the **sampling inefficiency** for tail-sensitive constraints (CVaR gradient estimation requires rare-event samples), and the **multi-timescale step-size sensitivity** (four coupled timescales with no convergence rate guarantees). The augmented state space also introduces a **curse of dimensionality** for the critic that grows with the resolution of the constraint cost discretization. For financial applications where state and action spaces are continuous and the relevant tail events have probability $< 0.05$, these limitations represent significant barriers to direct application without further methodological development (importance sampling, neural function approximation, off-policy corrections).
