[[Policy Gradient for Coherent Risk Measures.pdf]]

# **Abstract:**

The paper extends policy gradient methods to the entire class of **coherent risk measures** — risk measures satisfying the four axioms of convexity, monotonicity, translation invariance, and positive homogeneity. Prior work on risk-sensitive policy gradients addressed specific risk measures (variance, CVaR) in isolation; this paper provides a *unified* framework that subsumes those results as special cases.

- Two problem settings are addressed:
	- **Static coherent risk** of the total discounted return from an MDP — optimized via a sampling-based gradient formula derived from the **Envelope Theorem**.
	- **Time-consistent dynamic Markov coherent risk** — optimized via a generalized **Policy Gradient Theorem** and an accompanying actor-critic algorithm with function approximation.
- The approach combines standard policy gradient sampling with **convex programming** at each update step, leveraging the **Representation Theorem** that characterizes any coherent risk as a worst-case expectation over a **risk envelope**.

- *Context*:
	- The key insight is that the Representation Theorem (Theorem 2.1) transforms the gradient of an abstract risk measure into the gradient of an expectation — but under an adversarially chosen density $\xi^*$ from the risk envelope $\mathcal{U}$. This converts the problem from "differentiate through a complex risk functional" to "differentiate through a re-weighted expectation," which is exactly what the standard likelihood-ratio trick handles. The convex programming step solves for the worst-case density; the sampling step estimates the gradient under that density.

---

# 1. Introduction

The paper motivates the shift from risk-neutral to risk-sensitive RL by identifying a fundamental limitation: standard RL minimizes *expected* cost, ignoring cost variability. In domains like finance, robotics, and operations research, variability management is as important as mean optimization.

- Existing risk-sensitive RL approaches are fragmented:
	- Exponential utility functions (Borkar, 2001)
	- Mean-variance models (Moody & Saffell, 2001; Tamar et al., 2012; Prashanth & Ghavamzadeh, 2013)
	- CVaR in the static setting (Chow & Ghavamzadeh, 2014; Tamar et al., 2015)
	- Dynamic coherent risk for linear dynamics only (Petrik & Subramanian, 2012; Chow & Pavone, 2014)
- The paper argues that the *choice* of risk measure is problem-dependent and should not be hardcoded. The coherent risk axioms (Artzner et al., 1999) identify a principled class of measures that satisfy basic rationality requirements.

- *Context*:
	- The fragmentation problem is real and costly: each prior paper derived its own gradient formula from scratch for its chosen risk measure, with no reusable structure. By targeting the entire coherent class, this paper provides a *template* — derive the risk envelope for your chosen measure, plug it into the general formula, and obtain a gradient estimator. This is the risk-sensitive analogue of how the standard Policy Gradient Theorem provides a template for all expected-return objectives.

For sequential decision problems, the paper also requires **time consistency**: if a policy is risk-optimal for an $n$-stage problem, the sub-policy from stage $t < n$ onward must also be risk-optimal. The class of **Markov coherent risk measures** (Ruszczyński, 2010) satisfies both coherence and time consistency.

- *Context*:
	- Time consistency is not a technicality — it is essential for dynamic programming. Without it, the Bellman equation does not hold, backward induction fails, and any policy optimized over the full horizon may be suboptimal when restricted to a suffix. Example 2.1 in Iancu et al. (2011), referenced in the paper, shows that optimizing a static CVaR measure over a multi-period problem can produce policies that no rational agent would follow from any intermediate state. Markov coherent risk measures eliminate this paradox by construction.

The paper's three contributions:
1. A new **gradient formula for static coherent risk** that is convenient for sampling-based estimation.
2. An **algorithm** for general static coherent risk that combines sampling with convex programming, with a consistency result.
3. A new **Policy Gradient Theorem for Markov coherent risk** with an actor-critic algorithm.

### Related Work

- Ruszczyński & Shapiro (2006) studied optimization of coherent risk in stochastic programming, but their formulation assumes policy parameters do not affect the distribution of the stochastic system — only the reward function — making it unsuitable for most RL problems.
- Osogami (2012) showed that an MDP with a dynamic coherent risk objective is essentially a **robust MDP**. This connection is exploited in Section 5 for value function approximation.
- Tamar et al. (2014) developed function approximation for robust MDPs; this paper uses their TD algorithm as a subroutine in the critic.

- *Context*:
	- The robust MDP connection is important: it means the value function $V_\theta(x)$ from the risk-sensitive Bellman equation is *identical* to the value function of a robust MDP with uncertainty set $\mathcal{U}(x, P_\theta(\cdot|x))$. This lets the paper borrow existing robust TD algorithms for the critic rather than inventing new ones. The novelty is in the *actor* — deriving the policy gradient under this robust/risk-sensitive value function.

---

# 2. Preliminaries

The paper works on a probability space $(\Omega, \mathcal{F}, P_\theta)$ where $\Omega$ is a finite sample space, $\mathcal{F}$ is a $\sigma$-algebra over $\Omega$, and $P_\theta \in \mathcal{B}$ is a probability measure parameterized by $\theta \in \mathbb{R}^K$. The set $\mathcal{B} := \{\xi : \sum_{\omega \in \Omega} \xi(\omega) = 1, \; \xi \geq 0\}$ is the simplex of probability distributions.

- The random variable $Z : \Omega \to (-\infty, \infty)$ is interpreted as a **cost** (lower is better).
- A $\xi$-weighted expectation is defined as $\mathbb{E}_\xi[Z] := \sum_{\omega \in \Omega} P_\theta(\omega)\xi(\omega)Z(\omega)$.

- *Context*:
	- The density function $\xi$ re-weights the base probability $P_\theta$. When $\xi(\omega) = 1$ for all $\omega$, the $\xi$-weighted expectation reduces to the standard expectation under $P_\theta$. Non-uniform $\xi$ tilts the distribution toward or away from certain outcomes — this is the mechanism through which coherent risk measures emphasize worst-case scenarios.

An **MDP** is defined as $M = (\mathcal{X}, \mathcal{A}, C, P, \gamma, x_0)$:
- $\mathcal{X}$, $\mathcal{A}$: state and action spaces
- $C(x) \in [-C_{\max}, C_{\max}]$: bounded, deterministic, state-dependent cost
- $P(\cdot|x, a)$: transition probability
- $\gamma$: discount factor
- $x_0$: initial state
- Actions follow a $\theta$-parameterized stationary Markov policy $\mu_\theta(\cdot|x)$.

## 2.1 Coherent Risk Measures

A **coherent risk measure** $\rho : \mathcal{Z} \to \mathbb{R}$ satisfies four axioms for all $Z, W \in \mathcal{Z}$:

- **A1 — Convexity:** $\forall \lambda \in [0,1]$, $\rho(\lambda Z + (1-\lambda)W) \leq \lambda\rho(Z) + (1-\lambda)\rho(W)$
	- Diversification reduces risk.
- **A2 — Monotonicity:** If $Z \leq W$ pointwise, then $\rho(Z) \leq \rho(W)$
	- Higher costs in every scenario imply higher risk.
- **A3 — Translation Invariance:** $\forall a \in \mathbb{R}$, $\rho(Z + a) = \rho(Z) + a$
	- Adding a deterministic cost shifts risk by that amount (the deterministic part does not contribute to *variability*).
- **A4 — Positive Homogeneity:** If $\lambda \geq 0$, then $\rho(\lambda Z) = \lambda\rho(Z)$
	- Scaling a position scales its risk linearly.

- *Context*:
	- These axioms originate from Artzner et al. (1999), one of the most influential papers in mathematical finance. They formalize what it means for a risk measure to be "rational" for single-period assessments. Not all common risk measures are coherent: variance violates A3 and A4; Value-at-Risk (VaR) violates A1 (it is not convex — two individually acceptable portfolios can combine into an unacceptable one). CVaR *is* coherent. Mean-semideviation is coherent for $\alpha \in [0,1]$.

### Representation Theorem (Theorem 2.1)

A risk measure $\rho$ is coherent if and only if there exists a convex, bounded, and closed set $\mathcal{U} \subset \mathcal{B}$ such that:

$$
\rho(Z) = \max_{\xi : \xi P_\theta \in \mathcal{U}(P_\theta)} \mathbb{E}_\xi[Z]
$$

- The set $\mathcal{U}(P_\theta)$ is called the **risk envelope**. It is a set of test density functions.
- Any coherent risk measure is uniquely represented by its risk envelope, and vice versa.
- The risk is the expectation of $Z$ under the *worst-case* density $\xi$ chosen adversarially from $\mathcal{U}$.

- *Context*:
	- This theorem is the linchpin of the entire paper. It converts a potentially opaque risk functional into a tractable optimization problem: maximize a *linear* objective ($\mathbb{E}_\xi[Z]$ is linear in $\xi$) over a *convex* constraint set ($\mathcal{U}$). This max-of-expectations structure is what makes the gradient derivation possible — the Envelope Theorem applies directly to such problems.
	- Different coherent risk measures correspond to different risk envelopes:
		- **Expectation:** $\mathcal{U} = \{\xi : \xi(\omega) = 1 \;\forall \omega\}$ (a singleton — no adversarial choice).
		- **CVaR at level $\alpha$:** $\mathcal{U} = \{\xi P_\theta : \xi(\omega) \in [0, \alpha^{-1}], \sum_\omega \xi(\omega) P_\theta(\omega) = 1\}$ — the adversary can inflate probabilities of bad outcomes by up to a factor of $1/\alpha$.
		- **Worst-case:** $\mathcal{U} = \mathcal{B}$ (the entire simplex — the adversary can concentrate all mass on the worst outcome).

### Assumption 2.2 (General Form of Risk Envelope)

The risk envelope $\mathcal{U}(P_\theta)$ is given in a canonical convex programming form:

$$
\mathcal{U}(P_\theta) = \left\{\xi P_\theta : g_e(\xi, P_\theta) = 0 \;\forall e \in \mathcal{E}, \; f_i(\xi, P_\theta) \leq 0 \;\forall i \in \mathcal{I}, \; \sum_{\omega} \xi(\omega)P_\theta(\omega) = 1, \; \xi(\omega) \geq 0 \right\}
$$

where:
- Each $g_e$ is **affine** in $\xi$ (equality constraints)
- Each $f_i$ is **convex** in $\xi$ (inequality constraints)
- A **strictly feasible point** $\xi$ exists (Slater's condition)
- For fixed $\xi$, $f_i(\xi, p)$ and $g_e(\xi, p)$ are twice differentiable in $p$
- The derivatives $df_i/dp(\omega)$ and $dg_e/dp(\omega)$ are uniformly bounded by some $M > 0$

- *Context*:
	- Slater's condition guarantees strong duality for the convex program, which is essential for the Envelope Theorem application. The smoothness assumptions on the constraints ensure that the gradient formula involves well-defined derivatives — without smoothness, the saddle point could behave pathologically as $\theta$ changes. These assumptions hold for CVaR, mean-semideviation, and spectral risk measures, covering essentially all coherent measures encountered in practice.

## 2.2 Dynamic Risk Measures

Static risk measures do not account for the *temporal structure* of costs in sequential problems. **Dynamic risk measures** explicitly incorporate the multi-period nature of MDP trajectories.

The motivation is **time consistency**: if a certain outcome is considered less risky in all states at stage $t+1$, it should also be considered less risky at stage $t$.

- *Context*:
	- Optimizing a static risk measure (e.g., static CVaR of total discounted return) over a multi-period MDP can produce time-inconsistent behaviour: a policy that is optimal when planned at $t=0$ may no longer appear optimal when re-evaluated at $t=5$. This makes the policy unimplementable in practice — a rational agent at $t=5$ would deviate from the plan. Dynamic risk measures resolve this by ensuring the Bellman principle of optimality holds.

### Markov Coherent Risk Measures

For a $T$-length horizon and MDP $M$, the **Markov coherent risk measure** $\rho_T(M)$ is defined recursively:

$$
\rho_T(M) = C(x_0) + \gamma\rho\left(C(x_1) + \cdots + \gamma\rho\left(C(x_{T-1}) + \gamma\rho(C(x_T))\right)\right)
$$

- Each $\rho$ is a static coherent risk measure satisfying Assumption 2.2.
- At each state $x$, the static risk $\rho$ is **induced by the transition probability** $P_\theta(\cdot|x) = \sum_{a \in \mathcal{A}} P(x'|x, a)\mu_\theta(a|x)$.
- The infinite-horizon version is $\rho_\infty(M) := \lim_{T \to \infty} \rho_T(M)$, well-defined since $\gamma < 1$ and cost is bounded.
- The risk measure is *Markov*: evaluation at each stage depends only on the current state, not the full history.

- *Context*:
	- The recursive nesting is the defining structure: risk is not applied to the total return as a lump sum (that would be static), but is applied *stage by stage*, each time wrapping the future risk inside the current-stage risk. This nesting is what enables the Bellman equation — the value at state $x$ depends on the risk-weighted value at next states, not on all future states simultaneously.
	- The notation is dense but the intuition is clean: the agent evaluates "how risky is the cost I pay now, plus a discounted version of how risky the future looks from the next state, where 'how risky' is defined by the worst-case density over next states."

---

# 3. Problem Formulation

The paper defines two optimization problems:

### Static Risk Problem (SRP)

$$
\min_\theta \rho(Z)
$$

where $Z$ is a random variable (e.g., the cumulative discounted cost $Z = C(x_0) + \gamma C(x_1) + \cdots + \gamma^T C(x_T)$ of a trajectory induced by the MDP under policy $\theta$).

### Dynamic Risk Problem (DRP)

$$
\min_\theta \rho_\infty(M)
$$

where $\rho_\infty$ is the Markov coherent dynamic risk measure.

- Neither problem is expected to be tractable in general — the dependence of $\rho$ on $\theta$ may be complex and non-convex.
- The paper pursues **locally optimal** $\theta$ via gradient descent.
- The central technical problem: compute the gradients $\nabla_\theta \rho(Z)$ and $\nabla_\theta \rho_\infty(M)$ using sampling.
- For the static case, i.i.d. samples of $Z$ are assumed available.
- For the dynamic case, i.i.d. samples of next states $x' \sim P(\cdot|x, a)$ are assumed available (simulator access).

- *Context*:
	- The SRP is simpler and serves as a building block for the DRP. The gradient formula for static risk (Section 4) becomes a subroutine inside the dynamic gradient formula (Section 5). This modular structure is one of the paper's elegant design choices — rather than deriving the dynamic case from scratch, they compose the static result with a Bellman decomposition.

---

# 4. Gradient Formula for Static Risk

This section derives the paper's first central result: a sampling-based formula for $\nabla_\theta \rho(Z)$.

### Assumption 4.1

The **likelihood ratio** $\nabla_\theta \log P(\omega)$ is well-defined and bounded for all $\omega \in \Omega$. Given $\omega$, the ratio can be easily calculated.

- *Context*:
	- This is the standard REINFORCE assumption. It requires that $P_\theta(\omega) > 0$ wherever the gradient is needed and that the parameterization is smooth enough for the log-derivative to exist. It holds for softmax policies, Gaussian policies, and most standard policy parameterizations.

### The Lagrangian

Using the Representation Theorem and Assumption 2.2, $\rho(Z)$ is the solution to the convex optimization problem in Eq. (1) for each $\theta$. The **Lagrangian** of this problem is:

$$
L_\theta(\xi, \lambda_P, \lambda_E, \lambda_I) = \sum_{\omega \in \Omega} \xi(\omega)P_\theta(\omega)Z(\omega) - \lambda_P\left(\sum_{\omega} \xi(\omega)P_\theta(\omega) - 1\right) - \sum_{e \in \mathcal{E}} \lambda_E(e)g_e(\xi, P_\theta) - \sum_{i \in \mathcal{I}} \lambda_I(i)f_i(\xi, P_\theta)
$$

where $\lambda_P$ enforces the probability normalization, $\lambda_E$ enforces equality constraints, and $\lambda_I$ enforces inequality constraints.

- Slater's condition (from Assumption 2.2) implies **strong duality**: $\rho(Z) = \max_{\xi \geq 0} \min_{\lambda_P, \lambda_I \geq 0, \lambda_E} L_\theta$.
- The Lagrangian has a non-empty set of saddle points $\mathcal{S}$.

### Theorem 4.2 (Static Gradient Formula)

For any saddle point $(\xi^*_\theta, \lambda^{*,P}_\theta, \lambda^{*,E}_\theta, \lambda^{*,I}_\theta) \in \mathcal{S}$:

$$
\nabla_\theta \rho(Z) = \mathbb{E}_{\xi^*_\theta}\left[\nabla_\theta \log P(\omega)(Z - \lambda^{*,P}_\theta)\right] - \sum_{e \in \mathcal{E}} \lambda^{*,E}_\theta(e) \nabla_\theta g_e(\xi^*_\theta; P_\theta) - \sum_{i \in \mathcal{I}} \lambda^{*,I}_\theta(i) \nabla_\theta f_i(\xi^*_\theta; P_\theta)
$$

- *Context*:
	- This formula is the paper's first major result. It says: the gradient of the risk equals the *risk-weighted* policy gradient (first term) minus correction terms from the constraint derivatives (second and third terms).
	- The first term has the classic REINFORCE structure — $\nabla_\theta \log P(\omega) \cdot (\text{signal})$ — but under the adversarial density $\xi^*_\theta$ rather than the base distribution, and with the baseline $\lambda^{*,P}_\theta$ (the Lagrange multiplier for probability normalization) automatically arising from the KKT conditions.
	- The constraint-dependent terms vanish for risk measures whose envelope constraints do not depend on $\theta$. For CVaR, for instance, the envelope is $\xi(\omega) \in [0, \alpha^{-1}]$ with no $\theta$-dependence, so only the first term survives.
	- The proof (Appendix A) applies the **Envelope Theorem** for saddle-point problems (Milgrom & Segal, 2002): the gradient of the optimal value equals the gradient of the Lagrangian evaluated at the saddle point. The likelihood-ratio trick then converts the resulting sum into an expectation.

## 4.1 Example 1: CVaR

The **Conditional Value at Risk** at level $\alpha \in [0,1]$ is:

$$
\rho_{\text{CVaR}}(Z; \alpha) = \inf_{t \in \mathbb{R}} \left\{t + \alpha^{-1}\mathbb{E}[(Z - t)_+]\right\}
$$

For continuous $Z$, this equals $\mathbb{E}[Z \mid Z > q_\alpha]$ where $q_\alpha$ is the $(1-\alpha)$-quantile.

- The risk envelope is $\mathcal{U} = \{\xi P_\theta : \xi(\omega) \in [0, \alpha^{-1}], \; \sum_\omega \xi(\omega) P_\theta(\omega) = 1\}$.
- The saddle point satisfies $\xi^*_\theta(\omega) = \alpha^{-1}$ when $Z(\omega) > \lambda^{*,P}_\theta$ and $\xi^*_\theta(\omega) = 0$ when $Z(\omega) < \lambda^{*,P}_\theta$, where $\lambda^{*,P}_\theta$ is any $(1-\alpha)$-quantile of $Z$.

Plugging into Theorem 4.2:

$$
\nabla_\theta \rho_{\text{CVaR}}(Z; \alpha) = \mathbb{E}\left[\nabla_\theta \log P(\omega)(Z - q_\alpha) \mid Z(\omega) > q_\alpha\right]
$$

- *Context*:
	- This re-derives the CVaR gradient formula of Tamar et al. (2015) as a special case, but under weaker assumptions and with a simpler proof. The formula says: the gradient of CVaR is a standard policy gradient, but computed only over the *tail* of the cost distribution (outcomes exceeding the quantile $q_\alpha$), with the quantile serving as a natural baseline.
	- The practical implication is that CVaR gradient estimation requires sampling from the tail — inherently inefficient because tail events are rare. This is the sample efficiency challenge flagged in the paper's conclusion.

## 4.2 Example 2: Mean-Semideviation

The **semi-deviation** is $\text{SD}[Z] := (\mathbb{E}[(Z - \mathbb{E}[Z])^2_+])^{1/2}$, measuring variability *only above the mean*.

For $\alpha \in [0,1]$, the **mean-semideviation** risk measure is:

$$
\rho_{\text{MSD}}(Z; \alpha) = \mathbb{E}[Z] + \alpha \cdot \text{SD}[Z]
$$

**Proposition 4.3** gives the gradient:

$$
\nabla_\theta \rho_{\text{MSD}}(Z; \alpha) = \nabla_\theta \mathbb{E}[Z] + \alpha \cdot \frac{\mathbb{E}\left[(Z - \mathbb{E}[Z])_+ \left(\nabla_\theta \log P(\omega)(Z - \mathbb{E}[Z]) - \nabla_\theta \mathbb{E}[Z]\right)\right]}{\text{SD}(Z)}
$$

where $\nabla_\theta \mathbb{E}[Z] = \mathbb{E}[\nabla_\theta \log P(\omega) Z]$.

- *Context*:
	- The mean-semideviation is particularly interesting because it penalizes only *downside* variability — costs above the mean — while ignoring upside variability. Standard deviation treats upside and downside symmetrically, which can lead to irrational risk aversion (penalizing high returns). This distinction is central to the numerical experiment in Section 6.
	- The gradient formula decomposes into: (1) the gradient of the mean (standard policy gradient), plus (2) a correction term that weights the policy gradient signal by how far above the mean each outcome falls, normalized by the semi-deviation. The proof (Appendix B) uses the Envelope Theorem on the dual representation of the semi-deviation as a supremum over an $L_2$-ball.

## 4.3 General Gradient Estimation Algorithm

For a general coherent risk where the Lagrangian saddle point is not known analytically, the paper proposes a **Sample Average Approximation (SAA)** of Theorem 4.2.

Given $N$ i.i.d. samples $\omega_i \sim P_\theta$:
1. Form the empirical distribution $P_{\theta;N}(\omega) = \frac{1}{N}\sum_{i=1}^N \mathbb{1}\{\omega_i = \omega\}$.
2. Solve the **empirical risk envelope optimization** (a convex program with $O(N)$ variables):

$$
\rho_N(Z) = \max_{\xi : \xi P_{\theta;N} \in \mathcal{U}(P_{\theta;N})} \sum_{i=1}^N P_{\theta;N}(\omega_i)\xi(\omega_i)Z(\omega_i)
$$

3. Obtain the saddle point $\xi^*_{\theta;N}$ and KKT multipliers $\lambda^{*,P}_{\theta;N}, \lambda^{*,E}_{\theta;N}, \lambda^{*,I}_{\theta;N}$ from the convex solver.
4. Compute the **gradient estimator**:

$$
\nabla_{\theta;N} \rho(Z) = \sum_{i=1}^N P_{\theta;N}(\omega_i)\xi^*_{\theta;N}(\omega_i)\nabla_\theta \log P(\omega_i)(Z(\omega_i) - \lambda^{*,P}_{\theta;N}) - \sum_{e \in \mathcal{E}} \lambda^{*,E}_{\theta;N}(e)\nabla_\theta g_e(\xi^*_{\theta;N}; P_{\theta;N}) - \sum_{i \in \mathcal{I}} \lambda^{*,I}_{\theta;N}(i)\nabla_\theta f_i(\xi^*_{\theta;N}; P_{\theta;N})
$$

- *Context*:
	- This is a **two-step procedure**: sample, then solve a convex program, then compute the gradient from the solution. The convex program must be solved at every policy update step — this is the computational overhead that distinguishes risk-sensitive policy gradient from its risk-neutral counterpart. For CVaR and mean-semideviation, the saddle point has a closed form and this step is cheap. For exotic coherent risk measures, an interior point solver is needed.

### Proposition 4.4 (Consistency)

Under mild regularity conditions (non-empty bounded saddle point set, continuous constraints, well-defined empirical risk), the SAA estimator is consistent:

$$
\lim_{N \to \infty} \rho_N(Z) = \rho(Z) \quad \text{and} \quad \lim_{N \to \infty} \nabla_{\theta;N} \rho(Z) = \nabla_\theta \rho(Z) \quad \text{w.p. 1}
$$

- The proof (Appendix C) proceeds in three parts: (1) the SAA Lagrangian converges uniformly to the true Lagrangian, (2) the SAA saddle points converge to the true saddle points, and (3) boundedness of the likelihood ratio and constraint derivatives ensures the gradient estimator inherits this convergence.

---

# 5. Gradient Formula for Dynamic Risk

This section derives the paper's second central result: a generalized Policy Gradient Theorem for Markov coherent dynamic risk.

### Risk-Sensitive Value Function

The risk-sensitive value function is $V_\theta(x) := \rho_\infty(M \mid x_0 = x)$. Due to the Markov structure of $\rho_\infty$, the value function satisfies the **risk-sensitive Bellman equation**:

$$
V_\theta(x) = C(x) + \gamma \max_{\xi P_\theta(\cdot|x) \in \mathcal{U}(x, P_\theta(\cdot|x))} \mathbb{E}_\xi[V_\theta(x')]
$$

where the expectation is over the next state transition. The optimization target satisfies $\nabla_\theta \rho_\infty(M) = \nabla_\theta V_\theta(x_0)$.

- *Context*:
	- This Bellman equation differs from the risk-neutral version in a critical way: the transition to the next state is *re-weighted* by the worst-case density $\xi^*$ from the risk envelope. Effectively, the agent plans as if the environment adversarially chooses transition probabilities within the envelope $\mathcal{U}$. This is the formal connection to **robust MDPs** (Osogami, 2012): the risk-sensitive value function equals the value function of a robust MDP with uncertainty set $\mathcal{U}$.
	- The value function $V_\theta$ is the unique fixed point of this equation, established by Ruszczyński (2010, Theorem 1).

### Assumption 5.1

The likelihood ratio $\nabla_\theta \log \mu_\theta(a|x)$ is well-defined and bounded for all $x \in \mathcal{X}$ and $a \in \mathcal{A}$.

### Theorem 5.2 (Dynamic Policy Gradient Theorem)

For each state $x \in \mathcal{X}$, let $(\xi^*_{\theta,x}, \lambda^{*,P}_{\theta,x}, \lambda^{*,E}_{\theta,x}, \lambda^{*,I}_{\theta,x})$ be a saddle point of the Lagrangian corresponding to state $x$ (with $P_\theta(\cdot|x)$ replacing $P_\theta$ and $V_\theta$ replacing $Z$). Then:

$$
\nabla V_\theta(x) = \mathbb{E}_{\xi^*_\theta}\left[\sum_{t=0}^{\infty} \gamma^t \nabla_\theta \log \mu_\theta(a_t | x_t) \, h_\theta(x_t, a_t) \;\middle|\; x_0 = x\right]
$$

where:
- $\mathbb{E}_{\xi^*_\theta}[\cdot]$ is the expectation over trajectories with **$\xi^*$-weighted transitions**: $P_\theta(\cdot|x)\xi^*_{\theta,x}(\cdot)$.
- The **stage-wise cost function** $h_\theta(x, a)$ is:

$$
h_\theta(x, a) = C(x, a) + \sum_{x' \in \mathcal{X}} P(x'|x, a)\xi^*_{\theta,x}(x')\left[\gamma V_\theta(x') - \lambda^{*,P}_{\theta,x} - \sum_{i \in \mathcal{I}} \lambda^{*,I}_{\theta,x}(i)\frac{df_i(\xi^*_{\theta,x}, p)}{dp(x')} - \sum_{e \in \mathcal{E}} \lambda^{*,E}_{\theta,x}(e)\frac{dg_e(\xi^*_{\theta,x}, p)}{dp(x')}\right]
$$

- *Context*:
	- This theorem is a direct generalization of the standard Policy Gradient Theorem (Sutton et al., 2000). In the risk-neutral case ($\xi^* = 1$, no constraint terms), it reduces to the familiar formula $\nabla V(x) = \mathbb{E}[\sum_t \gamma^t \nabla \log \mu(a_t|x_t) Q(x_t, a_t)]$, where the Q-function replaces $h_\theta$.
	- The two key differences from the risk-neutral case:
		1. **Trajectories are sampled under $\xi^*$-weighted transitions**, not the original transitions. This means the agent simulates the "worst-case" MDP corresponding to the risk envelope.
		2. **The cost signal $h_\theta$ incorporates KKT multiplier corrections** from the risk envelope constraints, in addition to the immediate cost and discounted future value. These corrections account for how the policy gradient affects the risk envelope constraints at each state.
	- The proof (Appendix D) applies the Envelope Theorem at each state to the risk-sensitive Bellman equation and unfolds the resulting recursion, using the bounded convergence theorem to handle the infinite horizon.

### Actor-Critic Algorithm

Theorem 5.2 enables an actor-critic framework:

**Critic:** Approximate $V_\theta$ using function approximation $V_\theta(x) \approx v^\top \phi(x)$, where $\phi(x) \in \mathbb{R}^{\kappa_2}$ is a feature vector. The critic uses a **Projected Risk-Sensitive Value Iteration (PRSVI)** algorithm (adapted from Tamar et al., 2014 for robust MDPs) with the update rule:

$$
v_{k+1} = \left(\frac{1}{N}\sum_{t=0}^{N-1} \phi(x_t)\phi(x_t)^\top\right)^{-1}\left[\frac{1}{N}\sum_{t=0}^{N-1} \phi(x_t)C_\theta(x_t) + \gamma \frac{1}{N}\sum_{t=0}^{N-1} \phi(x_t) \max_{\xi P_\theta(\cdot|x_t) \in \mathcal{U}} \mathbb{E}_\xi[\Phi v_k]\right]
$$

- *Context*:
	- The PRSVI update resembles the standard projected TD fixed-point iteration, but with the inner maximization over the risk envelope replacing the simple expectation. The inner optimization is itself a convex program that must be solved at each state visited in the trajectory. In practice, this is handled via the SAA approach from Section 4.3, with an $\ell_2$-regularization term to ensure convergence of the optimizers and KKT multipliers.

**Actor:** Given the critic's $V_\theta$ estimate, estimate $\nabla V_\theta$ via a **two-phase sampling procedure**:
1. Generate $N$ trajectories from the $\xi^*$-weighted MDP (using the critic's $V_\theta$ to derive $\xi^*$).
2. For each state-action pair in each trajectory, sample next states to estimate $h_\theta(x, a)$.
3. Compute the gradient estimate by averaging $\gamma^t \nabla_\theta \log \mu_\theta(a_t|x_t) h_{\theta,N}(x_t, a_t)$ over all samples.

**Assumption E.2 (Contraction Condition):** There exists $\kappa \in (0,1)$ such that $\xi(x') \leq \kappa/\gamma$ for all $\xi(\cdot)P_\theta(\cdot|x) \in \mathcal{U}(x, P_\theta(\cdot|x))$ and all $x, x'$. This ensures the projected risk-sensitive Bellman operator $\Pi T_\theta$ is a contraction, guaranteeing a unique fixed point for the critic.

### Theorem E.5 (Gradient Estimation Consistency)

The two-phase sampling procedure converges:

$$
\left|\lim_{N \to \infty} \frac{1}{N}\sum_{j=1}^N \sum_{t=0}^{\infty} \gamma^t \nabla \log \mu_\theta(a_t^{(j)}|x_t^{(j)}) h_{\theta,N}(x_t^{(j)}, a_t^{(j)}) - \nabla V_\theta(x_0)\right| = O(\Delta)
$$

where $\Delta = \|\Phi v^*_\theta - V_\theta\|_\infty$ is the value function approximation error.

- *Context*:
	- The $O(\Delta)$ bound means the gradient estimate converges to the true gradient *up to the critic's approximation error*. As the function approximation improves (more features, better representation), the gradient becomes more accurate. This is the standard price of function approximation in actor-critic methods — the novelty is extending it to the risk-sensitive setting.

---

# 6. Numerical Illustration

The experiment demonstrates that **flexibility in risk-measure design** is essential for rational risk-sensitive behaviour.

### Setup

A trading agent selects one of three assets:
- **Asset 1** ($A_1$): Normal, $\mu = 1$, $\sigma^2 = 1$
- **Asset 2** ($A_2$): Normal, $\mu = 4$, $\sigma^2 = 6$
- **Asset 3** ($A_3$): Pareto, $\alpha = 1.5$, mean $= 3$, **variance = $\infty$** (heavy upper tail)

The agent uses a softmax policy: $P(A_i) \propto \exp(\theta_i)$, $\theta \in \mathbb{R}^3$.

### Three Policies

- $\pi_1$: **Risk-neutral** — $\max_\theta \mathbb{E}[Z]$. Trained with standard policy gradient.
- $\pi_2$: **Mean-semideviation** — $\max_\theta \mathbb{E}[Z] - \text{SD}[Z]$. Trained with the algorithm from Section 4.2.
- $\pi_3$: **Mean-standard-deviation** — $\max_\theta \mathbb{E}[Z] - \sqrt{\text{Var}[Z]}$. Trained using the algorithm of Tamar et al. (2012).

### Results

- $\pi_1$ (risk-neutral) converges to Asset 2 (highest mean return, $\mu = 4$) — rational.
- $\pi_2$ (mean-semideviation) converges to **Asset 3** — rational. Although $A_3$ has infinite variance, its *downside* risk is controlled (the heavy tail is on the *upside*). The semideviation correctly ignores upside variability.
- $\pi_3$ (mean-standard-deviation) converges to **Asset 1** — *irrational*. Despite $A_3$ stochastically dominating $A_1$, the standard deviation penalizes $A_3$'s heavy upper tail (high returns are treated as "risk"). The agent avoids the strictly superior asset.

- *Context*:
	- This is a clean demonstration of the danger of using the wrong risk measure. Standard deviation is symmetric — it penalizes outcomes both above and below the mean equally. For an investor, high returns are *good* variability, not something to penalize. The Pareto distribution's heavy upper tail produces infinite standard deviation, driving $\pi_3$ to the inferior Asset 1. The semideviation, by ignoring upside variability, makes the rational choice.
	- The term **"Risk Shaping"** is introduced here: the idea that the choice of risk measure (the shape of the risk envelope $\mathcal{U}$) should be carefully tailored to the problem domain, much as reward shaping tailors the cost function. The coherent risk framework provides the vocabulary for this design choice — different envelopes penalize different aspects of the cost distribution.
	- Asset 3 **stochastically dominates** Asset 1: for every return level $z$, the probability that $A_3$ exceeds $z$ is at least as large as the probability that $A_1$ exceeds $z$. Any rational decision-maker should prefer $A_3$ over $A_1$. Policy $\pi_3$ violates this basic rationality criterion.

---

# 7. Conclusion

The paper's contributions are:
1. A unified framework for risk-sensitive policy gradients covering the entire class of coherent risk measures.
2. Both static and dynamic (time-consistent) formulations, with provably consistent gradient estimators.
3. A demonstration that risk-measure design flexibility ("Risk Shaping") is essential for rational behaviour.

### Future Directions

- **Importance sampling** for improving convergence rates of gradient estimates, particularly for tail-sensitive measures like CVaR.
- **Risk Shaping** as a principled approach to **model misspecification**: the Representation Theorem (Theorem 2.1) connects risk to model uncertainty through the duality between coherent risk and worst-case expectations. By shaping the risk envelope, the decision-maker can implicitly protect against model errors. The procedure for choosing the "correct" envelope remains an open research question.

- *Context*:
	- The connection between coherent risk and model uncertainty is profound: optimizing a coherent risk measure is *equivalent* to optimizing expected cost under the worst-case model within an uncertainty set (the risk envelope). This means risk-averse RL and robust RL are two views of the same problem, connected by the Representation Theorem. This duality suggests that advances in distributional robustness can be imported into risk-sensitive RL and vice versa.

---

# MATHEMATICS

The mathematical framework proceeds from the axiomatic definition of coherent risk, through the Representation Theorem that converts risk into a tractable optimization, to the derivation of gradient formulas for both static and dynamic settings. The key technical tools are the Envelope Theorem (for differentiating through the risk envelope optimization), the likelihood-ratio trick (for converting gradients into expectations), and a Bellman decomposition (for extending static results to the dynamic case).

### 1. Coherent Risk and the Risk Envelope

A risk measure $\rho : \mathcal{Z} \to \mathbb{R}$ is **coherent** if it satisfies convexity, monotonicity, translation invariance, and positive homogeneity (axioms A1–A4). The **Representation Theorem** establishes the dual characterization:

**Representation Theorem ($\rho$):**

$$
\rho(Z) = \max_{\xi : \xi P_\theta \in \mathcal{U}(P_\theta)} \mathbb{E}_\xi[Z] = \max_{\xi : \xi P_\theta \in \mathcal{U}(P_\theta)} \sum_{\omega \in \Omega} P_\theta(\omega)\xi(\omega)Z(\omega) \tag{1}
$$

where $\mathcal{U}(P_\theta)$ is the **risk envelope** — a convex, bounded, closed subset of the probability simplex $\mathcal{B}$. The density function $\xi$ re-weights the base measure $P_\theta$; the product $\xi P_\theta$ defines an alternative probability measure. Every coherent risk measure corresponds to a unique risk envelope, and vice versa.

The risk envelope is given in **canonical convex programming form** (Assumption 2.2):

**Risk Envelope ($\mathcal{U}$):**

$$
\mathcal{U}(P_\theta) = \left\{\xi P_\theta : g_e(\xi, P_\theta) = 0, \; f_i(\xi, P_\theta) \leq 0, \; \sum_\omega \xi(\omega)P_\theta(\omega) = 1, \; \xi(\omega) \geq 0\right\} \tag{2}
$$

where $g_e$ are affine in $\xi$ (equality constraints, indexed by $e \in \mathcal{E}$), $f_i$ are convex in $\xi$ (inequality constraints, indexed by $i \in \mathcal{I}$), and Slater's condition holds. This structure guarantees strong duality when Eq. (1) is formulated as a constrained optimization.

### 2. The Lagrangian and Strong Duality

The Lagrangian of the constrained optimization in Eq. (1) is:

**Lagrangian ($L_\theta$):**

$$
L_\theta(\xi, \lambda_P, \lambda_E, \lambda_I) = \sum_\omega \xi(\omega) P_\theta(\omega) Z(\omega) - \lambda_P\!\left(\sum_\omega \xi(\omega)P_\theta(\omega) - 1\right) - \sum_{e \in \mathcal{E}} \lambda_E(e)\,g_e(\xi, P_\theta) - \sum_{i \in \mathcal{I}} \lambda_I(i)\,f_i(\xi, P_\theta) \tag{6}
$$

where $\lambda_P \in \mathbb{R}$ is the multiplier for the probability normalization constraint, $\lambda_E \in \mathbb{R}^{|\mathcal{E}|}$ for equality constraints, and $\lambda_I \in \mathbb{R}_+^{|\mathcal{I}|}$ for inequality constraints.

Slater's condition ensures:

$$
\rho(Z) = \max_{\xi \geq 0} \min_{\lambda_P, \lambda_I \geq 0, \lambda_E} L_\theta = \min_{\lambda_P, \lambda_I \geq 0, \lambda_E} \max_{\xi \geq 0} L_\theta
$$

The set of saddle points $\mathcal{S} = \{(\xi^*_\theta, \lambda^{*,P}_\theta, \lambda^{*,E}_\theta, \lambda^{*,I}_\theta)\}$ is non-empty and bounded.

### 3. Static Gradient via the Envelope Theorem

The **Envelope Theorem** for saddle-point problems (Milgrom & Segal, 2002, Theorem 4) states that the gradient of the optimal value equals the gradient of the Lagrangian evaluated at any saddle point:

$$
\nabla_\theta \rho(Z) = \nabla_\theta L_\theta(\xi, \lambda_P, \lambda_E, \lambda_I)\big|_{(\xi^*_\theta, \lambda^{*,P}_\theta, \lambda^{*,E}_\theta, \lambda^{*,I}_\theta)} \tag{10}
$$

Computing the right-hand side explicitly and applying the **likelihood-ratio trick** ($\nabla_\theta P_\theta(\omega) = P_\theta(\omega) \nabla_\theta \log P_\theta(\omega)$, justified by Assumption 4.1):

$$
\sum_\omega \xi(\omega) \nabla_\theta P_\theta(\omega) Z(\omega) - \lambda_P \sum_\omega \xi(\omega) \nabla_\theta P_\theta(\omega) = \sum_\omega \xi(\omega) P_\theta(\omega) \nabla_\theta \log P_\theta(\omega)(Z(\omega) - \lambda_P)
$$

**Static Gradient Formula (Theorem 4.2):**

$$
\nabla_\theta \rho(Z) = \mathbb{E}_{\xi^*_\theta}\!\left[\nabla_\theta \log P(\omega)(Z - \lambda^{*,P}_\theta)\right] - \sum_{e \in \mathcal{E}} \lambda^{*,E}_\theta(e)\,\nabla_\theta g_e(\xi^*_\theta; P_\theta) - \sum_{i \in \mathcal{I}} \lambda^{*,I}_\theta(i)\,\nabla_\theta f_i(\xi^*_\theta; P_\theta) \tag{T4.2}
$$

The first term is a $\xi^*$-weighted policy gradient with automatic baseline $\lambda^{*,P}_\theta$. The remaining terms correct for $\theta$-dependence in the risk envelope constraints.

### 4. CVaR Gradient as a Special Case

For CVaR at level $\alpha$, the risk envelope is $\mathcal{U} = \{\xi P_\theta : 0 \leq \xi(\omega) \leq \alpha^{-1}\}$, with no $\theta$-dependent constraints. The saddle point satisfies $\xi^*(\omega) = \alpha^{-1}$ when $Z(\omega) > q_\alpha$ and $\xi^*(\omega) = 0$ when $Z(\omega) < q_\alpha$, where $q_\alpha$ is the $(1-\alpha)$-quantile. Substituting into (T4.2):

**CVaR Gradient:**

$$
\nabla_\theta \rho_{\text{CVaR}}(Z; \alpha) = \mathbb{E}\!\left[\nabla_\theta \log P(\omega)(Z - q_\alpha) \mid Z(\omega) > q_\alpha\right] \tag{CVaR}
$$

### 5. Mean-Semideviation Gradient

The mean-semideviation risk $\rho_{\text{MSD}}(Z; \alpha) = \mathbb{E}[Z] + \alpha \cdot \text{SD}[Z]$ has the dual representation:

$$
\rho_{\text{MSD}}(Z) = \sup_{\|\xi\|_2 \leq 1, \, \xi \geq 0} \langle 1 + c\xi - c\mathbb{E}[\xi],\; Z \rangle \tag{12}
$$

The optimal $\bar{\xi}$ is the "contact point" $\bar{\xi} = (Z - \mathbb{E}[Z])_+ / \text{SD}(Z)$. Since the constraints in Eq. (12) do not depend on $\theta$, the Envelope Theorem yields:

**Mean-Semideviation Gradient (Proposition 4.3/B.1):**

$$
\nabla_\theta \rho_{\text{MSD}}(Z; \alpha) = \nabla_\theta \mathbb{E}[Z] + \frac{\alpha}{\text{SD}(Z)}\,\mathbb{E}\!\left[(Z - \mathbb{E}[Z])_+\!\left(\nabla_\theta \log P(\omega)(Z - \mathbb{E}[Z]) - \nabla_\theta \mathbb{E}[Z]\right)\right] \tag{MSD}
$$

### 6. Sample Average Approximation (SAA)

For general coherent risk, the gradient is estimated by replacing the true distribution with the empirical distribution and solving the resulting finite-dimensional convex program. Given $N$ samples $\omega_1, \ldots, \omega_N \sim P_\theta$:

**SAA Gradient Estimator:**

$$
\nabla_{\theta;N} \rho(Z) = \sum_{i=1}^N P_{\theta;N}(\omega_i)\xi^*_{\theta;N}(\omega_i)\nabla_\theta \log P(\omega_i)(Z(\omega_i) - \lambda^{*,P}_{\theta;N}) - \sum_{e} \lambda^{*,E}_{\theta;N}(e)\nabla_\theta g_e - \sum_{i} \lambda^{*,I}_{\theta;N}(i)\nabla_\theta f_i \tag{8}
$$

where $(\xi^*_{\theta;N}, \lambda^{*,P}_{\theta;N}, \lambda^{*,E}_{\theta;N}, \lambda^{*,I}_{\theta;N})$ solve the empirical version of the Lagrangian. Proposition 4.4 establishes $\nabla_{\theta;N}\rho(Z) \to \nabla_\theta \rho(Z)$ w.p. 1 as $N \to \infty$.

### 7. Risk-Sensitive Bellman Equation

For the dynamic case, the risk-sensitive Bellman operator $T_\theta : \mathcal{B}(\mathcal{X}) \to \mathcal{B}(\mathcal{X})$ is:

**Risk-Sensitive Bellman Operator ($T_\theta$):**

$$
T_\theta[V](x) := C_\theta(x) + \gamma \max_{\xi P_\theta(\cdot|x) \in \mathcal{U}(x, P_\theta(\cdot|x))} \mathbb{E}_\xi[V] \tag{22}
$$

where $C_\theta(x) = \sum_a C(x,a)\mu_\theta(a|x)$. By Ruszczyński (2010, Theorem 1), $T_\theta$ has a unique fixed point $V_\theta$ satisfying $T_\theta[V_\theta](x) = V_\theta(x)$ for all $x$, with $V_\theta(x_0) = \rho_\infty(M)$.

Under Assumption E.2 ($\xi(x') \leq \kappa/\gamma$ for all feasible $\xi$), the projected operator $\Pi T_\theta$ is a contraction in the $d_\theta$-weighted norm, guaranteeing convergence of the PRSVI algorithm.

### 8. Dynamic Policy Gradient Theorem

Applying the Envelope Theorem to the risk-sensitive Bellman equation at each state $x$ and unfolding the resulting recursion:

**Dynamic Policy Gradient (Theorem 5.2/E.4):**

$$
\nabla V_\theta(x) = \mathbb{E}_{\xi^*_\theta}\!\left[\sum_{t=0}^{\infty} \gamma^t \nabla_\theta \log \mu_\theta(a_t|x_t)\,h_\theta(x_t, a_t) \;\middle|\; x_0 = x\right] \tag{T5.2}
$$

where trajectories follow $\xi^*$-weighted transitions $P_\theta(\cdot|x)\xi^*_{\theta,x}(\cdot)$, and:

**Stage-wise Cost ($h_\theta$):**

$$
h_\theta(x, a) = C(x,a) + \sum_{x'} P(x'|x,a)\xi^*_{\theta,x}(x')\!\left[\gamma V_\theta(x') - \lambda^{*,P}_{\theta,x} - \sum_{i \in \mathcal{I}} \lambda^{*,I}_{\theta,x}(i)\frac{df_i}{dp(x')} - \sum_{e \in \mathcal{E}} \lambda^{*,E}_{\theta,x}(e)\frac{dg_e}{dp(x')}\right] \tag{26}
$$

The proof proceeds by differentiating the Bellman equation at each state using the Envelope Theorem, defining $\hat{h}_\theta(x) = \sum_a \mu_\theta(a|x)\nabla_\theta \log \mu_\theta(a|x)h_\theta(x,a)$, and unfolding the recursion $\nabla V_\theta(x_0) = \hat{h}_\theta(x_0) + \gamma \sum_{x_1} P_\theta(x_1|x_0)\xi^*_\theta(x_1)[\hat{h}_\theta(x_1) + \cdots]$. The infinite sum converges by bounded convergence ($\gamma^t \nabla V_\theta(x) \to 0$ as $t \to \infty$).

### 9. Gradient Estimation Consistency Under Function Approximation

When the critic uses $V_\theta(x) \approx v^{*\top}_\theta \phi(x)$ with approximation error $\Delta = \|\Phi v^*_\theta - V_\theta\|_\infty$, the gradient estimate satisfies:

**Approximation Error Bound (Theorem E.5):**

$$
\left|\lim_{N \to \infty} \frac{1}{N}\sum_{j=1}^N \sum_{t=0}^{\infty} \gamma^t \nabla \log \mu_\theta(a_t^{(j)}|x_t^{(j)})\,h_{\theta,N}(x_t^{(j)}, a_t^{(j)}) - \nabla V_\theta(x_0)\right| = O(\Delta) \tag{TE.5}
$$

The proof uses Proposition G.1 (sensitivity of KKT solutions to objective perturbations, via the Implicit Function Theorem) to bound the error in the saddle points and stage-wise costs introduced by function approximation, then Lemma G.3 to bound the error in the occupancy measure.

---

### **1. Problem:**

The paper addresses the fragmentation and specificity of risk-sensitive policy gradient methods. Prior work developed gradient formulas for individual risk measures (variance, CVaR) in isolation, with no shared structure or reusable methodology. The gap is the absence of a *unified* policy gradient framework that covers the entire class of coherent risk measures — both static and time-consistent dynamic — enabling practitioners to choose problem-appropriate risk criteria without re-deriving gradient estimators from scratch.

### **2. Setup:**

The method assumes a finite-horizon discounted MDP $(\mathcal{X}, \mathcal{A}, C, P, \gamma, x_0)$ with bounded deterministic costs, parameterized stationary Markov policies $\mu_\theta(\cdot|x)$, and a finite sample space $\Omega$ (results extend to $L_p$ spaces). The risk envelope $\mathcal{U}(P_\theta)$ must be known in canonical convex programming form with affine equality constraints, convex inequality constraints, and a strictly feasible point. For the dynamic case, an additional contraction condition ($\xi(x') \leq \kappa/\gamma$) is required. Simulator access is assumed for obtaining i.i.d. samples from transition dynamics.

### **3. Key Idea:**

The core contribution is the insight that the Representation Theorem — characterizing any coherent risk as a worst-case expectation over a convex set of distributions — combined with the Envelope Theorem, yields a general-purpose gradient formula for coherent risk that decomposes into a re-weighted policy gradient plus constraint correction terms. This formula is modular: for static risk, it is used directly; for dynamic risk, it is composed with a Bellman decomposition to produce a generalized Policy Gradient Theorem.

### **4. Assumptions:**

**Explicit:**
- The risk envelope is known in an explicit convex programming form (Assumption 2.2) with smooth constraints
- Slater's condition (strict feasibility) holds for the risk envelope
- The likelihood ratio $\nabla_\theta \log P(\omega)$ (static) or $\nabla_\theta \log \mu_\theta(a|x)$ (dynamic) is well-defined and bounded
- For function approximation: the feature mapping $\Phi$ has full column rank (Assumption E.1)
- The contraction condition $\xi(x') \leq \kappa/\gamma$ for all feasible densities (Assumption E.2)

**Implicit:**
- Simulator access for obtaining i.i.d. samples from $P(\cdot|x, a)$
- Availability of efficient convex programming solvers (interior point methods) at every policy update step
- The finite sample space restriction is a technical simplification; results extend to $L_p$ spaces but details are omitted
- The policy parameterization is rich enough for the locally optimal $\theta$ to be meaningful
- The convex program solution varies smoothly with $\theta$ (required for the Envelope Theorem, but follows from the regularity assumptions)

### **5. Limitation:**

- **Computational overhead:** Solving a convex optimization problem at every policy update step is expensive compared to risk-neutral policy gradient, which requires only a sample average. For measures without closed-form saddle points, this involves calling an interior point solver at every step.
- **Sample efficiency:** Gradient estimates for tail-sensitive risk measures (especially CVaR) converge slowly because they depend on rare events. The paper identifies importance sampling as a future direction but provides no solution.
- **Risk Shaping is ad hoc:** The paper argues that choosing the right risk measure is critical (the numerical experiment demonstrates this) but provides no principled procedure for selecting the risk envelope. The "correct" choice remains an open question.
- **Function approximation error:** The actor-critic gradient is accurate only up to the critic's approximation error $\Delta$. No convergence rate is provided for the full actor-critic algorithm — only consistency of the gradient estimator.
- **Finite action and state space assumptions:** The dynamic gradient theorem requires finite $\mathcal{X}$ and $\mathcal{A}$. Extension to continuous spaces is not addressed.

### **6. Relevance & Open Questions:**

This paper provides the theoretical backbone for risk-sensitive policy gradient methods in RL. Its framework is directly relevant to portfolio optimization, where the choice of risk measure (CVaR, semideviation, spectral risk) is a fundamental design decision. Key open questions and connections:

- **Risk Shaping as a design methodology:** How should the risk envelope $\mathcal{U}$ be chosen for a specific application? Can it be *learned* from data or expert preferences, rather than specified a priori?
- **Connection to distributional RL:** [[PNOTES - A Distributional Perspective on Reinforcement Learning]] learns the full return distribution, which is the natural input to any coherent risk measure. Can the distributional RL infrastructure (categorical distributions, quantile regression) be combined with the policy gradient framework here to avoid the convex programming step entirely?
- **Importance sampling for tail risk:** The paper flags slow convergence for CVaR gradients. Subsequent work on importance sampling and adaptive sampling for rare events is essential for practical deployment.
- **Robust RL duality:** The equivalence between coherent risk and robust MDPs (Osogami, 2012) suggests that advances in distributionally robust optimization can be directly imported. Can the risk envelope be parameterized as a Wasserstein ball or f-divergence ball for principled model uncertainty handling?
- **Continuous action spaces:** The dynamic gradient theorem assumes finite actions. Extension to continuous action spaces (via reparameterization or compatible function approximation) is needed for robotics and continuous portfolio allocation.

---

### Integration:

* **Problem:** This paper fills a critical gap for risk-sensitive RL research: it provides a *general-purpose* policy gradient framework rather than a one-off derivation for each risk measure. For portfolio RL, where the choice between CVaR, semideviation, entropic risk, and other coherent measures is a foundational design decision, this framework means the gradient machinery is settled once — the research question shifts to *which risk envelope to use*, not *how to compute the gradient*. The Representation Theorem's duality between risk and model uncertainty also positions this work as a bridge between risk-sensitive and robust RL, two perspectives that converge naturally in financial applications where both stochastic risk and model misspecification are first-order concerns.

* **Limitation:** The most pressing limitation for practical portfolio RL is the **computational overhead** of solving a convex program at every policy update. In high-frequency or large-state-space settings, this bottleneck may be prohibitive. The **sample efficiency** problem for tail-sensitive measures like CVaR is equally concerning: portfolio risk management is fundamentally about tail events, yet the gradient estimates for tail-sensitive objectives converge slowly precisely because those events are rare. The absence of a principled **Risk Shaping** procedure is also a gap — the numerical experiment convincingly demonstrates that the wrong risk measure produces irrational behavior, but the paper offers no methodology for choosing the right one beyond domain expertise and experimentation.