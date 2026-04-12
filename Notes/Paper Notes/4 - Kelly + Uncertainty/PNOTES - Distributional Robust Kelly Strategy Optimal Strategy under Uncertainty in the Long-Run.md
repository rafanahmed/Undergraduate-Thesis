[[Distributional Robust Kelly Strategy - Optimal Strategy under Uncertainty in the Long-Run.pdf]]

# **Abstract:**

The paper proposes the **Distributional Robust Kelly Problem (DRKP)**, a robust extension of the classic Kelly gambling framework that accounts for *distributional uncertainty*. Where the classic Kelly strategy maximizes expected log growth under a *known* probability distribution, the DRKP maximizes the *worst-case* (smallest) expected log growth across a prescribed set of possible distributions.

- The classic Kelly strategy, despite its strong theoretical properties (asymptotic growth rate maximization, magnitude dominance), is rarely used in practice because the true probability distribution is never known.
	- The **optimizer's curse** arises: decisions optimized under an estimated distribution perform poorly out-of-sample.
- The DRKP addresses this by replacing the point estimate $\pi$ with an **uncertainty set** $\Xi$ of plausible distributions and optimizing against the worst member of that set.

The paper contributes along three axes:

1. **Computational tractability:** For a large class of uncertainty sets — polyhedral, box, ellipsoidal, $f$-divergence balls, Wasserstein balls, and moment-based sets — the DRKP is reformulated into **Disciplined Convex Programming (DCP)** forms solvable via CVXPY (Theorems 1–6).
2. **Theoretical optimality:** Extensions of Breiman's classical results to the robust setting, proving that the DRKP asymptotically maximizes the worst-case growth rate and dominates any essentially different strategy by magnitude (Theorems 7–8).
3. **Numerical validation:** A horse racing simulation demonstrates significant worst-case growth improvement over the standard Kelly bet.

- *Context*:
	- The classic Kelly strategy sits at the intersection of information theory and portfolio optimization. Kelly (1956) showed that maximizing expected log growth is equivalent to maximizing the long-run geometric growth rate of wealth. Breiman (1961) proved the two defining optimality properties: (a) the Kelly bet asymptotically maximizes wealth growth, and (b) it dominates any "essentially different" strategy by magnitude — meaning the ratio of Kelly wealth to alternative wealth goes to infinity. These results, however, assume the gambler *knows* the true probability distribution. This paper's central question is: can we preserve these optimality guarantees when the distribution is uncertain?
	- The paper bridges two major fields: the Kelly criterion literature (information theory, portfolio theory) and distributional robust optimization (convex optimization, operations research). The synthesis is natural — both fields deal with decision-making under probability distributions — but the technical challenge lies in showing that robustifying the Kelly problem does not destroy either tractability or asymptotic optimality.

---

# 1. Introduction

The paper frames the problem by establishing the classic Kelly strategy and its practical limitations.

## The Classic Kelly Strategy

**Kelly gambling** was proposed by John Kelly in 1956. The gambler repeatedly allocates fractions of her wealth across $n$ bets. The allocation $\mathbf{b} \in \mathcal{S}_n$ (the probability simplex) determines how wealth is distributed, with $b_i$ the fraction bet on outcome $i$.

- Returns $\mathbf{r} \in \mathbb{R}^n_+$ are random: $r_i \geq 0$ is the payoff per dollar on bet $i$.
- Wealth is multiplied each round by the factor $\mathbf{r}^\top \mathbf{b}$.
- Since wealth compounds multiplicatively, the log of wealth over time is a **random walk** with increment $\log(\mathbf{r}^\top \mathbf{b})$.

In the **finite outcome case** with $K$ events, the return vectors $\mathbf{r}_1, \dots, \mathbf{r}_K$ form a payoff matrix $\mathbf{R} \in \mathbb{R}^{n \times K}$, and the **mean log growth rate** is:

$$
G_\pi(\mathbf{b}) = \mathbb{E}_\pi[\log(\mathbf{r}^\top \mathbf{b})] = \boldsymbol{\pi}^\top \log(\mathbf{R}^\top \mathbf{b}) = \sum_{k=1}^{K} \pi_k \log(\mathbf{r}_k^\top \mathbf{b})
$$

The classic Kelly problem maximizes $G_\pi(\mathbf{b})$ subject to $\mathbf{b} \in \mathcal{B}$, where $\mathcal{B} \subseteq \mathcal{S}_n$ is a convex constraint set. This is a convex optimization problem because $G_\pi(\mathbf{b})$ is concave in $\mathbf{b}$.

- *Context*:
	- The concavity of $G_\pi(\mathbf{b})$ follows from the composition: $\mathbf{R}^\top \mathbf{b}$ is linear in $\mathbf{b}$, and $\log(\cdot)$ is concave, so each term $\log(\mathbf{r}_k^\top \mathbf{b})$ is concave. The non-negative weighted sum $\sum \pi_k \log(\mathbf{r}_k^\top \mathbf{b})$ preserves concavity. This means any local maximum is a global maximum, and standard convex solvers can find it efficiently.
	- The assumption $r_n = 1$ almost surely designates the $n$-th bet as a **cash position** — the gambler holds that fraction of wealth without risk. The allocation $\mathbf{b} = \mathbf{e}_n = (0, \dots, 0, 1)$ corresponds to not betting at all. This ensures the gambler always has a "safe" option, preventing forced total loss.
	- The connection to the Kelly Criterion as used in portfolio optimization is discussed in [[PNOTES - Deep Reinforcement Learning for Optimal Asset Allocation Using DDPG with TiDE]], where the log-wealth reward $r_t = \log(W_{t+1}/W_t)$ is derived from the same Kelly framework applied to continuous asset allocation via DDPG.

## The Uncertainty Problem

The paper identifies three sources of uncertainty that undermine the classic Kelly strategy:

1. **In-sample / out-of-sample gap:** The empirical distribution from training data differs from the true distribution. Optimal decisions under the empirical distribution often disappoint out-of-sample — the **optimizer's curse**.
2. **Distribution shift:** The environment changes between model training and deployment. In finance, return time series are typically *non-stationary*.
3. **Estimation error:** Even first-moment (mean) and second-moment (covariance) estimates carry significant noise. Higher-order distributional information may be uninformative due to measurement noise.

- *Context*:
	- The optimizer's curse is a well-documented phenomenon in decision analysis. When you optimize over estimated parameters, the optimizer systematically selects parameter configurations where noise *happens* to make the objective look better than it truly is. The more parameters you optimize over (i.e., the richer the distribution model), the worse the curse. This creates a paradox: more detailed distributional models are more accurate in principle but more vulnerable to overfitting in practice. Robust optimization breaks this paradox by hedging against the worst distribution in a set, explicitly trading nominal performance for out-of-sample resilience.

## The Distributional Robust Kelly Problem

The DRKP replaces the known distribution $\pi$ with an uncertainty set $\Xi$ of possible distributions. The **worst-case log growth rate** is:

$$
G_\Xi(\mathbf{b}) = \inf_{\pi \in \Xi} G_\pi(\mathbf{b})
$$

The DRKP is:

$$
\max_{\mathbf{b} \in \mathcal{B}} \quad \inf_{\pi \in \Xi} \mathbb{E}_\pi[\log(\mathbf{r}^\top \mathbf{b})]
$$

This is a convex optimization problem because $G_\Xi(\mathbf{b})$ is an infimum of concave functions (hence concave). However, convexity does not automatically imply tractability — the paper's computational contribution is showing that for specific, practically useful forms of $\Xi$, the DRKP can be reformulated as a DCP problem.

- *Context*:
	- The distinction between convexity and tractability is subtle but critical. A convex problem is one where any local minimum is global, but that does not mean an efficient algorithm exists to find it. **DCP tractability** is a stronger property: the problem can be expressed in a form that obeys the composition rules of disciplined convex programming, allowing automatic transformation to a conic program solvable by interior-point methods. The paper uses "DCP tractable" in this strict sense — expressible in CVXPY.

## Related Work

The paper situates itself within two literatures:

- **Uncertainty aversion / ambiguity aversion:** In decision theory, *risk* refers to situations with known probabilities, while *uncertainty* refers to unknown probabilities. Uncertainty aversion (Ellsberg, 1961; Fox & Tversky, 1995) provides a behavioral foundation for max-min expected utility — exactly the DRKP's objective.
- **Distributional robust optimization (DRO):** A well-studied field with work on moment-based constraints (Delage & Ye, 2010), $f$-divergence balls (Namkoong & Duchi, 2016), and Wasserstein balls (Blanchet et al., 2018; Esfahani & Kuhn, 2017). The paper applies this DRO machinery specifically to the log-growth objective.

- *Context*:
	- The distinction between risk and uncertainty traces back to Frank Knight (1921). Knightian uncertainty — situations where probabilities themselves are unknown — is the precise setting of the DRKP. The max-min formulation is not merely conservative; it has axiomatic behavioral foundations in the work of Gilboa & Schmeidler (1989), who showed that agents who satisfy certain rationality axioms under uncertainty will behave as if maximizing the minimum expected utility over a set of priors.

---

# 2. DCP Tractable Forms for Distributional Robust Kelly Strategy

This is the paper's central computational contribution. For each uncertainty set, the authors derive a DCP tractable reformulation of the DRKP using **Lagrangian duality**. The strategy is consistent: express $G_\Xi(\mathbf{b})$ as the value of a minimization over $\pi$, form the Lagrangian dual, verify strong duality (via Slater's condition), and show that the resulting dual problem follows DCP rules.

## Theorem 1: Polyhedron Uncertainty Set

For $\Xi = \{\pi \in \mathcal{S}_K \mid \mathbf{A}_0 \pi = \mathbf{d}_0, \; \mathbf{A}_1 \pi \leq \mathbf{d}_1\}$:

$$
\max \quad \min(\log(\mathbf{R}^\top \mathbf{b}) + \mathbf{A}_0^\top \nu + \mathbf{A}_1^\top \lambda) - \mathbf{d}_0^\top \nu - \mathbf{d}_1^\top \lambda
$$

$$
\text{subject to} \quad \mathbf{b} \in \mathcal{B}, \quad \lambda \geq 0
$$

with variables $\mathbf{b}, \nu, \lambda$.

- *Context*:
	- The derivation proceeds by writing $G_\Xi(\mathbf{b})$ as the optimal value of a linear program (LP) in $\pi$, then taking its LP dual. The simplex constraint $\pi \in \mathcal{S}_K$ is absorbed via an indicator function, and the linear constraints are dualized. The $\min(\cdot)$ operation (minimum component of a vector) arises naturally from minimizing a linear objective over the simplex — the minimum is attained at a vertex $\mathbf{e}_k$, selecting the smallest component. Strong duality holds by Slater's condition: if the polyhedron has a strict interior point in $\mathcal{S}_K$, the primal and dual values coincide.
	- This is DCP tractable because $\min(\cdot)$ of an affine expression is concave, and the remaining terms are linear.

## Theorem 2: Box Uncertainty Set

For $\Xi = \{\pi \in \mathcal{S}_K \mid |\pi - \pi_{\text{nom}}| \leq \boldsymbol{\rho}\}$, where $\pi_{\text{nom}}$ is the nominal distribution and $\boldsymbol{\rho}$ is a vector of radii:

$$
\max \quad \min(\log(\mathbf{R}^\top \mathbf{b}) + \mu) - \pi_{\text{nom}}^\top \mu - \boldsymbol{\rho}^\top |\mu|
$$

$$
\text{subject to} \quad \mathbf{b} \in \mathcal{B}
$$

with variables $\mathbf{b}, \mu$.

- *Context*:
	- The box set is the most intuitive uncertainty model: each probability $\pi_k$ can deviate from its nominal value $\pi_{\text{nom},k}$ by at most $\rho_k$. The derivation specializes Theorem 1 by encoding the box constraints as $\mathbf{A}_1 \pi \leq \mathbf{d}_1$ with block structure, then simplifying using the substitution $\mu = \lambda^+ - \lambda^-$ and $|\mu| = \lambda^+ + \lambda^-$.
	- The term $\boldsymbol{\rho}^\top |\mu|$ acts as a **regularizer** — larger uncertainty radii $\rho$ penalize the dual variables more heavily, forcing the objective to be more conservative. When $\boldsymbol{\rho} = \mathbf{0}$, the problem reduces to the classic Kelly problem.

## Theorem 3: Ellipsoidal Uncertainty Set

For $\Xi = \{\pi \in \mathcal{S}_K \mid \|\mathbf{W}^{-1}(\pi - \pi_{\text{nom}})\|_p \leq 1\}$, where $\mathbf{W}$ is nonsingular:

$$
\max \quad \pi_{\text{nom}}^\top \mathbf{u} - \|\mathbf{W}^\top(\mathbf{u} - \mu \mathbf{1})\|_q
$$

$$
\text{subject to} \quad \mathbf{u} \leq \log(\mathbf{R}^\top \mathbf{b}), \quad \mathbf{b} \in \mathcal{B}
$$

with variables $\mathbf{b}, \mathbf{u}, \mu$, and $q$ defined by $1/p + 1/q = 1$ (Hölder conjugate).

- *Context*:
	- The ellipsoidal set generalizes the box: the matrix $\mathbf{W}$ defines the shape and orientation of the ellipsoid in probability space. When $\mathbf{W} = \text{diag}(\boldsymbol{\rho})$ and $p = \infty$, this reduces to the box set. The $p$-norm parametrization allows tuning the geometry — $p = 2$ gives a standard ellipsoid, $p = 1$ gives an $\ell_1$ ball (a diamond), $p = \infty$ gives a box.
	- The key duality step uses **Hölder's inequality**: $\sup_{\|\mathbf{z}\|_p \leq 1} \mathbf{z}^\top \mathbf{W}^\top \mathbf{x} = \|\mathbf{W}^\top \mathbf{x}\|_q$. The $q$-norm of an affine expression is convex, so $-\|\mathbf{W}^\top(\mathbf{u} - \mu \mathbf{1})\|_q$ is concave, preserving DCP compliance.

## Theorem 4: $f$-Divergence Ball

For $\Xi = \{\pi \in \mathcal{S}_K \mid D_f(\pi \| \pi_{\text{nom}}) \leq \epsilon\}$, where $D_f$ is an $f$-divergence with generating function $f$:

$$
\max \quad -\pi_{\text{nom}}^\top \mathbf{w} - \epsilon \lambda - \beta
$$

$$
\text{subject to} \quad \mathbf{w} \geq \lambda f^*\left(\frac{\mathbf{z}}{\lambda}\right), \quad \mathbf{z} \geq -\log(\mathbf{R}^\top \mathbf{b}) - \beta, \quad \lambda \geq 0, \quad \mathbf{b} \in \mathcal{B}
$$

with variables $\mathbf{b}, \lambda, \beta, \mathbf{w}, \mathbf{z}$, where $f^*(s) = \sup_{t \geq 0}(ts - f(t))$ is the **Fenchel conjugate** of $f$.

- *Context*:
	- The $f$-divergence is a general class of statistical distances. The generating function $f$ determines the specific divergence: $f(t) = t\log t - t + 1$ gives KL-divergence, $f(t) = -\log t + t - 1$ gives reverse KL, $f(t) = \frac{1}{2}(t-1)^2$ gives Pearson $\chi^2$. The paper provides a full catalogue of these in its supplementary material (Section 6).
	- The expression $\lambda f^*(z/\lambda)$ is the **perspective function** of $f^*$. If $f^*$ is convex and non-decreasing, its perspective is also convex and non-decreasing, which is essential for DCP compliance. The composition rule then applies: $-\log(\mathbf{R}^\top \mathbf{b}) - \beta$ is convex in $(\mathbf{b}, \beta)$, and feeding it into a non-decreasing convex function preserves convexity.
	- The $f$-divergence ball requires $\pi$ to be **absolutely continuous** with respect to $\pi_{\text{nom}}$ — that is, $\pi$ can only assign positive probability to outcomes where $\pi_{\text{nom}}$ is also positive. This means the $f$-divergence set cannot model "black swan" events that have zero nominal probability. This limitation motivates the Wasserstein set.

## Theorem 5: Wasserstein Distance Ball

For $\Xi = \{\pi \in \mathcal{S}_K \mid D_c(\pi, \pi_{\text{nom}}) \leq s\}$, where $D_c$ is the Wasserstein distance with cost matrix $\mathbf{C} \in \mathbb{R}^{K \times K_{\text{nom}}}_+$:

$$
\max \quad \sum_j \pi_{\text{nom},j} \min_i \left(\log(\mathbf{R}^\top \mathbf{b})_i + \lambda c_{ij}\right) - s\lambda
$$

$$
\text{subject to} \quad \mathbf{b} \in \mathcal{B}, \quad \lambda \geq 0
$$

with dual variable $\lambda \in \mathbb{R}_+$.

- *Context*:
	- The **Wasserstein distance** (also called the earth-mover distance) measures the minimum cost of transporting probability mass from $\pi_{\text{nom}}$ to $\pi$, with $c_{ij}$ as the cost of moving one unit of mass from outcome $j$ to outcome $i$. Unlike $f$-divergence, the Wasserstein distance does **not** require $\pi$ to be absolutely continuous with respect to $\pi_{\text{nom}}$. This means the uncertainty set can include distributions that assign probability to outcomes with zero nominal probability — critical for hedging against **black swan events**.
	- The derivation uses LP duality on the optimal transport formulation. The inner $\min_i(\cdot)$ arises from the dual of the transport constraint, and the resulting objective is concave because $\log(\mathbf{R}^\top \mathbf{b})_i + \lambda c_{ij}$ is concave in $(\mathbf{b}, \lambda)$ and $\min$ preserves concavity.
	- This connects to the Wasserstein metric used in [[PNOTES - A Distributional Perspective on Reinforcement Learning]], where the Wasserstein distance was the key metric for proving contraction of the distributional Bellman operator. In that RL context, the Wasserstein distance measured closeness of value distributions; here it measures closeness of probability distributions over outcomes.

## Theorem 6: Mean and Covariance Uncertainty (SDP)

For uncertainty in the *moments* of the return vector $\mathbf{r} \in \mathbb{R}^n$:

$$
\Xi = \left\{\pi \in \mathcal{S}_K \;\middle|\; (\mathbb{E}_\pi[\mathbf{r}] - \boldsymbol{\mu}_0)^\top \boldsymbol{\Sigma}_0^{-1} (\mathbb{E}_\pi[\mathbf{r}] - \boldsymbol{\mu}_0) \leq \gamma_1, \quad \mathbb{E}_\pi[(\mathbf{r} - \boldsymbol{\mu}_0)(\mathbf{r} - \boldsymbol{\mu}_0)^\top] \preceq \gamma_2 \boldsymbol{\Sigma}_0 \right\}
$$

The DRKP is equivalent to the **Semidefinite Program (SDP)**:

$$
\min \quad u_1 + u_2
$$

$$
\text{subject to} \quad u_1 \geq -\log(\mathbf{r}_i^\top \mathbf{b}) - \mathbf{r}_i^\top \mathbf{Y} \mathbf{r}_i - \mathbf{r}_i^\top \mathbf{y}, \quad \forall\, i = 1, \dots, K
$$

$$
u_2 \geq (\gamma_2 \boldsymbol{\Sigma}_0 + \boldsymbol{\mu}_0 \boldsymbol{\mu}_0^\top) \bullet \mathbf{Y} + \boldsymbol{\mu}_0^\top \mathbf{y} + \sqrt{\gamma_1} \|\boldsymbol{\Sigma}_0^{1/2}(\mathbf{y} + 2\mathbf{Y}\boldsymbol{\mu}_0)\|
$$

$$
\mathbf{Y} \succeq 0, \quad \mathbf{b} \in \mathcal{B}
$$

with auxiliary variables $u_1, u_2 \in \mathbb{R}$, $\mathbf{Y} \in \mathbb{R}^{n \times n}$, $\mathbf{y} \in \mathbb{R}^n$.

- *Context*:
	- This formulation follows Delage & Ye (2010) and is especially relevant for **stock investment**, where practitioners typically only have reliable estimates of means and covariances — and even those have significant estimation error. The bounds $\gamma_1$ and $\gamma_2$ quantify confidence in the mean and covariance estimates, respectively.
	- The notation $\bullet$ denotes the **Frobenius inner product** (trace of the matrix product). The constraint $\mathbf{Y} \succeq 0$ (positive semidefiniteness) is what makes this an SDP rather than a simpler conic program. SDPs are more expensive to solve than linear or second-order cone programs, but remain tractable — interior-point methods solve them in polynomial time.
	- The Slater condition for strong duality requires a strictly feasible $\pi_0 \in \Xi$ with $\mathbb{E}_{\pi_0}[\mathbf{r}] = \boldsymbol{\mu}_0$ and $\mathbb{E}_{\pi_0}[(\mathbf{r} - \boldsymbol{\mu}_0)(\mathbf{r} - \boldsymbol{\mu}_0)^\top] = \boldsymbol{\Sigma}_0$. This is a non-trivial assumption — it requires that the nominal moments are *achievable* by some distribution in the finite-outcome space.

---

# 3. Theoretical Properties of Distributional Robust Kelly Strategy

This section extends Breiman's classical optimality results to the robust setting. The key innovation is the use of **non-linear expectation theory** to handle the $\inf_{\pi \in \Xi} \mathbb{E}_\pi[\cdot]$ operator.

## Sequential Setting

The paper considers a sequential gambling setting with a fixed uncertainty set $\Xi$:

- At period $N$, the return vector $\mathbf{r}_N$ has distribution $\pi_N \in \Xi$, which may vary freely across periods.
- Returns are bounded: $r_{N,i} \in [0, R_M]$ for all $i, N$.
- A strategy $\Lambda = (\mathbf{b}_1, \dots, \mathbf{b}_N, \dots)$ produces period-$N$ growth $V_N := \mathbf{r}_N^\top \mathbf{b}_N$.
- Accumulated wealth: $S_0 = 1$, $S_{N+1} = S_N \cdot V_N$.

The **distributional robust bet** $\mathbf{b}_N^*$ maximizes $\inf_{\pi_N \in \Xi} \mathbb{E}_{\pi_N}[\log(\mathbf{r}^\top \mathbf{b}) \mid \mathcal{O}_{N-1}]$, where $\mathcal{O}_{N-1}$ denotes the observed outcomes through period $N-1$.

## Theorem 7: Asymptotic Maximality

For any strategy $\Lambda$ producing wealth $S_N$:

$$
\lim_N \frac{S_N}{S_N^*} \text{ exists almost surely, and } \inf_{\pi_1, \dots, \pi_N \in \Xi} \mathbb{E}\left[\lim_N \frac{S_N}{S_N^*}\right] \leq 1
$$

- *Context*:
	- This theorem says the ratio of any strategy's wealth to the robust Kelly wealth is a **decreasing semimartingale** under the non-linear expectation operator $\bar{\mathbb{E}} := \inf_{\pi_N \in \Xi} \mathbb{E}_{\pi_N}$. In plain terms: no strategy can systematically outgrow the robust Kelly strategy in the worst case. The limit exists (convergence is guaranteed), and the worst-case expected value of the limiting ratio is at most 1.
	- The proof technique is elegant. The maximizing property of $\mathbf{b}_N^*$ implies that for any $\epsilon > 0$: $\inf_{\pi_N \in \Xi} \mathbb{E}[\epsilon \log V_N + (1 - \epsilon) \log V_N^* \mid \mathcal{O}_{N-1}] \leq \inf_{\pi_N \in \Xi} \mathbb{E}[\log V_N^* \mid \mathcal{O}_{N-1}]$. After algebraic manipulation using the **superlinearity** of the non-linear expectation operator — $\bar{\mathbb{E}}(X + Y) \geq \bar{\mathbb{E}}(X) + \bar{\mathbb{E}}(Y)$ — and taking $\epsilon \to 0^+$ via **Fatou's lemma**, the inequality $\inf_{\pi_N \in \Xi} \mathbb{E}[V_N / V_N^* \mid \mathcal{O}_{N-1}] \leq 1$ is established.
	- This extends Breiman's (1961) original result, which assumed a *fixed, known* distribution. Here the distribution can change adversarially within $\Xi$ at every period.

## Theorem 8: Magnitude Dominance

If $\Lambda$ is a **nonterminating strategy** (no period exists where $V_s = 0$), then:

$$
\left\{\lim_{N \to \infty} \frac{S_N}{S_N^*} = 0\right\} = \left\{\sum_{N=1}^\infty \left[\inf_{\pi_N \in \Xi} \mathbb{E}_{\pi_N}[\log V_N^* - \log V_N \mid \mathcal{O}_{N-1}]\right] = \infty\right\} \quad \text{a.s.}
$$

- *Context*:
	- This is the robust analogue of Breiman's magnitude dominance. It says: the robust Kelly strategy dominates any "essentially different" strategy by magnitude almost surely. Two strategies are **essentially different under uncertainty** if $\sum_{N=1}^\infty [\inf_{\pi_N \in \Xi} \mathbb{E}_{\pi_N}[\log V_N^* - \log V_N \mid \mathcal{O}_{N-1}]] = \infty$ — that is, the cumulative worst-case expected log-growth advantage of the robust Kelly strategy diverges. In that case, $S_N / S_N^* \to 0$: the alternative strategy's wealth becomes infinitely smaller than the robust Kelly wealth.
	- The proof combines Theorem 7 with a generalized martingale convergence theorem from non-linear expectation theory (Peng, 2010; Soner, Touzi & Zhang, 2011). The sequence $\log(S_N^* / S_N) - \sum_{s=1}^N [\inf_{\pi_s \in \Xi} \mathbb{E}[\log V_s^* - \log V_s \mid \mathcal{O}_{s-1}]]$ forms a **$G$-martingale** under the non-linear expectation, and Doob's convergence theorem for $G$-martingales guarantees almost-sure convergence.
	- The practical implication is stark: in sequential decision-making under distributional uncertainty, any strategy that systematically differs from the robust Kelly strategy will eventually be dominated by an arbitrarily large factor.

---

# 4. Numerical Example

The paper validates the DRKP with a **horse racing simulation**.

## Setup

- $n = 20$ horses; bets on each horse *placing* (first or second).
- $K = n(n-1)/2 = 190$ possible outcomes (ordered pairs $j < k$).
- Win probabilities $\mu_i$ are drawn proportional to $\exp(z_i)$ with $z_i \sim \mathcal{N}(0, 1/4)$, yielding $\mu_i$ ranging from ~1% to ~20%.
- Nominal outcome distribution: $\pi_{\text{nom},jk} = \mu_j \mu_k \left(\frac{1}{1-\mu_j} + \frac{1}{1-\mu_k}\right)$.
- Returns use **parimutuel betting**: $R_{i,jk} = \frac{n}{1 + \mu_j/\mu_k}$ if $i = j$, $\frac{n}{1 + \mu_k/\mu_j}$ if $i = k$, and $0$ otherwise.

## Two Uncertainty Sets

1. **Box set**: $\Xi_\eta = \{\pi \mid |\pi - \pi_{\text{nom}}| \leq \eta \cdot \pi_{\text{nom}}\}$ with $\eta = 0.26$.
2. **$\ell_2$ ball set**: $\Xi_c = \{\pi \mid \|\pi - \pi_{\text{nom}}\|_2 \leq c\}$ with $c = 0.016$.

Parameters are chosen so both sets produce the same worst-case growth rate of $-2.2\%$ for the standard Kelly bet.

## Results

| Growth Rate | Classic Kelly $\mathbf{b}_K$ | Robust Kelly $\mathbf{b}_{RK}$: box ($\eta = 0.26$) | Robust Kelly $\mathbf{b}_{RK}$: ball ($c = 0.016$) |
|---|---|---|---|
| Nominal $\pi_{\text{nom}}$ | 4.3% | 2.2% | 2.2% |
| Worst-case $\pi_{\text{worst}}$ | **-2.2%** | **0.7%** | **0.4%** |

- Under the nominal distribution, the classic Kelly bet outperforms (4.3% vs. 2.2%) — it is optimized for that specific distribution.
- Under the worst-case distribution, the classic Kelly bet **loses money** ($-2.2\%$), while the robust Kelly bet **maintains positive growth** (0.7% for box, 0.4% for ball).
- The robust Kelly bet produces a more **diverse** (less concentrated) allocation vector — it spreads bets more evenly rather than concentrating on the nominally best horses.

- *Context*:
	- The numerical example illustrates the fundamental trade-off in robust optimization: the robust strategy sacrifices *nominal* performance (2.2% vs. 4.3%) for dramatically improved *worst-case* performance (0.7% vs. $-2.2\%$). In practical investment, where the true distribution is never the nominal one, the robust strategy's worst-case guarantee may be far more valuable than the nominal strategy's best-case optimism.
	- The diversification effect is intuitive: when you hedge against adversarial distributions, you cannot afford to concentrate bets on outcomes that are only favorable under the nominal model. The adversary would simply shift probability away from those outcomes. Spreading bets limits the damage any single distributional shift can inflict.
	- The sensitivity to parameter tuning ($\eta$ and $c$) is a practical limitation. The paper shows (in Figure 2) how growth rates evolve as the uncertainty set size increases: the nominal growth of $\mathbf{b}_{RK}$ decreases while the worst-case growth improves, with the robust and nominal curves crossing at a specific radius.

---

# 5. Discussion: Choosing the Uncertainty Set

The paper addresses the practical challenge of constructing uncertainty sets.

## Chance Constraints

Uncertainty sets can be derived from **chance constraints** — probabilistic bounds like Value-at-Risk (VaR), Conditional Value-at-Risk (CVaR), and moment-based tail bounds. The relation between chance constraints and uncertainty sets (discussed in Duchi, 2018; Alexeenko & Bitar, 2020) allows practitioners to translate risk preferences into geometric constraints on the distribution space.

## Conformal Prediction

As a future direction, the paper proposes using **conformal prediction** to construct uncertainty sets with rigorous finite-sample coverage guarantees. Conformal prediction (Romano et al., 2019; Tibshirani & Foygel, 2019) generates set-valued predictions for black-box predictors with formal error control, which could provide statistically principled uncertainty sets without requiring distributional assumptions.

## Bi-Level Optimization for Uncertainty Set Tuning

The paper proposes automating the tuning of uncertainty set parameters $\theta$ through **bi-level optimization**:

$$
\mathbf{b}^*(\theta) = \arg\max_{\mathbf{b} \in \mathcal{B}} \min_{\pi \in \Xi(\theta)} \mathbb{E}_\pi[\log(\mathbf{r}^\top \mathbf{b})]
$$

$$
\max_\theta \; L(\theta) = \frac{1}{N} \sum_{i=1}^N \log(\mathbf{r}_i^\top \mathbf{b}^*(\theta))
$$

The outer objective $L(\theta)$ evaluates the robust bet $\mathbf{b}^*(\theta)$ on out-of-sample data. Because the inner problem is DCP, recent work on **differentiable convex optimization** (Agrawal et al., 2019) allows computing $\nabla_\theta L(\theta)$ through the solution map $\mathbf{b}^*(\theta)$, enabling gradient-based tuning via frameworks like PyTorch.

- *Context*:
	- The bi-level approach is powerful because it resolves the tension between coverage and tightness *data-adaptively* rather than by manual tuning. The outer objective enforces out-of-sample performance: if the uncertainty set is too tight (the robust bet is too aggressive), $L(\theta)$ will be low on adversarial out-of-sample data; if too wide (too conservative), $L(\theta)$ will be low because the bet is overly hedged even on typical data. The gradient of $L$ with respect to $\theta$ navigates this trade-off automatically.
	- The implementation uses **CVXPYLayers** (Agrawal et al., 2019), a software package that wraps CVXPY problems as differentiable PyTorch layers. This allows the entire pipeline — from uncertainty set parameters through the convex optimization to the out-of-sample loss — to be trained end-to-end via backpropagation.

---

# 6. Supplementary Material: Divergence Functions

The paper catalogues specific $f$-divergences and their Fenchel conjugates, parametrized by a one-parameter family with $\alpha \in \mathbb{R}$:

$$
f_\alpha(t) = \frac{t^\alpha - 1 - \alpha(t - 1)}{\alpha(\alpha - 1)}, \quad f_\alpha^*(s) = \frac{1}{\alpha}\left((1 + (\alpha - 1)s)^{\alpha/(\alpha-1)} - 1\right)
$$

| Divergence | $f(t)$ | $f^*(s)$ | $\alpha$ |
|---|---|---|---|
| KL | $t \log t - t + 1$ | $e^s - 1$ | 1 |
| Reverse KL | $-\log t + t - 1$ | $-\log(1 - s)$ for $s < 1$ | 0 |
| Pearson $\chi^2$ | $\frac{1}{2}(t-1)^2$ | $\frac{1}{2}(s+1)^2 - \frac{1}{2}$ for $s > -1$ | 2 |
| Neyman $\chi^2$ | $\frac{1}{2t}(t-1)^2$ | $1 - \sqrt{1-2s}$ for $s < \frac{1}{2}$ | $-1$ |
| Hellinger | $2(\sqrt{t} - 1)^2$ | $\frac{2s}{2-s}$ for $s < 2$ | $1/2$ |
| Total Variation | $|t - 1|$ | $s$ for $-1 \leq s \leq 1$ | — |

- *Context*:
	- The choice of divergence determines the geometry of the uncertainty set and the nature of the robustness guarantee. KL-divergence penalizes large *multiplicative* deviations in probability ratios, making it sensitive to rare events. Pearson $\chi^2$ penalizes *additive* deviations, making it more sensitive to perturbations of high-probability events. Total variation gives a "hard" constraint: no probability can shift by more than a fixed amount. The Wasserstein set (Theorem 5) is fundamentally different from all $f$-divergences because it considers the *geometry of outcomes*, not just probability mass.

---

# 11. Supplementary Material: CVXPY Implementation

The paper provides concrete CVXPY code for implementing the DRKP, demonstrating that the DCP reformulations are not just theoretically tractable but practically implementable in a few lines.

**Box uncertainty set:**

```python
pi_nom = Parameter(K, nonneg=True)
rho = Parameter(K, nonneg=True)
b = Variable(n)
mu = Variable(K)
wc_growth_rate = min(log(R.T * b) + mu) - pi_nom.T * abs(mu) - rho.T * mu
constraints = [sum(b) == 1, b >= 0]
DRKP = Problem(Maximize(wc_growth_rate), constraints)
DRKP.solve()
```

**Ball uncertainty set:**

```python
pi_nom = Parameter(K, nonneg=True)
c = Parameter((1, 1), nonneg=True)
b = Variable(n)
U = Variable(K)
mu = Variable(K)
log_growth = log(R.T * b)
wc_growth_rate = pi_nom.T * U - c * norm(U - mu, 2)
constraints = [sum(b) == 1, b >= 0, U <= log_growth]
DRKP = Problem(Maximize(wc_growth_rate), constraints)
DRKP.solve()
```

**CvxpyLayer for bi-level learning:**

```python
from cvxpylayers.torch import CvxpyLayer
policy = CvxpyLayer(problem, parameters=[R_cvx, pi_0, rho], variables=[b])
```

- *Context*:
	- The code examples demonstrate the paper's central claim about DCP tractability: the robust Kelly problems are not just solvable in theory — they require only a few lines of CVXPY, with no custom solver code. The `CvxpyLayer` wrapper then enables differentiation through the optimization for uncertainty set learning. The entire computational pipeline from problem specification to gradient-based tuning fits within standard Python scientific computing tools.

---

# MATHEMATICS

The mathematical framework of this paper proceeds in two phases. First, a computational phase (Theorems 1–6) shows that the distributional robust Kelly problem can be reformulated as tractable convex optimization for a wide class of uncertainty sets. Second, a theoretical phase (Theorems 7–8) extends Breiman's classical optimality results to the robust setting using non-linear expectation theory. The derivation chain follows the logical dependency: problem definition → Lagrangian duality for each uncertainty set → non-linear expectation → asymptotic optimality.

### 1. Mean Log Growth Rate and the Kelly Problem

The gambler allocates wealth fractions $\mathbf{b} \in \mathcal{S}_n$ (the $n$-simplex) across $n$ bets. In a finite-outcome setting with $K$ events, $\mathbf{R} \in \mathbb{R}^{n \times K}$ is the payoff matrix with columns $\mathbf{r}_1, \dots, \mathbf{r}_K$, and $\boldsymbol{\pi} = (\pi_1, \dots, \pi_K) \in \mathcal{S}_K$ is the probability vector.

**Mean log growth rate ($G_\pi$):**

$$
G_\pi(\mathbf{b}) = \mathbb{E}_\pi[\log(\mathbf{r}^\top \mathbf{b})] = \boldsymbol{\pi}^\top \log(\mathbf{R}^\top \mathbf{b}) = \sum_{k=1}^K \pi_k \log(\mathbf{r}_k^\top \mathbf{b}) \tag{1}
$$

where $\log$ is applied elementwise. This is the mean drift of the log-wealth random walk.

The **classic Kelly problem** maximizes (1) over $\mathbf{b} \in \mathcal{B} \subseteq \mathcal{S}_n$, where $\mathcal{B}$ is a convex constraint set. Since $\log(\mathbf{r}_k^\top \mathbf{b})$ is concave in $\mathbf{b}$ (composition of concave $\log$ with affine $\mathbf{r}_k^\top \mathbf{b}$) and $\pi_k \geq 0$, $G_\pi$ is concave — a convex optimization problem.

### 2. Worst-Case Log Growth and the DRKP

Given an uncertainty set $\Xi \subseteq \mathcal{S}_K$, the worst-case log growth rate is:

**Worst-case log growth ($G_\Xi$):**

$$
G_\Xi(\mathbf{b}) = \inf_{\boldsymbol{\pi} \in \Xi} G_\pi(\mathbf{b}) = \inf_{\boldsymbol{\pi} \in \Xi} \boldsymbol{\pi}^\top \log(\mathbf{R}^\top \mathbf{b}) \tag{2}
$$

This is concave in $\mathbf{b}$ because it is the infimum of a family of concave functions. The **distributional robust Kelly problem** is:

$$
\max_{\mathbf{b} \in \mathcal{B}} \; G_\Xi(\mathbf{b}) = \max_{\mathbf{b} \in \mathcal{B}} \; \inf_{\boldsymbol{\pi} \in \Xi} \boldsymbol{\pi}^\top \log(\mathbf{R}^\top \mathbf{b}) \tag{3}
$$

### 3. Polyhedron Uncertainty — Lagrangian Duality (Theorem 1)

For $\Xi = \{\boldsymbol{\pi} \in \mathcal{S}_K \mid \mathbf{A}_0 \boldsymbol{\pi} = \mathbf{d}_0, \; \mathbf{A}_1 \boldsymbol{\pi} \leq \mathbf{d}_1\}$, the inner minimization in (2) is the linear program:

$$
\min_{\boldsymbol{\pi}} \; \boldsymbol{\pi}^\top \log(\mathbf{R}^\top \mathbf{b}) \quad \text{s.t.} \quad \boldsymbol{\pi} \in \mathcal{S}_K, \; \mathbf{A}_0 \boldsymbol{\pi} = \mathbf{d}_0, \; \mathbf{A}_1 \boldsymbol{\pi} \leq \mathbf{d}_1 \tag{4}
$$

The Lagrangian, keeping the simplex constraint as an indicator $I_{\mathcal{S}}(\boldsymbol{\pi})$, is:

$$
L(\boldsymbol{\nu}, \boldsymbol{\lambda}, \boldsymbol{\pi}) = \boldsymbol{\pi}^\top \log(\mathbf{R}^\top \mathbf{b}) + \boldsymbol{\nu}^\top(\mathbf{A}_0 \boldsymbol{\pi} - \mathbf{d}_0) + \boldsymbol{\lambda}^\top(\mathbf{A}_1 \boldsymbol{\pi} - \mathbf{d}_1) + I_{\mathcal{S}}(\boldsymbol{\pi}) \tag{5}
$$

where $\boldsymbol{\nu} \in \mathbb{R}^{m_0}$ and $\boldsymbol{\lambda} \in \mathbb{R}^{m_1}_+$ are dual variables. Minimizing over $\boldsymbol{\pi} \in \mathcal{S}_K$:

**Dual function:**

$$
g(\boldsymbol{\nu}, \boldsymbol{\lambda}) = \min\left(\log(\mathbf{R}^\top \mathbf{b}) + \mathbf{A}_0^\top \boldsymbol{\nu} + \mathbf{A}_1^\top \boldsymbol{\lambda}\right) - \mathbf{d}_0^\top \boldsymbol{\nu} - \mathbf{d}_1^\top \boldsymbol{\lambda} \tag{6}
$$

where $\min(\mathbf{v})$ denotes the minimum component of vector $\mathbf{v}$. This arises because the minimum of $\boldsymbol{\pi}^\top \mathbf{v}$ over $\boldsymbol{\pi} \in \mathcal{S}_K$ is $\min(\mathbf{v})$ (the optimum is attained at a vertex of the simplex). Strong duality holds by Slater's condition (the simplex has non-empty relative interior). The DRKP becomes:

$$
\max_{\mathbf{b}, \boldsymbol{\nu}, \boldsymbol{\lambda} \geq 0} \; \min\left(\log(\mathbf{R}^\top \mathbf{b}) + \mathbf{A}_0^\top \boldsymbol{\nu} + \mathbf{A}_1^\top \boldsymbol{\lambda}\right) - \mathbf{d}_0^\top \boldsymbol{\nu} - \mathbf{d}_1^\top \boldsymbol{\lambda} \tag{7}
$$

This is DCP because $\min(\cdot)$ of a concave vector expression is concave.

### 4. Box Uncertainty — Specialization (Theorem 2)

The box set $\Xi = \{\boldsymbol{\pi} \in \mathcal{S}_K \mid |\boldsymbol{\pi} - \boldsymbol{\pi}_{\text{nom}}| \leq \boldsymbol{\rho}\}$ is encoded as $\mathbf{A}_1 \boldsymbol{\pi} \leq \mathbf{d}_1$ with $\mathbf{A}_1 = \begin{pmatrix} \mathbf{I} \\ -\mathbf{I} \end{pmatrix}$, $\mathbf{d}_1 = \begin{pmatrix} \boldsymbol{\pi}_{\text{nom}} + \boldsymbol{\rho} \\ \boldsymbol{\rho} - \boldsymbol{\pi}_{\text{nom}} \end{pmatrix}$. Substituting into (7) and defining $\boldsymbol{\mu} = \boldsymbol{\lambda}^+ - \boldsymbol{\lambda}^-$:

**Box DRKP:**

$$
\max_{\mathbf{b}, \boldsymbol{\mu}} \; \min\left(\log(\mathbf{R}^\top \mathbf{b}) + \boldsymbol{\mu}\right) - \boldsymbol{\pi}_{\text{nom}}^\top \boldsymbol{\mu} - \boldsymbol{\rho}^\top |\boldsymbol{\mu}| \tag{8}
$$

### 5. Ellipsoidal Uncertainty — Hölder Duality (Theorem 3)

For $\Xi = \{\boldsymbol{\pi} \in \mathcal{S}_K \mid \|\mathbf{W}^{-1}(\boldsymbol{\pi} - \boldsymbol{\pi}_{\text{nom}})\|_p \leq 1\}$, define $\mathbf{x} = -\log(\mathbf{R}^\top \mathbf{b})$ and $\mathbf{z} = \mathbf{W}^{-1}(\boldsymbol{\pi} - \boldsymbol{\pi}_{\text{nom}})$. The worst-case growth becomes:

$$
G_\Xi(\mathbf{b}) = -\sup_{\mathbf{z} \in D_{p,\mathbf{W}}} \mathbf{z}^\top \mathbf{W}^\top \mathbf{x} + \boldsymbol{\pi}_{\text{nom}}^\top \mathbf{x} \tag{9}
$$

Applying Lagrangian duality (keeping the $p$-norm constraint, dualizing the simplex-related constraints with multipliers $\mu, \boldsymbol{\lambda}$) and invoking **Hölder's equality**:

$$
\sup_{\|\mathbf{z}\|_p \leq 1} \mathbf{z}^\top \mathbf{W}^\top \mathbf{v} = \|\mathbf{W}^\top \mathbf{v}\|_q, \quad \frac{1}{p} + \frac{1}{q} = 1 \tag{10}
$$

yields the DCP form. Substituting $\mathbf{u} = -\mathbf{x} - \mu\mathbf{1} = \log(\mathbf{R}^\top \mathbf{b}) - \mu\mathbf{1} \leq \log(\mathbf{R}^\top \mathbf{b})$:

**Ellipsoidal DRKP:**

$$
\max_{\mathbf{b}, \mathbf{u}, \mu} \; \boldsymbol{\pi}_{\text{nom}}^\top \mathbf{u} - \|\mathbf{W}^\top(\mathbf{u} - \mu\mathbf{1})\|_q \quad \text{s.t.} \quad \mathbf{u} \leq \log(\mathbf{R}^\top \mathbf{b}), \; \mathbf{b} \in \mathcal{B} \tag{11}
$$

DCP compliance: $\boldsymbol{\pi}_{\text{nom}}^\top \mathbf{u}$ is linear, $-\|\mathbf{W}^\top(\mathbf{u} - \mu\mathbf{1})\|_q$ is concave for $q \geq 1$, and $\mathbf{u} \leq \log(\mathbf{R}^\top \mathbf{b})$ is a concave constraint.

### 6. $f$-Divergence Ball — Fenchel Conjugate (Theorem 4)

For $\Xi = \{\boldsymbol{\pi} \in \mathcal{S}_K \mid D_f(\boldsymbol{\pi} \| \boldsymbol{\pi}_{\text{nom}}) \leq \epsilon\}$, where $D_f(\boldsymbol{\pi}_1 \| \boldsymbol{\pi}_2) = \boldsymbol{\pi}_2^\top f(\boldsymbol{\pi}_1 / \boldsymbol{\pi}_2)$ (elementwise). The Lagrangian with dual variables $\lambda \geq 0$ (for the $\epsilon$-constraint) and $\beta$ (for the simplex) gives, after maximizing over $\boldsymbol{\pi} \geq 0$:

$$
\sup_{\boldsymbol{\pi} \geq 0} L = \sum_{i=1}^K \pi_{\text{nom},i} \sup_{t_i \geq 0}\left(t_i(x_i - \beta) - \lambda f(t_i)\right) + \lambda\epsilon + \beta = \sum_{i=1}^K \pi_{\text{nom},i} \cdot \lambda f^*\!\left(\frac{x_i - \beta}{\lambda}\right) + \lambda\epsilon + \beta \tag{12}
$$

where $f^*(s) = \sup_{t \geq 0}(ts - f(t))$ is the Fenchel conjugate and $x_i = -\log(\mathbf{R}^\top \mathbf{b})_i$.

**$f$-Divergence DRKP (via convex relaxation):**

$$
\max_{\mathbf{b}, \lambda, \beta, \mathbf{w}, \mathbf{z}} \; -\boldsymbol{\pi}_{\text{nom}}^\top \mathbf{w} - \epsilon\lambda - \beta \tag{13}
$$

$$
\text{s.t.} \quad \mathbf{w} \geq \lambda f^*\!\left(\frac{\mathbf{z}}{\lambda}\right), \quad \mathbf{z} \geq -\log(\mathbf{R}^\top \mathbf{b}) - \beta, \quad \lambda \geq 0, \; \mathbf{b} \in \mathcal{B}
$$

The function $\lambda f^*(z/\lambda)$ is the **perspective** of $f^*$, convex and non-decreasing in $z$ when $f^*$ is convex and non-decreasing. The DCP composition rule applies: $-\log(\mathbf{R}^\top \mathbf{b}) - \beta$ is convex, and feeding it into a non-decreasing convex function preserves convexity.

### 7. Wasserstein Ball — Optimal Transport Duality (Theorem 5)

For $\Xi = \{\boldsymbol{\pi} \in \mathcal{S}_K \mid D_c(\boldsymbol{\pi}, \boldsymbol{\pi}_{\text{nom}}) \leq s\}$, the Wasserstein distance $D_c$ is defined as the optimal transport cost:

$$
D_c(\boldsymbol{\pi}, \boldsymbol{\pi}_{\text{nom}}) = \min_{\mathbf{Q} \geq 0} \sum_{i,j} Q_{ij} c_{ij} \quad \text{s.t.} \quad \mathbf{Q}\mathbf{1} = \boldsymbol{\pi}, \; \mathbf{Q}^\top \mathbf{1} = \boldsymbol{\pi}_{\text{nom}} \tag{14}
$$

where $\mathbf{C} \in \mathbb{R}^{K \times K_{\text{nom}}}_+$ is the cost matrix. Using LP duality on the joint optimization over $\boldsymbol{\pi}$ and $\mathbf{Q}$:

**Wasserstein DRKP:**

$$
\max_{\mathbf{b}, \lambda \geq 0} \; \sum_j \pi_{\text{nom},j} \min_i \left(\log(\mathbf{R}^\top \mathbf{b})_i + \lambda c_{ij}\right) - s\lambda \tag{15}
$$

DCP compliance: $\log(\mathbf{R}^\top \mathbf{b})_i + \lambda c_{ij}$ is concave in $(\mathbf{b}, \lambda)$; $\min_i$ preserves concavity; the weighted sum with $\pi_{\text{nom},j} \geq 0$ preserves concavity; $-s\lambda$ is linear.

### 8. Moment Uncertainty — SDP Formulation (Theorem 6)

For $\Xi$ defined by bounds on the mean and covariance estimation error (as in **Section 2, Theorem 6** above), the DRKP is reformulated as:

**Moment uncertainty DRKP (SDP):**

$$
\min_{u_1, u_2, \mathbf{Y}, \mathbf{y}} \; u_1 + u_2 \tag{16}
$$

$$
\text{s.t.} \quad u_1 \geq -\log(\mathbf{r}_i^\top \mathbf{b}) - \mathbf{r}_i^\top \mathbf{Y} \mathbf{r}_i - \mathbf{r}_i^\top \mathbf{y}, \quad \forall i
$$

$$
u_2 \geq (\gamma_2 \boldsymbol{\Sigma}_0 + \boldsymbol{\mu}_0 \boldsymbol{\mu}_0^\top) \bullet \mathbf{Y} + \boldsymbol{\mu}_0^\top \mathbf{y} + \sqrt{\gamma_1}\|\boldsymbol{\Sigma}_0^{1/2}(\mathbf{y} + 2\mathbf{Y}\boldsymbol{\mu}_0)\|_2
$$

$$
\mathbf{Y} \succeq 0, \quad \mathbf{b} \in \mathcal{B}
$$

where $u_1, u_2 \in \mathbb{R}$, $\mathbf{Y} \in \mathbb{R}^{n \times n}$ (symmetric positive semidefinite), $\mathbf{y} \in \mathbb{R}^n$, $\boldsymbol{\mu}_0$ is the estimated mean, $\boldsymbol{\Sigma}_0$ is the estimated covariance, and $\gamma_1, \gamma_2$ are the error bounds. The equivalence follows from Delage & Ye (2010) after verifying Slater's constraint qualification and integrability on the finite event space.

### 9. Sequential Setting and Non-Linear Expectation

For the theoretical results, the paper moves to a sequential setting. The **non-linear expectation operator** is:

$$
\bar{\mathbb{E}}[\cdot] := \inf_{\pi_N \in \Xi} \mathbb{E}_{\pi_N}[\cdot] \tag{17}
$$

This operator is:
- **Superlinear:** $\bar{\mathbb{E}}[X + Y] \geq \bar{\mathbb{E}}[X] + \bar{\mathbb{E}}[Y]$ (because $\inf$ of a sum $\geq$ sum of $\inf$'s).
- **Monotone:** $X \leq Y$ a.s. implies $\bar{\mathbb{E}}[X] \leq \bar{\mathbb{E}}[Y]$.
- **Constant-preserving:** $\bar{\mathbb{E}}[c] = c$ for constants $c$.

A sequence $\{M_N\}$ is a **$G$-supermartingale** under $\bar{\mathbb{E}}$ if $\bar{\mathbb{E}}[M_{N+1} \mid \mathcal{O}_N] \leq M_N$ almost surely for all $N$.

### 10. Asymptotic Maximality (Theorem 7)

The robust Kelly bet $\mathbf{b}_N^*$ maximizes $\bar{\mathbb{E}}[\log(\mathbf{r}^\top \mathbf{b}) \mid \mathcal{O}_{N-1}]$. For any strategy $\Lambda$ with period growth $V_N = \mathbf{r}_N^\top \mathbf{b}_N$ and robust period growth $V_N^* = \mathbf{r}_N^\top \mathbf{b}_N^*$:

**Key inequality:** For any $\mathbf{b}_N$:

$$
\bar{\mathbb{E}}\left[\frac{V_N}{V_N^*} \;\middle|\; \mathcal{O}_{N-1}\right] \leq 1 \tag{18}
$$

**Proof of (18):** By the maximizing property of $\mathbf{b}_N^*$, for any $\epsilon > 0$:

$$
\bar{\mathbb{E}}[\epsilon \log V_N + (1-\epsilon)\log V_N^* \mid \mathcal{O}_{N-1}] \leq \bar{\mathbb{E}}[\log V_N^* \mid \mathcal{O}_{N-1}] \tag{19}
$$

Rewriting the left side and using superlinearity of $\bar{\mathbb{E}}$:

$$
\bar{\mathbb{E}}\left[\frac{1}{\epsilon}\log\!\left(1 + \frac{\epsilon}{1-\epsilon}\frac{V_N}{V_N^*}\right) \;\middle|\; \mathcal{O}_{N-1}\right] \leq \frac{1}{\epsilon}\log\!\left(\frac{1}{1-\epsilon}\right) \tag{20}
$$

Taking $\epsilon \to 0^+$, using the bounded convergence of $V_N/V_N^*$ (since returns are bounded in $[0, R_M]$) and **Fatou's lemma**:

$$
\bar{\mathbb{E}}\left[\frac{V_N}{V_N^*} \;\middle|\; \mathcal{O}_{N-1}\right] = \bar{\mathbb{E}}\left[\lim_{\epsilon \to 0^+} \frac{1}{\epsilon}\log\!\left(1 + \frac{\epsilon}{1-\epsilon}\frac{V_N}{V_N^*}\right) \;\middle|\; \mathcal{O}_{N-1}\right] \leq \lim_{\epsilon \to 0^+} \frac{1}{\epsilon}\log\!\left(\frac{1}{1-\epsilon}\right) = 1 \tag{21}
$$

This makes $S_N / S_N^*$ a **decreasing $G$-supermartingale** under $\bar{\mathbb{E}}$, and:

$$
\inf_{\pi_1, \dots, \pi_N \in \Xi} \mathbb{E}\left[\lim_N \frac{S_N}{S_N^*}\right] \leq 1 \tag{22}
$$

### 11. Magnitude Dominance (Theorem 8)

Define the sequence:

$$
M_N := \log\!\left(\frac{S_N^*}{S_N}\right) - \sum_{s=1}^N \bar{\mathbb{E}}[\log V_s^* - \log V_s \mid \mathcal{O}_{s-1}] \tag{23}
$$

This forms a **$G$-martingale** under $\bar{\mathbb{E}}$, as developed in Peng (2010). By the non-linear Doob martingale convergence theorem, $M_N$ converges almost surely if $\sum_{s=1}^N \bar{\mathbb{E}}[\log V_s^* - \log V_s \mid \mathcal{O}_{s-1}]$ is uniformly bounded for all $N$.

The theorem states:

$$
\left\{\lim_{N \to \infty} \frac{S_N}{S_N^*} = 0\right\} = \left\{\sum_{N=1}^\infty \bar{\mathbb{E}}[\log V_N^* - \log V_N \mid \mathcal{O}_{N-1}] = \infty\right\} \quad \text{a.s.} \tag{24}
$$

The left set is where the robust Kelly strategy dominates $\Lambda$ by magnitude. The right set is the condition for $\Lambda$ and $\Lambda^*$ being **essentially different under uncertainty**: their cumulative worst-case expected log-growth advantage diverges. Thus, the robust Kelly strategy dominates any essentially different strategy by magnitude almost surely.

---

### **1. Problem:**

The classic Kelly criterion provides the theoretically optimal betting strategy for maximizing long-run wealth growth, but requires knowledge of the true probability distribution — a condition never satisfied in practice. Estimation errors, distribution shifts, and noisy moment estimates lead to the **optimizer's curse**, where decisions optimized under an estimated distribution perform poorly out-of-sample. The paper fills the gap between the Kelly criterion's theoretical elegance and its practical fragility by introducing a distributional robust formulation that hedges against distributional uncertainty while preserving computational tractability and asymptotic optimality.

### **2. Setup:**

The paper assumes a repeated gambling setting with $n$ bets, $K$ finite outcomes, and IID returns (though distributions may vary within the uncertainty set in the sequential theory). Returns are bounded: $r_{N,i} \in [0, R_M]$. The bet allocation $\mathbf{b}$ lies in a convex subset $\mathcal{B}$ of the probability simplex $\mathcal{S}_n$. The uncertainty set $\Xi$ constrains the probability vector $\boldsymbol{\pi} \in \mathcal{S}_K$. The computational framework is **Disciplined Convex Programming** via CVXPY, targeting interior-point solvers. The theoretical framework extends to sequential decision-making with time-varying distributions using **non-linear expectation theory**.

### **3. Key Idea:**

The core contribution is replacing the classic Kelly objective $\max_\mathbf{b} \mathbb{E}_\pi[\log(\mathbf{r}^\top \mathbf{b})]$ with the distributional robust objective $\max_\mathbf{b} \inf_{\pi \in \Xi} \mathbb{E}_\pi[\log(\mathbf{r}^\top \mathbf{b})]$. The paper demonstrates that this robust formulation is computationally tractable (DCP reformulations exist for polyhedral, box, ellipsoidal, $f$-divergence, Wasserstein, and moment-based uncertainty sets) and theoretically optimal (the robust bet preserves Breiman's asymptotic maximality and magnitude dominance properties under the worst-case distribution sequence).

### **4. Assumptions:**

**Explicit:**
- Returns in different rounds are IID (standard Kelly assumption; relaxed in sequential theory where $\pi_N$ varies within $\Xi$).
- Returns are bounded: $r_{N,i} \in [0, R_M]$.
- Finite number of outcomes $K$ (for the computational tractability results).
- Additional constraints $\mathcal{B}$ on the bet allocation must be convex.
- Slater's constraint qualification holds (for strong duality in the Lagrangian reformulations).

**Implicit:**
- The uncertainty set $\Xi$ is "high quality" — neither so wide that the robust bet degenerates to the cash position, nor so narrow that it fails to capture the true distribution. The paper acknowledges this but does not provide automatic methods beyond the bi-level optimization proposal.
- The finite-outcome assumption limits direct applicability to continuous return distributions, which must be discretized.
- The sequential theory assumes a *fixed* uncertainty set $\Xi$ across all periods — the set itself does not adapt or expand.

### **5. Limitation:**

- **Uncertainty set construction** is the fundamental bottleneck. The theoretical guarantees hold for *any* $\Xi$ containing the true distribution, but practical performance depends critically on the quality of $\Xi$. The paper's bi-level optimization approach is promising but presented only as a sketch.
- **Coverage-tightness tension:** A wide $\Xi$ provides more robust guarantees but produces overly conservative (low-growth) bets. A tight $\Xi$ produces aggressive bets that may fail if the true distribution falls outside. No principled method exists for resolving this trade-off without out-of-sample data.
- **Computational overhead:** While DCP tractable, the robust formulations are significantly more expensive than the classic Kelly problem (which is a simple concave maximization). The SDP formulation (Theorem 6) is particularly expensive. This may limit applicability in high-frequency or real-time settings.
- **Static uncertainty set:** The theory assumes a fixed $\Xi$. In practice, the appropriate uncertainty set may change over time (e.g., during regime changes in financial markets), and the theory does not address adaptive uncertainty sets.
- **Finite-time properties unexplored:** The theoretical results are asymptotic. Finite-time properties — drawdown probabilities, variance of the growth rate, convergence speed — are acknowledged as open questions.

### **6. Relevance & Open Questions:**

This paper is directly relevant to any research program that uses the Kelly criterion as a foundation for sequential decision-making under uncertainty — particularly in reinforcement learning for portfolio optimization. The distributional robust framework provides the missing bridge between the Kelly criterion's theoretical optimality and practical deployment conditions.

Key open questions:
- **Finite-time drawdown analysis:** Can distributional robustness provide finite-time guarantees on maximum drawdown or tail risk, complementing the asymptotic optimality results?
- **Conformal prediction integration:** Can conformal prediction provide finite-sample coverage guarantees for uncertainty sets in a computationally tractable form compatible with the DCP framework?
- **Connection to distributional RL:** The Wasserstein uncertainty set (Theorem 5) and the Wasserstein contraction in distributional RL ([[PNOTES - A Distributional Perspective on Reinforcement Learning]]) use the same mathematical distance. Can the DRKP be embedded as the "policy improvement" step in a distributional RL agent that learns the return distribution online?
- **Non-stationary uncertainty sets:** How should $\Xi$ adapt when the data-generating process is non-stationary? Can the bi-level optimization be run in an online fashion?
- **Continuous action spaces:** The paper treats the bet allocation over a finite set of discrete bets. Extension to continuous portfolio weights (as in Merton-style continuous-time models) would require different convex formulations.

---

### Integration:

* **Problem:** The DRKP addresses the same fundamental tension identified in practical Kelly strategies for portfolio RL: the gap between the theoretical optimality of log-growth maximization and the brittleness of that optimality under distribution misspecification. For any RL agent using log-wealth as a reward signal (as in [[PNOTES - Deep Reinforcement Learning for Optimal Asset Allocation Using DDPG with TiDE]]), the optimizer's curse is not hypothetical — it is the default failure mode when the environment's return distribution is estimated from finite, non-stationary data. The DRKP provides a principled mechanism for converting distributional uncertainty into a tractable, conservative correction to the Kelly objective, and the asymptotic dominance results (Theorems 7–8) ensure that this conservatism does not sacrifice long-run optimality.

* **Limitation:** The most significant limitation for practical integration is the static nature of the uncertainty set. Financial markets and real-world environments exhibit regime changes, structural breaks, and evolving correlations — the uncertainty set must adapt accordingly. The bi-level optimization proposal is a step in this direction, but it assumes access to out-of-sample data and a differentiable solution map, which may not be available in online RL settings. Additionally, the finite-outcome assumption and the computational cost of SDP formulations limit direct deployment in high-dimensional, continuous-action portfolio optimization. The finite-time gap — no guarantees on drawdown or growth variance before the asymptotic regime — is a critical concern for risk-sensitive applications where the investor cannot wait for the long run.
