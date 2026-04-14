[[Risk-Constrained Kelly Gambling.pdf]]

# **Abstract:**

The paper addresses the well-known drawdown vulnerability of the classic **Kelly criterion**: while Kelly-optimal bets maximize the long-term growth rate of wealth, they expose the investor to severe path-dependent drops in capital. The authors develop a **bound on drawdown probability** that, when substituted for the true (intractable) drawdown risk constraint, yields a convex optimization problem called the **Risk-Constrained Kelly (RCK)** gambling problem. The RCK problem is a *restriction* of the ideal drawdown-constrained problem — its feasible set is a subset of the ideal feasible set — meaning any RCK-feasible bet is guaranteed to satisfy the drawdown constraint. Numerical experiments show the bound is reasonably tight (typically within 30% of the true drawdown risk). The method is parametrized by a single **risk-aversion parameter** $\lambda = \log \beta / \log \alpha$ that controls the growth-risk trade-off. Simulations demonstrate that RCK bets outperform heuristic **fractional Kelly** bets at the same drawdown risk level. Finally, the paper derives a **quadratic approximation (QRCK)** that connects the RCK framework to classical **Markowitz mean-variance portfolio optimization**.

- *Context*:
	- This paper sits at the intersection of information-theoretic growth optimization and convex portfolio theory. Where [[PNOTES - Optimal Betting Under Parameter Uncertainty - Improving the Kelly Criterion]] addresses Kelly's vulnerability to *parameter uncertainty* (unknown $p$) via bet shrinkage, and [[PNOTES - Modified Kelly criteria]] addresses it via Bayesian estimation, the present paper addresses a different vulnerability: *path-dependent drawdown risk* even when the return distribution is perfectly known. The two failure modes are orthogonal — a bettor with perfect knowledge of return distributions can still suffer extreme drawdowns under pure Kelly, and a bettor with uncertain estimates can still overbbet. A complete risk management framework would need to address both simultaneously.
	- The key methodological innovation is the transformation of an intractable probabilistic constraint ($\Pr(W_{\min} < \alpha) < \beta$) into a tractable convex constraint ($E[(r^\top b)^{-\lambda}] \leq 1$) via a tilted probability measure argument. This is a classic motif in robust optimization: replace a hard problem with a solvable restriction that guarantees feasibility, accepting a potentially smaller feasible set in exchange for computational tractability.

---

# 1. Introduction

The paper opens by situating the **Kelly criterion** (Kelly, 1956) in the broader landscape of growth-optimal investment. The key tension: Kelly bets maximize the asymptotic growth rate of wealth, but they do so without regard for the *path* of wealth — the investor may experience severe drawdowns before the long-run growth dominates.

- **Fractional Kelly** betting — placing only a fraction $f \in (0, 1)$ of the Kelly-optimal bet — is the standard *ad hoc* remedy. It reduces drawdown risk at the cost of reduced growth rate, but the fraction $f$ is chosen heuristically rather than derived from a principled risk constraint.
	- The same idea appears in the finance literature as growth-optimum portfolio scaling (Browne, 2000).
- **Markowitz mean-variance optimization** (Markowitz, 1952) trades off expected return against return variance — a related but distinct pair of objectives. The connection between Markowitz and Kelly is explored later in §6.
- The paper's contribution: a *direct* formulation of the growth-rate-versus-drawdown-risk trade-off as a convex optimization problem, avoiding the need for heuristic scaling.
	- Two use modes: (1) given drawdown specifications $\alpha, \beta$, solve for a *guaranteed-safe* bet; (2) treat $\lambda$ as a risk-aversion dial and sweep the growth-risk frontier.

- *Context*:
	- The distinction between fractional Kelly (a heuristic) and RCK (a principled constraint) is analogous to the distinction between ad hoc regularization and constrained optimization in machine learning. Fractional Kelly uniformly scales *all* risky bets toward cash, regardless of their individual risk contributions. RCK, by contrast, solves for the optimal allocation subject to a global risk budget — it can allocate risk unevenly across bets, concentrating exposure on bets with favorable risk-adjusted profiles. This is why RCK dominates fractional Kelly in the numerical experiments.

---

# 2. Kelly Gambling

## 2.1 Setup and Notation

The formalization uses a general $n$-bet framework:

- **Bet vector** $\mathbf{b} \in \mathbb{R}^n$: the fraction of total wealth allocated to each of $n$ bets, with $\mathbf{b} \geq 0$ and $\mathbf{1}^\top \mathbf{b} = 1$.
- **Return vector** $\mathbf{r} \in \mathbb{R}^n_+$: the random nonnegative payoff for each bet.
- The $n$-th bet is always **cash**: $r_n = 1$ almost surely. The cash allocation $b_n$ represents the fraction of wealth not wagered.
- The wealth after one round changes by the factor $\mathbf{r}^\top \mathbf{b}$.
- **Ruin** ($w_t = 0$) is impossible if $b_n > 0$, since $\mathbf{r}^\top \mathbf{b} \geq b_n > 0$ almost surely.
- All bets have finite expected return: $E[r_i] < \infty$ for $i = 1, \ldots, n$.

Special cases covered by this framework:

- **Two outcomes**: $n = 2$, binary win/loss with payoff $P > 1$ and probability $\pi$.
- **Mutually exclusive outcomes**: horse-race-style bets on $n - 1$ exclusive outcomes.
- **General finite outcomes**: $\mathbf{r}$ takes $K$ discrete values with probabilities $\pi_1, \ldots, \pi_K$.
- **General returns**: $\mathbf{r}$ drawn from an arbitrary infinite distribution (e.g., log-normal returns in financial markets).

- *Context*:
	- The cash asset $r_n = 1$ plays a structural role beyond risk-free investment: it ensures the return $\mathbf{r}^\top \mathbf{b}$ is bounded away from zero whenever $b_n > 0$, which prevents ruin and guarantees the log-wealth random walk is well-defined. Without this cash anchor, log-wealth can hit $-\infty$ in finite time, making the growth rate objective meaningless.

## 2.2 Wealth Growth

The gamble is repeated at IID epochs $t = 1, 2, \ldots$ with initial wealth $w_1 = 1$. Wealth at time $t$:

$$
w_t = (\mathbf{r}_1^\top \mathbf{b}) \cdots (\mathbf{r}_{t-1}^\top \mathbf{b})
$$

Log-wealth $v_t = \log w_t$ is a **random walk** with drift $E[\log(\mathbf{r}^\top \mathbf{b})]$:

$$
v_t = \log(\mathbf{r}_1^\top \mathbf{b}) + \cdots + \log(\mathbf{r}_{t-1}^\top \mathbf{b})
$$

The quantity $E[\log(\mathbf{r}^\top \mathbf{b})]$ is the **expected growth rate** — the average per-period increase in log-wealth.

- *Context*:
	- The log transformation is not merely a mathematical convenience — it converts the *multiplicative* wealth process into an *additive* random walk, unlocking the entire machinery of random walk theory (drift, variance, stopping times, Wald's identity). The growth rate $E[\log(\mathbf{r}^\top \mathbf{b})]$ is the natural measure of long-run performance in multiplicative environments because the law of large numbers applies to sums (additive), not products (multiplicative). Kelly's insight was to recognize that maximizing this drift maximizes the rate at which wealth compounds.

## 2.3 Kelly Gambling Problem

The Kelly gambling problem maximizes the growth rate:

$$
\begin{aligned}
&\text{maximize} \quad E[\log(\mathbf{r}^\top \mathbf{b})] \\
&\text{subject to} \quad \mathbf{1}^\top \mathbf{b} = 1, \quad \mathbf{b} \geq 0
\end{aligned} \tag{1}
$$

- This is a **convex optimization problem** (concave objective, convex constraints).
- The trivial bet $\mathbf{b} = \mathbf{e}_n$ (all cash) is always feasible, achieving growth rate zero.
- $\mathbf{b} = \mathbf{e}_n$ is Kelly optimal if and only if $E[r_i] \leq 1$ for all $i = 1, \ldots, n-1$ — that is, no bet has positive expected excess return.
- If any bet has $E[r_i] > 1$, the optimal growth rate is strictly positive.

- *Context*:
	- The condition $E[r_i] \leq 1$ is the no-edge condition. When all bets are losers in expectation, the best strategy is to hold cash. When at least one bet is a winner, the Kelly solution exploits it. This is the optimality of the "don't bet unless you have an edge" principle, formalized within the convex optimization framework.

## 2.4 Computing Kelly Optimal Bets

The paper describes three computational approaches, matched to the complexity of the return distribution:

- **Two outcomes**: closed-form solution $b_1^* = (\pi P - 1)/(P - 1)$ when $\pi P > 1$.
- **General finite outcomes**: standard convex optimization via CVXPY/ECOS.
- **General returns**: **projected stochastic gradient method with averaging**. The gradient $\nabla_{\mathbf{b}} E[\log(\mathbf{r}^\top \mathbf{b})] = E[\mathbf{r} / (\mathbf{r}^\top \mathbf{b})]$ is estimated by the stochastic gradient $\mathbf{r}^{(k)} / (\mathbf{r}^{(k)\top} \mathbf{b})$ from IID samples $\mathbf{r}^{(k)}$. The iterates are projected onto $\Delta_\varepsilon = \{\mathbf{b} \mid \mathbf{1}^\top \mathbf{b} = 1, \mathbf{b} \geq 0, b_n \geq \varepsilon\}$. Step sizes $t_k \to 0$ with $\sum t_k = \infty$ guarantee convergence of the weighted running average to the optimum.

- *Context*:
	- The stochastic gradient method is essential for the general returns case (e.g., log-normal mixtures in §7.2) where the expectation $E[\log(\mathbf{r}^\top \mathbf{b})]$ has no closed form. The constraint $b_n \geq \varepsilon$ prevents the iterates from approaching the boundary where the gradient blows up ($\mathbf{r}^\top \mathbf{b} \to 0$). In practice, $\varepsilon$ is chosen small and validated post-hoc by checking $(b^{(k)})_n > \varepsilon$. Batching — averaging the stochastic gradient over multiple samples per iteration — does not affect theoretical convergence but improves practical performance by reducing gradient variance.

---

# 3. Drawdown

## 3.1 Drawdown Definition

The **minimum wealth** is defined as the infimum of the wealth trajectory:

$$
W_{\min} = \inf_{t = 1, 2, \ldots} w_t
$$

- The **drawdown** is $1 - W_{\min}$. A drawdown of 0.3 means wealth dropped 30% from its initial value before eventually recovering.
- With positive growth rate ($E[\log(\mathbf{r}^\top \mathbf{b})] > 0$), $W_{\min} \in (0, 1]$ — wealth eventually recovers, but may first drop to a small fraction of the initial level.
- The **drawdown risk** is $\Pr(W_{\min} < \alpha)$ for a given threshold $\alpha \in (0, 1)$.
	- Example: drawdown risk of 0.1 at $\alpha = 0.7$ means a 10% probability of experiencing more than 30% drawdown.

- *Context*:
	- Drawdown risk is a *path-dependent* measure — it depends on the entire trajectory of $\{w_t\}$, not just the terminal value. This is fundamentally different from variance or CVaR of single-period returns. A strategy can have low per-period variance but high drawdown risk if the per-period losses are correlated across time (which they are not in this IID model, but the cumulative effect of even IID losses can produce deep drawdowns). The $W_{\min}$ statistic is particularly psychologically salient: it represents the worst point the investor must endure before the strategy pays off.

## 3.2 Fractional Kelly Gambling

The ad hoc drawdown mitigation: compute Kelly-optimal $\mathbf{b}^*$, then use:

$$
\mathbf{b} = f \mathbf{b}^* + (1 - f) \mathbf{e}_n \tag{4}
$$

where $f \in (0, 1)$ is the fraction. This scales all risky bets by $f$ and increases the cash position by $1 - f$.

- *Context*:
	- Fractional Kelly is a *linear interpolation* between the Kelly bet and all-cash. It reduces drawdown risk by moving toward cash, but it does so uniformly across all bets. If the Kelly solution concentrates heavily on one bet (because that bet has the best risk-adjusted edge), fractional Kelly preserves this concentration — it does not rebalance across bets. RCK, by contrast, can redistribute bets to achieve the same drawdown reduction more efficiently.

## 3.3 Kelly Gambling with Drawdown Risk

The *ideal* constrained problem adds the drawdown risk constraint directly:

$$
\begin{aligned}
&\text{maximize} \quad E[\log(\mathbf{r}^\top \mathbf{b})] \\
&\text{subject to} \quad \mathbf{1}^\top \mathbf{b} = 1, \quad \mathbf{b} \geq 0, \\
&\phantom{\text{subject to} \quad} \Pr(W_{\min} < \alpha) < \beta
\end{aligned} \tag{5}
$$

This problem is, as far as the authors know, **intractable** in general: the drawdown probability $\Pr(W_{\min} < \alpha)$ has no closed-form expression in terms of $\mathbf{b}$ for general return distributions.

- *Context*:
	- The intractability stems from the path-dependence of $W_{\min}$. To compute $\Pr(W_{\min} < \alpha)$ exactly, one would need to characterize the distribution of the running minimum of a random walk — a problem that is tractable only for special increment distributions (e.g., Gaussian). For general discrete or continuous return distributions, Monte Carlo simulation is the only reliable method, but embedding Monte Carlo inside an optimizer creates a stochastic bilevel optimization problem that is extremely difficult to solve. The authors' strategy is to sidestep this difficulty entirely by replacing the constraint with a *sufficient condition*.

---

# 4. Drawdown Risk Bound

This section contains the paper's central analytical contribution: a convex condition that *implies* the drawdown risk constraint.

## The Bound

For any $\lambda > 0$ and bet $\mathbf{b}$, and for any $\alpha, \beta \in (0, 1)$ satisfying $\lambda = \log \beta / \log \alpha$:

$$
E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] \leq 1 \implies \Pr(W_{\min} < \alpha) < \beta \tag{6}
$$

- *Context*:
	- This is the key innovation. The left-hand side is a *one-step* condition on the bet vector $\mathbf{b}$ and the return distribution — it involves no path-dependence, no trajectory simulation, and no stopping times. Yet it guarantees a bound on the *infinite-horizon* drawdown probability. The magic comes from the tilted probability measure argument in Lemma 5 (Appendix A), which converts a path-dependent stopping-time identity into a single-step moment condition.

## Proof Sketch

The proof leverages a stopping-time argument from random walk theory:

1. Define the stopping time $\tau = \inf\{t \geq 1 \mid w_t < \alpha\}$, so $\tau < \infty$ iff $W_{\min} < \alpha$.
2. From **Lemma 5** (a modified Wald identity using a **tilted probability measure** $d\mu_\lambda$):

$$
E[\exp(-\lambda \log w_\tau - \tau \log E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}]) \mid \tau < \infty] \cdot \Pr(W_{\min} < \alpha) \leq 1
$$

3. When $E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] \leq 1$, the term $-\tau \log E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] \geq 0$, so it can be dropped to get:

$$
E[\exp(-\lambda \log w_\tau) \mid \tau < \infty] \cdot \Pr(W_{\min} < \alpha) \leq 1
$$

4. Since $w_\tau < \alpha$ when $\tau < \infty$, we have $\exp(-\lambda \log w_\tau) > \exp(-\lambda \log \alpha) = \alpha^{-\lambda} = 1/\beta$.
5. Therefore $\Pr(W_{\min} < \alpha) < \beta$.

- *Context*:
	- The tilted probability measure technique (also called exponential change of measure or Esscher transform) is a standard tool in applied probability for converting moment generating function conditions into tail probability bounds. The idea: define a new probability measure under which the random walk has a different drift, then use the Radon-Nikodym derivative to relate probabilities under the original and tilted measures. The condition $E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] \leq 1$ ensures that the moment generating function of the log-return at parameter $\lambda$ is non-positive, which via Lemma 5 translates into a bound on the probability that the random walk ever crosses a threshold. The proof is tight enough that the bound is typically within 30% of the true drawdown risk.

---

# 5. Risk-Constrained Kelly Gambling

## 5.1 The RCK Problem

Replacing the intractable constraint in (5) with the convex bound (6) yields the **Risk-Constrained Kelly (RCK)** problem:

$$
\begin{aligned}
&\text{maximize} \quad E[\log(\mathbf{r}^\top \mathbf{b})] \\
&\text{subject to} \quad \mathbf{1}^\top \mathbf{b} = 1, \quad \mathbf{b} \geq 0, \\
&\phantom{\text{subject to} \quad} E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] \leq 1
\end{aligned} \tag{7}
$$

- This is a **restriction** of problem (5): RCK-feasible $\implies$ drawdown-constrained-feasible.
- **Convexity**: the objective $E[\log(\mathbf{r}^\top \mathbf{b})]$ is concave; the function $(\mathbf{r}^\top \mathbf{b})^{-\lambda}$ is convex in $\mathbf{b}$ for $\mathbf{r}^\top \mathbf{b} > 0$ (composition of a convex decreasing power function with a linear function), so $E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}]$ is convex.
- The trivial bet $\mathbf{b} = \mathbf{e}_n$ is always feasible (it achieves $(\mathbf{r}^\top \mathbf{b})^{-\lambda} = 1$).
- As $\lambda \to 0$, the constraint becomes vacuous and RCK reduces to unconstrained Kelly.
- As $\lambda \to \infty$, the constraint forces $\mathbf{r}^\top \mathbf{b} \geq 1$ almost surely — only risk-free bets are feasible.

- *Context*:
	- The fact that RCK is a *restriction* (not a relaxation) is crucial for practical safety guarantees. A relaxation would provide a lower bound on the optimal growth rate but could violate the drawdown constraint. A restriction provides an *upper* bound on the achievable growth rate (the true constrained optimum may be higher) but guarantees that the drawdown constraint is satisfied. For risk management, this conservative bias is acceptable — the investor would rather have a slightly suboptimal growth rate than a violated risk constraint.

## 5.2 Risk Aversion Parameter

The parameter $\lambda = \log \beta / \log \alpha$ controls the entire family of drawdown constraints. For a fixed $\lambda$, the single constraint $E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] \leq 1$ implies:

$$
\Pr(W_{\min} < \alpha) < \alpha^\lambda \quad \text{for all } \alpha \in (0, 1) \tag{8}
$$

This bounds the *entire CDF* of $W_{\min}$: the CDF stays below the curve $\alpha \mapsto \alpha^\lambda$.

- Example: $\alpha = 0.7$, $\beta = 0.1$ gives $\lambda = 6.46$. This also implies $\Pr(W_{\min} < 0.5) < 0.5^{6.46} = 0.011$.
- $\lambda$ can be used as a **risk-aversion dial**, analogous to the risk-aversion parameter in Markowitz optimization: sweep $\lambda$ to trace out the growth-risk trade-off frontier.

- *Context*:
	- The CDF interpretation is powerful: a single parameter $\lambda$ controls an infinite family of drawdown constraints simultaneously. Increasing $\lambda$ tightens *all* thresholds — it doesn't just reduce the probability of one specific drawdown level, it shifts the entire distribution of $W_{\min}$ upward. This makes $\lambda$ a natural global risk control, unlike specifying individual $(\alpha, \beta)$ pairs which would require solving separate problems for each threshold.

## 5.3 Light and Heavy Risk Aversion Regimes

**Heavy risk aversion** ($\lambda \to \infty$): the constraint reduces to $\mathbf{r}^\top \mathbf{b} \geq 1$ almost surely — only deterministic (risk-free) bets are allowed.

**Light risk aversion** ($\lambda \to 0$): expanding via the exponential disutility:

$$
\frac{1}{\lambda} \log E[\exp(-\lambda \log(\mathbf{r}^\top \mathbf{b}))] \approx -E[\log(\mathbf{r}^\top \mathbf{b})] + \frac{\lambda}{2} \text{Var}[\log(\mathbf{r}^\top \mathbf{b})]
$$

So the constraint becomes:

$$
\frac{\lambda}{2} \text{Var}[\log(\mathbf{r}^\top \mathbf{b})] \leq E[\log(\mathbf{r}^\top \mathbf{b})]
$$

- *Context*:
	- In the light-aversion regime, the RCK constraint reduces to a bound on the ratio of variance to mean growth in log-wealth — essentially a **Sharpe-ratio-like condition** on log-returns. This connects the drawdown bound to the mean-variance framework: small $\lambda$ approximates a constraint on the coefficient of variation of the log-return, while large $\lambda$ approaches an almost-sure no-loss condition. The full RCK constraint smoothly interpolates between these extremes.

## 5.4 Computing RCK Bets

**Two outcomes**: the RCK problem reduces to a univariate problem solvable by bisection, since the risk constraint is monotone in $b_1$.

**Finite outcomes**: the log-sum-exp reformulation:

$$
\begin{aligned}
&\text{maximize} \quad \sum_{i=1}^K \pi_i \log(\mathbf{r}_i^\top \mathbf{b}) \\
&\text{subject to} \quad \mathbf{1}^\top \mathbf{b} = 1, \quad \mathbf{b} \geq 0, \\
&\phantom{\text{subject to} \quad} \log\left(\sum_{i=1}^K \exp(\log \pi_i - \lambda \log(\mathbf{r}_i^\top \mathbf{b}))\right) \leq 0
\end{aligned} \tag{11}
$$

This is readily handled by DCP-compliant solvers (CVXPY/ECOS).

**General returns**: a **primal-dual stochastic gradient method** applied to the Lagrangian:

$$
L(\mathbf{b}, \kappa) = -E[\log(\mathbf{r}^\top \mathbf{b})] + \kappa(E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] - 1) \tag{12}
$$

The primal update (on $\mathbf{b}$) and dual update (on $\kappa \geq 0$) use stochastic gradients from IID return samples, with projection onto $\Delta_\varepsilon$ and $[0, M]$ respectively. Convergence is guaranteed under standard step-size conditions.

- *Context*:
	- The primal-dual stochastic gradient method is the workhorse for the infinite-distribution case. The dual variable $\kappa$ acts as an adaptive penalty weight on the risk constraint: when the current bet violates the constraint, $\kappa$ increases (tightening the penalty); when the constraint is slack, $\kappa$ decreases (relaxing it). This self-tuning mechanism avoids the need to hand-select a penalty parameter, which would be required in a penalty-method approach.

## 5.5 RCK Optimality Conditions

A pair $(\mathbf{b}^*, \kappa^*)$ is a solution of the RCK problem if and only if it satisfies:

$$
\begin{aligned}
&\mathbf{1}^\top \mathbf{b}^* = 1, \quad \mathbf{b}^* \geq 0, \quad E[(\mathbf{r}^\top \mathbf{b}^*)^{-\lambda}] \leq 1 \\
&\kappa^* \geq 0, \quad \kappa^*(E[(\mathbf{r}^\top \mathbf{b}^*)^{-\lambda}] - 1) = 0 \\
&E\left[\frac{r_i}{\mathbf{r}^\top \mathbf{b}^*}\right] + \kappa^* \lambda E\left[\frac{r_i}{(\mathbf{r}^\top \mathbf{b}^*)^{\lambda+1}}\right] \begin{cases} \leq 1 + \kappa^* \lambda & \text{if } b_i^* = 0 \\ = 1 + \kappa^* \lambda & \text{if } b_i^* > 0 \end{cases}
\end{aligned} \tag{13}
$$

These KKT conditions can be verified independently via Monte Carlo after a solution is computed. **See MATHEMATICS §5** for the derivation.

- *Context*:
	- The optimality conditions have a clean economic interpretation. The term $E[r_i / (\mathbf{r}^\top \mathbf{b}^*)]$ is the marginal contribution of bet $i$ to the growth rate. The term $\kappa^* \lambda E[r_i / (\mathbf{r}^\top \mathbf{b}^*)^{\lambda+1}]$ is the marginal contribution to the risk constraint, weighted by the dual price $\kappa^*$. At optimality, for every active bet ($b_i^* > 0$), the sum of marginal growth and risk-weighted marginal risk equals the same constant $1 + \kappa^* \lambda$. For inactive bets ($b_i^* = 0$), the marginal benefit is insufficient to justify activation. This is the standard complementary slackness structure of constrained optimization.

---

# 6. Quadratic Approximation

## 6.1 Taylor Expansion

Defining the **excess return** $\boldsymbol{\rho} = \mathbf{r} - \mathbf{1}$ so that $\mathbf{r}^\top \mathbf{b} - 1 = \boldsymbol{\rho}^\top \mathbf{b}$, and assuming $\boldsymbol{\rho}^\top \mathbf{b} \approx 0$ (i.e., $\mathbf{r}^\top \mathbf{b} \approx 1$), the Taylor approximations are:

$$
\log(\mathbf{r}^\top \mathbf{b}) \approx \boldsymbol{\rho}^\top \mathbf{b} - \frac{1}{2}(\boldsymbol{\rho}^\top \mathbf{b})^2
$$

$$
(\mathbf{r}^\top \mathbf{b})^{-\lambda} \approx 1 - \lambda \boldsymbol{\rho}^\top \mathbf{b} + \frac{\lambda(\lambda+1)}{2}(\boldsymbol{\rho}^\top \mathbf{b})^2
$$

## 6.2 QRCK Formulation

Substituting the Taylor approximations into the RCK problem (7) yields the **Quadratic Risk-Constrained Kelly (QRCK)** problem:

$$
\begin{aligned}
&\text{maximize} \quad E[\boldsymbol{\rho}^\top \mathbf{b}] - \frac{1}{2} E[(\boldsymbol{\rho}^\top \mathbf{b})^2] \\
&\text{subject to} \quad \mathbf{1}^\top \mathbf{b} = 1, \quad \mathbf{b} \geq 0, \\
&\phantom{\text{subject to} \quad} -\lambda E[\boldsymbol{\rho}^\top \mathbf{b}] + \frac{\lambda(\lambda+1)}{2} E[(\boldsymbol{\rho}^\top \mathbf{b})^2] \leq 0
\end{aligned} \tag{14}
$$

This is a **convex quadratic program (QP)** that can be solved efficiently. The QRCK solution provides a good initial point for the primal-dual stochastic gradient method applied to the full RCK problem.

- *Context*:
	- The QRCK approximation is valid when returns are close to 1 — i.e., when bets are "small" relative to wealth. This is precisely the regime of financial portfolio management, where daily returns are typically within a few percent of 1. For gambling with extreme payoffs (e.g., horse racing with $r_i = 0.2$ or $r_i = 2$), the approximation may be poor, as confirmed by the numerical experiments where QRCK underperforms RCK.

## 6.3 Connection to Markowitz

Using $\boldsymbol{\mu} = E[\boldsymbol{\rho}]$ (mean excess return) and $\mathbf{S} = E[\boldsymbol{\rho} \boldsymbol{\rho}^\top] = \boldsymbol{\Sigma} + \boldsymbol{\mu} \boldsymbol{\mu}^\top$ (raw second moment), and defining a **Markowitz portfolio** as a solution of:

$$
\begin{aligned}
&\text{maximize} \quad \boldsymbol{\mu}^\top \mathbf{b} - \frac{\gamma}{2} \mathbf{b}^\top \boldsymbol{\Sigma} \mathbf{b} \\
&\text{subject to} \quad \mathbf{1}^\top \mathbf{b} = 1, \quad \mathbf{b} \geq 0
\end{aligned} \tag{15}
$$

the paper proves that **every QRCK solution is a Markowitz portfolio** (for some $\gamma > 0$), provided there is **no arbitrage** (i.e., any long-only portfolio with positive expected excess return also has positive return variance).

The proof proceeds through strong Lagrange duality:

1. Dualize the risk constraint in QRCK to get a dual variable $\nu \geq 0$.
2. Substitute $\mathbf{S} = \boldsymbol{\mu}\boldsymbol{\mu}^\top + \boldsymbol{\Sigma}$ and rewrite the objective.
3. Show that the QRCK solution solves a modified Markowitz problem (17) with a specific $\gamma = \eta / (1 - \eta \boldsymbol{\mu}^\top \mathbf{b}_{\text{qp}})$.
4. The no-arbitrage assumption ensures $\boldsymbol{\mu}^\top \mathbf{b}_{\text{qp}} < 1/\eta$, preventing degeneracy.

- *Context*:
	- This result is theoretically significant: it establishes that Markowitz mean-variance optimization is a *special case* of the RCK framework (specifically, its quadratic approximation). The full RCK problem is a higher-order generalization that captures nonlinear risk-return trade-offs missed by the quadratic approximation. This means RCK can be seen as "Markowitz with better tail risk control" — it agrees with Markowitz when returns are close to 1 (where the quadratic approximation is valid) but provides superior risk management when returns exhibit extreme values. The numerical experiments confirm this: QRCK (= Markowitz) underperforms RCK on the growth-risk frontier, particularly when the return distribution has heavy tails.

---

# 7. Numerical Simulations

## 7.1 Finite Outcomes

### Setup

- $n = 20$ bets (19 risky, 1 cash), $K = 100$ possible outcomes.
- Returns $r_{ij}$ sampled uniformly from $[0.7, 1.3]$, with 30 entries set to $0.2$ and 30 to $2.0$ (creating extreme returns). The probability of at least one extreme return in a vector is approximately 0.45.

### Kelly vs. RCK (Table 1)

| Bet | Growth Rate $E[\log(\mathbf{r}^\top \mathbf{b})]$ | Risk Bound $\alpha^\lambda$ | Simulated $\Pr(W_{\min} < \alpha)$ |
|---|---|---|---|
| Kelly | 0.062 | — | 0.397 |
| RCK, $\lambda = 6.456$ | 0.043 | 0.100 | 0.073 |
| RCK, $\lambda = 5.500$ | 0.047 | 0.141 | 0.099 |

- Kelly-optimal bets experience 40% drawdown probability — unacceptable for most investors.
- RCK at $\lambda = 6.456$ (corresponding to $\alpha = 0.7, \beta = 0.1$) achieves 7.3% drawdown risk — comfortably below the 10% bound.
- RCK at $\lambda = 5.500$ achieves drawdown risk near 10%, with growth rate 0.047 — significantly above the guarantee of 0.043.
- The optimal value of the true (intractable) problem (5) lies between 0.047 and 0.062.

- *Context*:
	- The gap between the theoretical bound (10%) and simulated risk (7.3%) reflects the conservatism of the bound. This gap is the "price" of tractability — the convex restriction excludes some feasible bets. But the gap is modest (about 30%), suggesting the bound is reasonably tight. The tightness is a consequence of the Wald-type identity in Lemma 5: the bound is based on a stopping-time inequality that is tight for random walks with small drift, which is the relevant regime here (growth rates are a few percent per period).

### RCK vs. QRCK (Table 2)

| Bet | Growth Rate | Risk Bound | Simulated Risk |
|---|---|---|---|
| QRCK, $\lambda = 0.000$ | 0.054 | 1.000 | 0.218 |
| QRCK, $\lambda = 6.456$ | 0.027 | 0.100 | 0.025 |
| QRCK, $\lambda = 2.800$ | 0.044 | 0.368 | 0.100 |

- QRCK at the same $\lambda$ is significantly more conservative than RCK (growth rate 0.027 vs. 0.043 at $\lambda = 6.456$).
- QRCK provides no formal drawdown guarantee (the bound applies only to the full RCK formulation), but empirically the drawdown risk is low.
- The QRCK bets are less concentrated than Kelly bets: the Kelly solution concentrates on outcome 4, while QRCK and RCK spread bets more broadly (Figure 3).

### Growth-Risk Frontier (Figure 4)

The RCK frontier strictly dominates both QRCK and fractional Kelly:

- At 10% drawdown risk, RCK achieves growth rate 0.047 vs. fractional Kelly's 0.035 — a 34% improvement.
- The dominance is consistent across the entire frontier.

- *Context*:
	- The RCK dominance over fractional Kelly is the paper's strongest practical result. It demonstrates that a principled constraint-based approach consistently outperforms the heuristic scaling approach, often by a substantial margin. The economic interpretation: fractional Kelly uniformly dilutes *all* bets, including bets with excellent risk-adjusted profiles. RCK, by contrast, optimally allocates the risk budget across bets, concentrating exposure where the risk-adjusted return is highest. This is the same principle that makes Markowitz portfolios superior to equal-weight portfolios — diversification should be guided by risk-return trade-offs, not uniform scaling.

## 7.2 General Returns (Lognormal Mixture)

### Setup

- $n = 20$ bets, returns drawn from a mixture of two lognormals: $\mathbf{r} \sim 0.5 \cdot \log\mathcal{N}(\boldsymbol{\nu}_1, \boldsymbol{\Sigma}_1) + 0.5 \cdot \log\mathcal{N}(\boldsymbol{\nu}_2, \boldsymbol{\Sigma}_2)$.
- $10^6$ samples for optimization, $10^4$ trajectories of length 100 for Monte Carlo validation.
- The QRCK solution is used to warm-start the primal-dual stochastic gradient method.

### Kelly vs. RCK (Table 3)

| Bet | Growth Rate | Risk Bound | Simulated Risk |
|---|---|---|---|
| Kelly | 0.077 | — | 0.569 |
| RCK, $\lambda = 6.456$ | 0.039 | 0.100 | 0.080 |
| RCK, $\lambda = 5.700$ | 0.043 | 0.131 | 0.101 |

- Kelly's drawdown risk of 0.569 means the investor experiences a 30%+ drawdown more than half the time.
- RCK at $\lambda = 6.456$ reduces this to 8% while maintaining growth rate 0.039.
- The sample CDF of $W_{\min}$ (Figure 6) confirms the bound $\alpha^\lambda$ is a reasonable upper envelope.

### Growth-Risk Frontier (Figure 7)

- RCK again dominates QRCK, though the gap is smaller than in the finite case.
- Fractional Kelly performs comparably to RCK in this instance (lognormal returns are relatively well-behaved).

- *Context*:
	- The lognormal mixture example is designed to resemble a financial portfolio setting — a realistic model of multi-asset returns with regime switching (the two mixture components represent different market regimes). The finding that fractional Kelly performs comparably to RCK in this setting may reflect the fact that lognormal returns are relatively light-tailed compared to the extreme returns in the finite example ($r_i = 0.2$ or $2.0$). In heavy-tailed or highly concentrated return distributions, the RCK advantage is likely more pronounced.

---

# Appendix A: Miscellaneous Lemmas

## Lemma 1: Optimality of the Trivial Bet

The all-cash bet $\mathbf{b} = \mathbf{e}_n$ is optimal for both the Kelly problem (1) and the RCK problem (7) if and only if $E[r_i] \leq 1$ for all risky bets. The proof uses Jensen's inequality for the forward direction and a directional derivative argument for the converse.

## Lemma 2: Existence of Optimal Dual Variable

The RCK problem always has an optimal dual variable $\kappa^*$. If $\mathbf{e}_n$ is optimal, then $\kappa^* = 0$. Otherwise, Slater's condition is satisfied (strict feasibility of the risk constraint for small $f$ in a fractional bet), and the result follows from Rockafellar's duality theorem.

## Lemma 3: Projection onto $\Delta_\varepsilon$

The projection $\Pi(\mathbf{z})$ onto $\Delta_\varepsilon = \{\mathbf{b} \mid \mathbf{1}^\top \mathbf{b} = 1, \mathbf{b} \geq 0, b_n \geq \varepsilon\}$ has the closed-form $\Pi(\mathbf{z}) = (\mathbf{z} - \nu \mathbf{1})_{+,\varepsilon}$ where $\nu$ is found by bisection from $\mathbf{1}^\top (\mathbf{z} - \nu \mathbf{1})_{+,\varepsilon} = 1$.

## Lemma 4: RCK Optimality Conditions

Derives the KKT conditions (13) using directional derivatives of the Lagrangian. The proof shows that conditions (13) are both necessary and sufficient for $(\mathbf{b}^*, \kappa^*)$ to solve the RCK problem.

## Lemma 5: Tilted Measure Identity

For an IID sequence with partial sums $S_n$ and a stopping time $\tau$:

$$
E[\exp(-\lambda S_\tau - \tau \psi(\lambda)) \mid \tau < \infty] \cdot \Pr(\tau < \infty) \leq 1
$$

where $\psi(\lambda) = \log E[\exp(-\lambda X)]$. This is a modification of Wald's identity from sequential analysis. The proof defines the tilted measure $d\mu_\lambda(x) = \exp(-\lambda x - \psi(\lambda)) d\mu(x)$ and sums $\Pr_{\mu_\lambda}(\tau = n)$ over $n$.

- *Context*:
	- Lemma 5 is the mathematical engine of the paper. The substitution $X = \log(\mathbf{r}^\top \mathbf{b})$, $S_n = v_n = \log w_n$, and $\psi(\lambda) = \log E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}]$ converts the abstract random-walk stopping-time bound into the concrete drawdown risk bound (6). The condition $\psi(\lambda) \leq 0$ — equivalently $E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] \leq 1$ — ensures that the tilted random walk has non-positive drift, which guarantees that the stopping time $\tau$ has probability bounded by $\alpha^\lambda$.

---

# MATHEMATICS

The mathematical framework of this paper proceeds through a chain of five linked derivations. First, the classical Kelly growth maximization is formulated as a concave program. Second, drawdown risk is defined via the infimum of the multiplicative wealth process. Third, a tilted probability measure argument converts the intractable path-dependent drawdown constraint into a tractable one-step moment condition — the paper's central analytical innovation. Fourth, this condition is embedded into the Kelly problem to produce the convex RCK formulation. Fifth, a quadratic Taylor approximation connects RCK to the classical Markowitz mean-variance framework. The derivation chain follows: growth rate → Kelly problem → drawdown definition → tilted measure bound → RCK formulation → optimality conditions → Taylor expansion → QRCK → Markowitz equivalence.

### 1. Growth Rate and Kelly Problem

The bettor allocates fractions $\mathbf{b} \in \mathbb{R}^n$ ($\mathbf{b} \geq 0$, $\mathbf{1}^\top \mathbf{b} = 1$) across $n$ bets with random return vector $\mathbf{r} \in \mathbb{R}^n_+$, where $r_n = 1$ (cash). Wealth at time $t$ under IID returns is $w_t = \prod_{s=1}^{t-1} (\mathbf{r}_s^\top \mathbf{b})$, and log-wealth is the random walk $v_t = \sum_{s=1}^{t-1} \log(\mathbf{r}_s^\top \mathbf{b})$.

**Expected Growth Rate ($g(\mathbf{b})$):**

$$
g(\mathbf{b}) = E[\log(\mathbf{r}^\top \mathbf{b})] \tag{G}
$$

This is the drift of the log-wealth random walk. The **Kelly gambling problem** maximizes this concave function:

**Kelly Problem:**

$$
\begin{aligned}
&\text{maximize} \quad E[\log(\mathbf{r}^\top \mathbf{b})] \\
&\text{subject to} \quad \mathbf{1}^\top \mathbf{b} = 1, \quad \mathbf{b} \geq 0
\end{aligned} \tag{1}
$$

Concavity of the objective follows from the concavity of $\log$ composed with the linear function $\mathbf{r}^\top \mathbf{b}$. If $\mathbf{b}^*$ is Kelly optimal and $\tilde{\mathbf{b}}$ is any other bet, then $w_t^* / \tilde{w}_t \to \infty$ almost surely, since the log-wealth difference $\log w_t^* - \log \tilde{w}_t$ is a random walk with positive drift $g(\mathbf{b}^*) - g(\tilde{\mathbf{b}}) > 0$.

### 2. Drawdown Risk

**Minimum Wealth ($W_{\min}$):**

$$
W_{\min} = \inf_{t = 1, 2, \ldots} w_t \tag{W}
$$

The **drawdown** is $1 - W_{\min}$. The **drawdown risk** at threshold $\alpha \in (0, 1)$ is:

$$
\Pr(W_{\min} < \alpha)
$$

This is the probability that the wealth trajectory ever drops below $\alpha$ before the positive drift carries it upward. There is no closed-form expression for this probability in terms of $\mathbf{b}$ for general return distributions.

### 3. Tilted Measure Bound (Lemma 5)

The derivation of the drawdown risk bound relies on a stopping-time identity from random walk theory. Let $X_1, X_2, \ldots$ be IID with distribution $\mu$, $S_n = X_1 + \cdots + X_n$, and $\tau$ a stopping time. Define the cumulant generating function:

**Cumulant Generating Function ($\psi(\lambda)$):**

$$
\psi(\lambda) = \log E[\exp(-\lambda X)] = \log \int \exp(-\lambda x) \, d\mu(x) \tag{$\psi$}
$$

**Tilted Measure Identity:**

$$
E[\exp(-\lambda S_\tau - \tau \psi(\lambda)) \mid \tau < \infty] \cdot \Pr(\tau < \infty) \leq 1 \tag{L5}
$$

The proof defines the tilted probability measure $d\mu_\lambda(x) = \exp(-\lambda x - \psi(\lambda)) \, d\mu(x)$ and observes that $\Pr_{\mu_\lambda}(\tau = n) = E[\exp(-\lambda S_n - n\psi(\lambda)) \cdot \mathbf{1}_{\{\tau = n\}}]$. Summing over $n$ gives $\Pr_{\mu_\lambda}(\tau < \infty)$ on the left and the conditional expectation times $\Pr(\tau < \infty)$ on the right. Since $\Pr_{\mu_\lambda}(\tau < \infty) \leq 1$, the bound follows.

### 4. Drawdown Risk Bound

Applying Lemma 5 with $X = \log(\mathbf{r}^\top \mathbf{b})$, $S_n = v_n = \log w_n$, $\tau = \inf\{t \geq 1 \mid w_t < \alpha\}$, and noting $\psi(\lambda) = \log E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}]$:

When $E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] \leq 1$, we have $\psi(\lambda) \leq 0$, so $-\tau \psi(\lambda) \geq 0$ and can be dropped from (L5):

$$
E[\exp(-\lambda \log w_\tau) \mid \tau < \infty] \cdot \Pr(\tau < \infty) \leq 1
$$

Since $w_\tau < \alpha$ when $\tau < \infty$, we have $\exp(-\lambda \log w_\tau) > \alpha^{-\lambda}$:

$$
\alpha^{-\lambda} \cdot \Pr(W_{\min} < \alpha) < 1
$$

**Drawdown Risk Bound:**

$$
E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] \leq 1 \implies \Pr(W_{\min} < \alpha) < \alpha^\lambda = \beta \tag{6}
$$

where $\lambda = \log \beta / \log \alpha$ and $\alpha, \beta \in (0, 1)$. The left-hand side is a convex function of $\mathbf{b}$ (since $x^{-\lambda}$ is convex for $x > 0$ and $\lambda > 0$), making this a tractable convex constraint.

### 5. Risk-Constrained Kelly Problem and Optimality

**RCK Problem:**

$$
\begin{aligned}
&\text{maximize} \quad E[\log(\mathbf{r}^\top \mathbf{b})] \\
&\text{subject to} \quad \mathbf{1}^\top \mathbf{b} = 1, \quad \mathbf{b} \geq 0, \\
&\phantom{\text{subject to} \quad} E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] \leq 1
\end{aligned} \tag{7}
$$

The Lagrangian with dual variable $\kappa \geq 0$ for the risk constraint is:

$$
L(\mathbf{b}, \kappa) = -E[\log(\mathbf{r}^\top \mathbf{b})] + \kappa(E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] - 1) \tag{12}
$$

By Lemma 2, an optimal dual variable $\kappa^*$ always exists. By Lemma 4, the KKT conditions are:

**RCK Optimality Conditions:**

$$
\begin{aligned}
&\mathbf{1}^\top \mathbf{b}^* = 1, \quad \mathbf{b}^* \geq 0, \quad E[(\mathbf{r}^\top \mathbf{b}^*)^{-\lambda}] \leq 1 \\
&\kappa^* \geq 0, \quad \kappa^*(E[(\mathbf{r}^\top \mathbf{b}^*)^{-\lambda}] - 1) = 0 \\
&E\left[\frac{r_i}{\mathbf{r}^\top \mathbf{b}^*}\right] + \kappa^* \lambda E\left[\frac{r_i}{(\mathbf{r}^\top \mathbf{b}^*)^{\lambda+1}}\right] \begin{cases} \leq 1 + \kappa^* \lambda & b_i^* = 0 \\ = 1 + \kappa^* \lambda & b_i^* > 0 \end{cases}
\end{aligned} \tag{13}
$$

The third condition states that for active bets, the marginal growth contribution plus the risk-penalty-weighted marginal risk contribution equals a constant. This is derived by differentiating $L(\mathbf{b}, \kappa^*)$ with respect to $\mathbf{b}$ along feasible directions $\mathbf{b}_\varepsilon = (1 - \varepsilon)\mathbf{b}^* + \varepsilon \mathbf{b}$ and applying the right-sided derivative criterion $\partial^+ \varphi(0) \geq 0$ where $\varphi(\varepsilon) = L(\mathbf{b}_\varepsilon, \kappa^*)$.

### 6. Quadratic Approximation and Markowitz Connection

Define $\boldsymbol{\rho} = \mathbf{r} - \mathbf{1}$ (excess return), $\boldsymbol{\mu} = E[\boldsymbol{\rho}]$, $\mathbf{S} = E[\boldsymbol{\rho}\boldsymbol{\rho}^\top]$ (raw second moment), and $\boldsymbol{\Sigma} = \mathbf{S} - \boldsymbol{\mu}\boldsymbol{\mu}^\top$ (covariance). With $\mathbf{1}^\top \mathbf{b} = 1$, we have $\mathbf{r}^\top \mathbf{b} = 1 + \boldsymbol{\rho}^\top \mathbf{b}$.

**Taylor approximations** about $\boldsymbol{\rho}^\top \mathbf{b} \approx 0$:

$$
\log(\mathbf{r}^\top \mathbf{b}) \approx \boldsymbol{\rho}^\top \mathbf{b} - \frac{1}{2}(\boldsymbol{\rho}^\top \mathbf{b})^2
$$

$$
(\mathbf{r}^\top \mathbf{b})^{-\lambda} \approx 1 - \lambda \boldsymbol{\rho}^\top \mathbf{b} + \frac{\lambda(\lambda+1)}{2}(\boldsymbol{\rho}^\top \mathbf{b})^2
$$

**QRCK Problem:**

$$
\begin{aligned}
&\text{maximize} \quad \boldsymbol{\mu}^\top \mathbf{b} - \frac{1}{2}\mathbf{b}^\top \mathbf{S} \mathbf{b} \\
&\text{subject to} \quad \mathbf{1}^\top \mathbf{b} = 1, \quad \mathbf{b} \geq 0, \\
&\phantom{\text{subject to} \quad} -\lambda \boldsymbol{\mu}^\top \mathbf{b} + \frac{\lambda(\lambda+1)}{2}\mathbf{b}^\top \mathbf{S} \mathbf{b} \leq 0
\end{aligned} \tag{14}
$$

where the expectations have been written in terms of $\boldsymbol{\mu}$ and $\mathbf{S}$: $E[\boldsymbol{\rho}^\top \mathbf{b}] = \boldsymbol{\mu}^\top \mathbf{b}$ and $E[(\boldsymbol{\rho}^\top \mathbf{b})^2] = \mathbf{b}^\top \mathbf{S} \mathbf{b}$.

By strong Lagrange duality, dualizing the risk constraint with multiplier $\nu \geq 0$ and substituting $\mathbf{S} = \boldsymbol{\mu}\boldsymbol{\mu}^\top + \boldsymbol{\Sigma}$ shows that the QRCK solution $\mathbf{b}_{\text{qp}}$ solves:

$$
\begin{aligned}
&\text{maximize} \quad (1 - \eta \boldsymbol{\mu}^\top \mathbf{b}_{\text{qp}}) \boldsymbol{\mu}^\top \mathbf{b} - \frac{\eta}{2}\mathbf{b}^\top \boldsymbol{\Sigma} \mathbf{b} \\
&\text{subject to} \quad \mathbf{1}^\top \mathbf{b} = 1, \quad \mathbf{b} \geq 0
\end{aligned} \tag{17}
$$

for some $\eta > 0$. Under the no-arbitrage assumption ($\boldsymbol{\mu}^\top \mathbf{b} > 0$ and $\mathbf{b} \geq 0$ implies $\mathbf{b}^\top \boldsymbol{\Sigma} \mathbf{b} > 0$), we have $\boldsymbol{\mu}^\top \mathbf{b}_{\text{qp}} < 1/\eta$, so $(1 - \eta \boldsymbol{\mu}^\top \mathbf{b}_{\text{qp}}) > 0$ and problem (17) is equivalent to the Markowitz problem (15) with $\gamma = \eta / (1 - \eta \boldsymbol{\mu}^\top \mathbf{b}_{\text{qp}})$.

---

### **1. Problem:**

The classic Kelly criterion maximizes the long-term growth rate of wealth but ignores the *path* of wealth — specifically, the probability of experiencing a severe drawdown (drop from peak wealth) before the asymptotic growth dominates. Existing remedies (fractional Kelly, Markowitz optimization) either apply ad hoc scaling without a principled risk constraint or trade off the wrong pair of objectives (mean vs. variance rather than growth rate vs. drawdown probability). The paper asks: can drawdown risk be directly constrained in a computationally tractable way?

### **2. Setup:**

The environment consists of $n$ bets with IID random returns $\mathbf{r} \in \mathbb{R}^n_+$ (one of which is a risk-free cash asset with $r_n = 1$). The bettor allocates a fixed fraction $\mathbf{b}$ of wealth to each bet at every epoch, with $\mathbf{b} \geq 0$ and $\mathbf{1}^\top \mathbf{b} = 1$. Wealth evolves multiplicatively: $w_t = \prod_{s=1}^{t-1} (\mathbf{r}_s^\top \mathbf{b})$. The log-wealth process $v_t = \log w_t$ is a random walk with drift $E[\log(\mathbf{r}^\top \mathbf{b})]$ and variance $\text{Var}[\log(\mathbf{r}^\top \mathbf{b})]$. The drawdown risk $\Pr(W_{\min} < \alpha)$ is the probability that the infimum of the wealth process falls below threshold $\alpha$.

### **3. Key Idea:**

The paper replaces the intractable constraint $\Pr(W_{\min} < \alpha) < \beta$ with a sufficient convex condition $E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] \leq 1$ (where $\lambda = \log \beta / \log \alpha$), derived via a tilted probability measure argument from random walk theory. This transformation converts an infinite-horizon path-dependent probabilistic constraint into a single-step moment condition that is convex in the bet vector $\mathbf{b}$. The resulting Risk-Constrained Kelly (RCK) problem is a convex optimization problem solvable by standard methods, with a single risk-aversion parameter $\lambda$ that controls the entire growth-risk trade-off.

### **4. Assumptions:**

**Explicit:**
- **IID returns**: payoffs at each epoch are independent and identically distributed.
- **Cash asset**: the $n$-th asset has return $r_n = 1$ almost surely.
- **Positive wealth**: initial wealth $w_1 = 1$ and all subsequent wealth must be positive ($b_n > 0$ guarantees this).
- **Finite expected returns**: $E[r_i] < \infty$ for all $i$.
- **Fixed fractions**: the bet vector $\mathbf{b}$ is constant across all epochs.

**Implicit:**
- **Known return distribution**: the optimization requires either the exact return distribution (for expectations) or a sufficient number of samples for Monte Carlo estimation.
- **No transaction costs or market impact**: wealth transitions are frictionless.
- **Strong Lagrange duality**: the existence of the optimal dual variable $\kappa^*$ (Lemma 2) relies on Slater's condition being satisfied.
- **Stationarity**: the return distribution does not change over time.

### **5. Limitation:**

- **Feasible set restriction**: RCK is a restriction of the ideal drawdown-constrained problem. There may exist bets that satisfy the true drawdown constraint but violate the convex bound — these are excluded from the RCK feasible set, potentially sacrificing growth.
- **Bound conservatism**: Monte Carlo simulations show the bound is typically 30% above the true drawdown risk. This gap represents suboptimal growth that could theoretically be recovered with a tighter bound.
- **Static portfolio**: the bet vector $\mathbf{b}$ is fixed for all epochs. No dynamic adjustment as wealth approaches or recedes from the threshold $\alpha$ — a significant limitation for real-world applications where rebalancing is possible.
- **IID requirement**: non-stationary or serially correlated return environments violate the IID assumption on which the tilted measure bound depends.
- **Distribution knowledge**: the framework assumes the return distribution is known or accurately sampled. Model misspecification in the return distribution propagates directly into the risk bound.

### **6. Relevance & Open Questions:**

This paper provides the first principled method for directly constraining drawdown risk in Kelly-type growth optimization. It fills a gap between the pure-growth Kelly criterion and the risk-but-not-growth Markowitz framework by optimizing growth *subject to* a drawdown risk budget.

- **Drawdown as a risk measure for RL**: in portfolio RL, drawdown is a more operationally meaningful risk measure than variance or CVaR because it directly captures the investor's psychological tolerance for loss. The RCK bound suggests that a convex penalty of the form $E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}]$ could serve as a tractable risk constraint in an RL objective, replacing or supplementing variance-based penalties.
- **Connection to parameter uncertainty**: this paper addresses drawdown risk under *known* distributions; [[PNOTES - Optimal Betting Under Parameter Uncertainty - Improving the Kelly Criterion]] addresses overbetting under *uncertain* distributions. A complete framework would combine both — constrain drawdown risk *and* shrink for parameter uncertainty. The RCK parameter $\lambda$ and Baker's shrinkage factor $k^*$ operate on different axes of the risk space.
- **Open question — dynamic RCK**: can the RCK framework be extended to allow dynamic rebalancing, where $\mathbf{b}_t$ depends on the current wealth level $w_t$? The IID assumption would need to be relaxed, and the tilted measure argument would require a time-inhomogeneous extension.
- **Open question — non-IID returns**: financial markets exhibit autocorrelation, regime switching, and time-varying volatility. Extending the drawdown bound to non-IID returns is the most important open problem identified by the paper.
- **Open question — computational scalability**: the primal-dual stochastic gradient method can be slow to converge for high-dimensional problems. Can second-order methods or variance-reduction techniques accelerate the solution of RCK in large-scale portfolio settings?

---

### Integration:

* **Problem:** Busseti, Ryu, and Boyd (2016) formalize the drawdown risk control problem that fractional Kelly and Markowitz address only approximately. For a research program exploring Kelly-based objectives in portfolio RL, the core contribution is the demonstration that drawdown risk — the most operationally relevant risk measure for long-horizon investors — can be tractably constrained via the convex condition $E[(\mathbf{r}^\top \mathbf{b})^{-\lambda}] \leq 1$. This opens the possibility of embedding drawdown-aware constraints directly into RL training objectives, rather than relying on post-hoc risk scaling or variance proxies. The single parameter $\lambda$ provides a natural knob for sweeping the growth-risk frontier during policy training, analogous to the temperature parameter in entropy-regularized RL.

* **Limitation:** The most consequential limitation for integration into portfolio RL is the static bet assumption — the bet vector $\mathbf{b}$ is constant across all epochs, whereas an RL agent generates a policy $\pi(\mathbf{b}_t \mid s_t)$ that varies with state. The IID return assumption is similarly restrictive: financial returns exhibit serial correlation and regime dependence, which the tilted measure bound does not account for. Extending the drawdown bound to state-dependent dynamic policies under non-IID returns would require replacing the Wald-type stopping-time argument with a martingale-based approach or a recursive Bellman-type decomposition of the drawdown probability — a significant theoretical challenge. Additionally, the requirement for accurate knowledge of the return distribution parallels the distributional estimation challenge in model-based RL: the agent must not only learn optimal allocations but also accurately model return dynamics to ensure the drawdown bound holds.
