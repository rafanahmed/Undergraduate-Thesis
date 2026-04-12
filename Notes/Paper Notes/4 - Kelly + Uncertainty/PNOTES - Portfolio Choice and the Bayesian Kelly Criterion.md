[[Portfolio Choice and the Bayesian Kelly Criterion.pdf.pdf]]

# **Abstract:**

The paper addresses the problem of **optimal portfolio choice** when the drift parameter of the underlying asset process is unobserved. Where the classical **Kelly criterion** assumes fixed, known parameters — yielding the well-known rule $f^* = \mu / \sigma^2$ — real investors face the **"Unknown Parameter" problem**, requiring simultaneous optimization of capital growth and parameter estimation. Browne and Whitt analyze this dual problem by treating the unknown drift as a random variable with a prior distribution and applying **Bayesian updating** as information accumulates.

- The paper's methodology begins in discrete time with a **Random Walk in a Random Environment (RWIRE)**, where the success probability $\theta$ is drawn from a **Beta prior** and updated via conjugate Bayesian inference.
- Through a carefully constructed **diffusion limit**, the discrete RWIRE converges weakly to a continuous-time process — specifically, a Brownian motion with a **normally distributed random drift**.
- The core result (**Theorem 6**) establishes that under logarithmic utility, the optimal investment policy is the **certainty equivalent** of the deterministic case: the investor replaces the unknown drift with its **posterior mean** and applies the standard Kelly/Merton rule.
- The paper quantifies the **financial value of learning** — the gap between the Bayesian investor's value function and that of a perfectly informed investor — showing it grows only logarithmically in time.
- For **power utility** and other HARA utility functions, the certainty equivalence principle **breaks down**: the optimal policy must account for the volatility of the posterior estimate itself.

- *Context*:
	- This paper occupies a pivotal position in the Kelly criterion literature. It bridges three traditions: Kelly's (1956) information-theoretic approach to gambling, Merton's (1971) continuous-time portfolio selection, and classical Bayesian filtering theory. The key intellectual contribution is not merely that "you should use the posterior mean" — that is intuitive — but that this intuitive rule is *provably optimal* for log-utility, and *provably suboptimal* for other utility functions. This separation result has deep implications for the design of adaptive investment strategies.
	- The paper connects directly to [[PNOTES - Optimal Betting Under Parameter Uncertainty - Improving the Kelly Criterion]], which addresses the same parameter uncertainty problem from a *frequentist* shrinkage perspective, and to [[PNOTES - Modified Kelly criteria]], which uses Bayesian estimation under various loss functions. Browne and Whitt's contribution is the rigorous *continuous-time* extension via diffusion limits.

---

# 1. Introduction

## Motivation: Learning While Earning

The paper opens by framing the fundamental tension in sequential investment under uncertainty: the investor must simultaneously *exploit* current estimates to maximize wealth and *explore* the information embedded in market returns to refine those estimates.

- In the classical Kelly framework, the success probability $\theta$ (discrete) or drift $\mu$ (continuous) is known. The optimal strategy is myopic — bet or invest the Kelly fraction at every step, independent of history.
- When these parameters are unknown, the problem changes qualitatively: past observations carry informational value beyond their immediate financial impact.
- The paper focuses on the case where the unknown parameter is a **fixed random variable** drawn once from a prior — as opposed to a time-varying stochastic process. This is the simplest non-trivial formulation of learning in sequential investment.

- *Context*:
	- The distinction between a *fixed unknown* parameter and a *stochastically evolving* parameter is critical. A fixed unknown is progressively revealed by observations — the posterior tightens over time, and the investor converges toward the true optimal. A stochastically evolving parameter never fully reveals itself, and the investor must perpetually balance estimation recency against noise. The paper's concluding remarks flag the stochastic-parameter case as an open problem.

## Classical Kelly Criterion Review

The paper reviews the foundational Kelly result for binary gambling:

- A gambler repeatedly wagers a fraction $f$ of current wealth $V_n$ on a biased coin with success probability $\theta > 1/2$.
- Wealth evolves multiplicatively: $V_n = V_0 \prod_{i=1}^n (1 + f_i Z_i)$, where $Z_i \in \{+1, -1\}$.
- For constant $f$, the **long-run growth rate** converges almost surely (by the Law of Large Numbers) to:

$$
G(f) = \theta \ln(1 + f) + (1 - \theta) \ln(1 - f)
$$

- The unique maximizer is $f^* = 2\theta - 1 = E[Z]$, and the corresponding maximum growth rate is:

$$
\Lambda(\theta) = \ln 2 + \theta \ln \theta + (1 - \theta) \ln(1 - \theta) \tag{3}
$$

- This can be rewritten as $\Lambda(\theta) = \ln 2 - H(\theta)$, where $H(\theta)$ is the **binary entropy** of the success probability.

- *Context*:
	- The entropy interpretation $\Lambda(\theta) = \ln 2 - H(\theta)$ provides the information-theoretic backbone of the Kelly criterion. Maximum entropy ($\theta = 1/2$) corresponds to a *fair game* with zero growth. As $\theta$ deviates from $1/2$ — i.e., as the game becomes more predictable — entropy decreases and growth increases. The Kelly gambler's profit is literally the *information advantage* over a fair game, measured in nats (or bits, if using $\log_2$). This is the original Kelly (1956) insight.

---

# 2. The Non-Bayesian Case

## Finite-Horizon Terminal Wealth

When parameters are known constants, the paper derives the optimal strategy for maximizing $E[\ln V_N]$ over $N$ rounds.

- The **Bellman value function** for current wealth $x$ with $N - n + 1$ rounds remaining is:

$$
F_n(x) = \ln x + (N - n + 1) \max_f E[\ln(1 + f Z)]
$$

- Because the increments are i.i.d., the optimal policy is **myopic and constant** — the one-step maximizer $f^*$ is also the $N$-step maximizer. There is no intertemporal trade-off when parameters are known.

- *Context*:
	- Myopia under known parameters is a defining property of logarithmic utility. For any other utility function, even with known parameters, the optimal strategy may be horizon-dependent. This myopia will carry over to the Bayesian case — the Bayesian log-utility investor is myopic with respect to the *posterior mean* — but it will fail for power utility, as shown in Section 6.

## Scaling Toward Continuous Time

To bridge discrete and continuous models, the paper introduces a small-step scaling:

- Increments become $Z_n \in \{\pm \Delta\}$ with $\theta = \frac{1}{2}(1 + \mu/\Delta)$, where $\mu$ is the desired drift.
- The rescaled Kelly fraction becomes $f^* = (2\theta - 1)/\Delta$, which in the limit gives:

$$
f^* = \frac{\mu}{\sigma^2} \tag{8}
$$

- This is the classical **continuous-time Kelly fraction** — the optimal proportion of wealth to invest in the risky opportunity when the drift $\mu$ and volatility $\sigma$ are known.

## Continuous-Time Wealth SDE

In the diffusion limit, wealth evolves according to:

$$
dV_t = f_t V_t \mu \, dt + \sigma f_t V_t \, dW_t \tag{7}
$$

- The **optimal value function** for terminal time $T$ is:

$$
F(x) = \ln x + \frac{T}{2} \frac{\mu^2}{\sigma^2} \tag{10}
$$

- *Context*:
	- Equation (10) has a clean interpretation: the growth rate of log-wealth under optimal Kelly investing is $\mu^2 / (2\sigma^2)$, which is the **squared Sharpe ratio** divided by 2. This is the information ratio of the risky opportunity, measured in terms of its contribution to log-wealth growth per unit time.

## Financial Value of Randomization

The paper introduces a key thought experiment: what if an oracle could tell the investor, before the start of trading, the true value of a randomly distributed drift $Z \sim N(\mu, c)$?

- With randomization (perfect information), the value function becomes:

$$
F^*(x) = \ln x + \frac{T}{2\sigma^2} E[Z^2] = \ln x + \frac{T}{2\sigma^2}(\mu^2 + c) \tag{12}
$$

- The **gain from perfect information** is $cT / (2\sigma^2)$ — linear in time $T$ and in the prior variance $c$.

- *Context*:
	- This linear gain serves as an upper bound on the benefit of learning. The Bayesian investor cannot match this performance because they must learn $Z$ gradually from noisy observations. The key result of Section 5 will show that the Bayesian investor's shortfall relative to this benchmark grows only *logarithmically* in $T$ — a remarkably small penalty for learning.

---

# 3. Bayesian Gambling in Discrete Time

This section introduces the **RWIRE** (Random Walk in a Random Environment) model, the discrete-time Bayesian gambling framework.

## The RWIRE Model

- The success probability $\theta$ is no longer a known constant but an **unobserved random variable** drawn from a prior distribution $f_0(u)$ on $(0, 1)$.
- Conditional on $\theta$, the increments $Z_1, Z_2, \ldots$ are i.i.d. Bernoulli$(\theta)$ — the game is standard Kelly gambling from the gambler's perspective at each step, but the win probability is unknown.
- The sufficient statistic for $\theta$ after $n$ rounds is the pair $(n, Y_n)$, where $Y_n = \sum_{i=1}^n W_i$ is the cumulative number of wins (with $W_i = (Z_i + 1)/2$).

## Posterior Distribution

Given the sufficient statistic, the **posterior distribution** of $\theta$ is:

$$
dP(\theta \le u \mid Y_n = y) = \frac{u^{(n+y)/2}(1-u)^{(n-y)/2} f_0(u) \, du}{\int_0^1 v^{(n+y)/2}(1-v)^{(n-y)/2} f_0(v) \, dv} \tag{19}
$$

## Beta Conjugate Prior

For analytical tractability, the paper specializes to the **Beta prior**:

$$
\theta \sim \text{Beta}(\alpha, \beta), \quad f_0(u) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} u^{\alpha - 1}(1 - u)^{\beta - 1} \tag{20}
$$

- Prior mean: $E(\theta) = \alpha / (\alpha + \beta)$
- Prior variance: $V(\theta) = \alpha\beta / [(\alpha + \beta)^2(\alpha + \beta + 1)]$

The posterior is also Beta (conjugacy), with updated parameters:

$$
\theta \mid Y_n = y \sim \text{Beta}\left(\alpha + \frac{n + y}{2}, \, \beta + \frac{n - y}{2}\right)
$$

The **posterior mean** is:

$$
E(\theta \mid Y_n) = \frac{Y_n + n + 2\alpha}{2(n + \alpha + \beta)}
$$

- *Context*:
	- The Beta-Binomial conjugacy is the engine of the discrete-time Bayesian framework. Each observation updates the Beta parameters by incrementing them: a win adds $1$ to $\alpha$, a loss adds $1$ to $\beta$. The posterior mean is a *weighted average* of the prior mean $\alpha/(\alpha + \beta)$ and the observed success rate $Y_n / n$, with the weights determined by the relative strength of the prior (via $\alpha + \beta$) versus the data (via $n$). As $n \to \infty$, the data overwhelm the prior and $E(\theta \mid Y_n) \to \theta$ almost surely.
	- The Beta prior is also used extensively in [[PNOTES - Modified Kelly criteria]] and [[PNOTES - Optimal Betting Under Parameter Uncertainty - Improving the Kelly Criterion]], making this a unifying element across the Kelly-under-uncertainty literature.

## Markov Structure of the Bayesian Walk

The random walk $S_n = \sum_{i=1}^n Z_i$ under the Bayesian framework is a **Markov process** in the state $(n, S_n)$ with state-dependent drift:

$$
E(S_{n+1} - S_n \mid S_n) = \frac{S_n + \alpha - \beta}{n + \alpha + \beta} \tag{22}
$$

- The drift is **self-correcting**: when $S_n$ is large (many wins), the posterior mean of $\theta$ is high, so the expected next increment is positive. The drift adapts to the accumulated evidence.

## Optimal Bayesian Betting

The discrete-time Bayesian Kelly rule emerges from the Bellman recursion. The optimal bet at step $j+1$, given the walk position $S_j$ and the number of prior observations $j$, is:

$$
f_{j+1}^*(S_j, j) = \frac{1}{\Delta} \cdot \frac{S_j + \alpha - \beta}{j + \alpha + \beta} \tag{26}
$$

This is the **certainty equivalent** of the classical Kelly fraction: replace $\theta$ with its posterior mean $E(\theta \mid S_j, j)$, then apply the standard formula $f^* = (2\theta - 1)/\Delta$.

- *Context*:
	- The certainty equivalence property is the discrete-time precursor to the paper's central result. It says: under logarithmic utility, the Bayesian gambler does not need to solve a complex stochastic control problem that accounts for future learning. They can simply treat the current posterior mean as if it were the truth, optimize myopically, and this *is* the globally optimal policy. This is a non-trivial result — it fails for every other utility function in the HARA family, as Section 6 demonstrates.

## One-Step Optimal Log Gain

The expected one-step log gain under the optimal control is the posterior analogue of $\Lambda(\theta)$:

$$
\Lambda(S_j, j) = \ln 2 + E(\theta \mid S_j, j) \ln E(\theta \mid S_j, j) + (1 - E(\theta \mid S_j, j)) \ln(1 - E(\theta \mid S_j, j)) \tag{27}
$$

- The **value function** decomposes as $F_n(x, S_k, k) = \ln x + C_i(S_k, k)$, where $C_i$ satisfies a recursion over the posterior-weighted tree of future walk positions.

---

# 4. Continuous-Time Diffusion Limits

This section is technically dense, establishing the **weak convergence** of the discrete RWIRE to a continuous-time diffusion. It provides the mathematical bridge between Sections 3 and 5.

## Parameterization for the Limit

To take the diffusion limit, the Beta prior parameters are scaled with the step size:

$$
\alpha_n = \frac{\sigma^2 n - (\mu^2 + c)}{2c\sqrt{n}} \left(\sqrt{n} + \frac{\mu}{\sigma}\right) \tag{31}
$$

$$
\beta_n = \frac{\sigma^2 n - (\mu^2 + c)}{2c\sqrt{n}} \left(\sqrt{n} - \frac{\mu}{\sigma}\right) \tag{32}
$$

- These parameterizations ensure that the discrete prior (Beta) converges, under the scaling, to a Normal$(\mu, c)$ prior on the continuous-time drift.

## The RWIRE Limit Theorem

The main convergence result (Theorem 3) establishes:

$$
\frac{\sigma S_{\lfloor nt \rfloor}^*}{\sqrt{n}} \Rightarrow X_t = (\sigma^2 + ct) W_{t/(\sigma^2 + ct)} + \mu t \tag{33}
$$

- The limit process $X_t$ is a **time-changed Brownian motion** with random drift — it can equivalently be written as $X_t = \sigma W_t + tZ$ where $Z \sim N(\mu, c)$.

- *Context*:
	- The equivalence $\sigma W_t + tZ \stackrel{d}{=} (\sigma^2 + ct) W_{t/(\sigma^2 + ct)} + \mu t$ (Equation 38) is a crucial distributional identity. On the left, the process is a standard Brownian motion plus a linear trend with random slope. On the right, it is a deterministically time-changed Brownian motion with a non-random drift. Both representations are useful: the left-hand side is more intuitive (noise + random signal), while the right-hand side facilitates the filtering calculations in Section 5.

## Kiefer Process and Brownian Sheet

The proof machinery involves the **Kiefer process**, a two-parameter stochastic process:

$$
B_t(x) = W_t(x) - x W_t(1) \tag{34}
$$

- This is a "tied-down" Brownian sheet — it is zero at $x = 0$ and $x = 1$, and for fixed $x$, satisfies $\{B_t(x) : t \ge 0\} \stackrel{d}{=} \{\sqrt{x(1-x)} \, W_t : t \ge 0\}$.
- The Kiefer process appears as the weak limit of the empirical process of the RWIRE — it captures how the walk's position varies simultaneously with time and with the unknown success probability.

- *Context*:
	- The Kiefer process is a technical tool from empirical process theory that enters because the RWIRE is a random walk whose *step distribution* is itself random. Standard CLT-type arguments fail because the distribution varies with $\theta$; the Kiefer process handles this two-dimensional variation. Readers unfamiliar with Brownian sheets can treat this section as the proof infrastructure — the applied results in Section 5 do not require direct manipulation of the Kiefer process.

---

# 5. Continuous-Time Bayesian Gambling

This section contains the paper's central results: the **optimal Bayesian control** and the **financial value of learning**.

## Filtering the Unknown Drift

Suppose the observed process satisfies $dX_t = Z \, dt + \sigma \, dW_t$ where $Z$ is an unobserved constant with prior $Z \sim N(\mu, c)$. The **Bayesian filtering** problem produces:

### Posterior distribution (Gaussian)

$$
Z \mid \mathcal{F}_t^X \sim N\left(\frac{cX_t + \mu\sigma^2}{\sigma^2 + ct}, \, \frac{\sigma^2 c}{\sigma^2 + ct}\right) \tag{42}
$$

- The **posterior mean** is $\hat{Z}_t = (cX_t + \mu\sigma^2)/(\sigma^2 + ct)$: a weighted combination of the observation-based estimate $X_t / t$ and the prior mean $\mu$, with weights that shift from the prior toward the data as $t$ increases.
- The **posterior variance** is $\sigma^2 c / (\sigma^2 + ct)$: it decreases deterministically in time, independent of the observed path.

- *Context*:
	- The deterministic decay of the posterior variance is a defining property of the Gaussian filtering problem (and a special case of the Kalman filter). It means the *rate* of learning is entirely determined by the noise level $\sigma^2$ and the prior uncertainty $c$, not by the actual observations. This is a simplification that breaks down for non-Gaussian priors, which is one reason the paper works primarily with the conjugate case.

### Filtered (innovations) representation

Under the posterior, $X_t$ satisfies the SDE:

$$
dX_t = \frac{cX_t + \mu\sigma^2}{\sigma^2 + ct} \, dt + \sigma \, dW_t' \tag{43}
$$

where $W_t'$ is the **innovations Brownian motion** — a standard Brownian motion with respect to the observation filtration $\mathcal{F}_t^X$.

- *Context*:
	- The transition from Equation (39) ($dX_t = Z \, dt + \sigma \, dW_t$ with $Z$ unobserved) to Equation (43) ($dX_t = \hat{Z}_t \, dt + \sigma \, dW_t'$ with $\hat{Z}_t$ observed) is the **Fujisaki-Kallianpur-Kunita (FKK) theorem** applied to the linear Gaussian case. It replaces the unknown true drift with its posterior mean and the original Brownian motion with the innovations process. This is a fundamental result in nonlinear filtering and is the technical key to the entire paper — it converts the partially observed control problem into a fully observed one.

## Optimal Bayesian Control (Theorem 6)

The wealth process under Bayesian uncertainty is $dV_t = f_t V_t \, dX_t$. The optimal control is:

$$
f_t^* = \frac{1}{\sigma^2} \left(\frac{cX_t + \mu\sigma^2}{\sigma^2 + ct}\right) \tag{51}
$$

This is the **certainty equivalent** of the standard Kelly rule: the unknown drift $Z$ is replaced by its posterior mean $\hat{Z}_t$, yielding $f_t^* = \hat{Z}_t / \sigma^2$.

- *Context*:
	- Equation (51) is the paper's central formula. It says: the optimal Bayesian Kelly strategy under logarithmic utility is structurally identical to the classical Kelly strategy — the only modification is to replace the true (unknown) drift with the best current estimate. This is the **Separation Principle** in action: estimation and control decouple, and the investor need only solve the filtering problem (compute $\hat{Z}_t$) and plug the result into the known-parameter formula. The principle holds here because the log-utility objective makes the Bellman equation separable.

### Convergence of Discrete to Continuous

The paper verifies that the discrete Bayesian Kelly fraction converges to the continuous-time control:

$$
\frac{2E(\theta_n \mid S_{\lfloor nt \rfloor}) - 1}{\sigma / \sqrt{n}} \Rightarrow \frac{1}{\sigma^2}\left(\frac{cX_t + \mu\sigma^2}{\sigma^2 + ct}\right) \tag{52}
$$

## Optimal Wealth and Bayesian Value Function

Under the optimal control, wealth evolves as:

$$
dV_t^* = V_t^* \left[\frac{1}{\sigma^2}\left(\frac{cX_t + \mu\sigma^2}{\sigma^2 + ct}\right)^2 dt + \frac{1}{\sigma}\left(\frac{cX_t + \mu\sigma^2}{\sigma^2 + ct}\right)dW_t\right] \tag{54}
$$

The expected log terminal wealth is:

$$
F^B(x) = \ln x + \frac{\mu^2}{2\sigma^2}T + \frac{c}{2\sigma^2}T - \frac{1}{2}\ln\left(1 + \frac{c}{\sigma^2}T\right) \tag{56}
$$

- *Context*:
	- The value function $F^B(x)$ has three terms beyond $\ln x$: (1) $\mu^2 T/(2\sigma^2)$, the growth from the known-mean component (same as the non-Bayesian case); (2) $cT/(2\sigma^2)$, the potential growth from the random component of the drift; and (3) $-\frac{1}{2}\ln(1 + cT/\sigma^2)$, the penalty for having to learn $Z$ rather than knowing it. The first two terms are linear in $T$; the penalty is only logarithmic. This asymmetry is the paper's most economically significant finding.

## Financial Value of Learning

The paper computes two key quantities — the **value of the Bayesian strategy over the naive strategy**, and the **cost of learning** relative to perfect information:

### Bayesian gain over naive

$$
F^B(x) - F(x) = \frac{c}{2\sigma^2}T - \frac{1}{2}\ln\left(1 + \frac{c}{\sigma^2}T\right) \tag{57}
$$

- This is always positive for $c > 0$, confirming that learning improves upon ignoring the random drift component.

### Cost of learning (information gap)

$$
F^P(x) - F^B(x) = \frac{1}{2}\ln\left(1 + \frac{c}{\sigma^2}T\right) \tag{58}
$$

- The gap between perfect information and Bayesian learning grows only **logarithmically** in $T$.

- *Context*:
	- The contrast between the linear gain from randomization ($cT / (2\sigma^2)$ from Equation 12) and the logarithmic cost of learning ($\frac{1}{2}\ln(1 + cT/\sigma^2)$ from Equation 58) is the paper's sharpest economic insight. It means that as the investment horizon grows, the fraction of potential value captured by the Bayesian learner approaches $1$ — the cost of not knowing $Z$ becomes negligible relative to the benefit of having a random drift to exploit. In the long run, the Bayesian investor's performance is *asymptotically equivalent* to the perfectly informed investor's. This is a powerful argument for the practical viability of adaptive strategies in uncertain environments.

---

# 6. Other Utility Functions

This section extends the analysis to **power utility** $u(x) = x^\eta$ (where $\eta < 1$, $\eta \neq 0$), including the HARA family. The key finding is negative: the certainty equivalence principle **fails** outside the logarithmic case.

## Power Utility: Known Parameters

With known $\theta$ and power utility, the one-step optimal bet is:

$$
f^* = \frac{1}{\Delta} \cdot \frac{\theta^{1/(1-\eta)} - (1-\theta)^{1/(1-\eta)}}{\theta^{1/(1-\eta)} + (1-\theta)^{1/(1-\eta)}} \tag{66}
$$

- For $\eta \to 0$ (log utility), this reduces to $f^* = (2\theta - 1)/\Delta$, recovering the Kelly fraction.
- For $\eta < 0$ (more risk-averse than log), the bet is *smaller* than Kelly.

## Power Utility: Bayesian Case

For the one-step Bayesian horizon with posterior mean $E(\theta \mid S_k, k)$, the optimal bet is:

$$
f^*(S_k, k) = \frac{1}{\Delta} \cdot \frac{[E(\theta \mid S_k, k)]^{1/(1-\eta)} - [1 - E(\theta \mid S_k, k)]^{1/(1-\eta)}}{[E(\theta \mid S_k, k)]^{1/(1-\eta)} + [1 - E(\theta \mid S_k, k)]^{1/(1-\eta)}} \tag{69}
$$

- This is **not** the certainty equivalent: it is not the same as plugging $E(\theta \mid S_k, k)$ into the known-$\theta$ formula (Equation 66) — it is the one-step optimal that accounts for the *current* posterior but not future learning.

## Multi-Step Power Utility Control

For the finite-horizon case with $n$ steps remaining, the optimal control involves a **nonlinear recursion**:

$$
f^*(x, S_k, k) = \frac{1}{\Delta} \cdot \frac{[\theta_k(S_k) g_{n-k}(k+1, S_k+1)]^{1/(1-\eta)} - [(1-\theta_k(S_k)) g_{n-k}(k+1, S_k-1)]^{1/(1-\eta)}}{[\theta_k(S_k) g_{n-k}(k+1, S_k+1)]^{1/(1-\eta)} + [(1-\theta_k(S_k)) g_{n-k}(k+1, S_k-1)]^{1/(1-\eta)}} \tag{70}
$$

where $g_n(r, v)$ satisfies:

$$
g_n(r, v) = \left[\theta_r(v) g_{n-1}(r+1, v+1)^\eta + (1-\theta_r(v)) g_{n-1}(r+1, v-1)^\eta\right]^{1/\eta} \tag{71}
$$

- *Context*:
	- The recursion in Equations (70)–(71) is the technical proof that certainty equivalence fails for power utility. The auxiliary function $g_n$ captures the *future value* of learning under power utility — it depends on the shape of the value function at future states, which in turn depends on the posterior distribution at those states. For log utility, $g_n$ factors out trivially because $\ln$ converts products into sums. For power utility, the nonlinear interaction between the value function's curvature and the posterior's evolution makes the problem fundamentally non-separable. The investor must hedge against the **volatility of the posterior estimate itself**, not just against the volatility of returns.
	- This breakdown of the Separation Principle is the theoretical basis for the empirical finding in [[PNOTES - Optimal Betting Under Parameter Uncertainty - Improving the Kelly Criterion]] that Kelly bets should be *shrunk* — the shrinkage partially compensates for the failure of certainty equivalence when the utility function is implicitly more risk-averse than log.

---

# 7. Concluding Remarks

The paper concludes by:

- Summarizing the main finding: for **logarithmic utility**, the certainty equivalent policy is optimal under Bayesian uncertainty, and the cost of learning is logarithmic.
- Noting that **explicit solutions** are tied to conjugate priors (Beta-Binomial in discrete time, Normal-Normal in continuous time), though the optimality of certainty equivalence for log-utility is a **general result** that does not depend on the prior being conjugate.
- Identifying open questions:
	- What is the optimal policy under **general (non-conjugate) priors**?
	- What happens when the unknown parameter is itself a **stochastic process** rather than a fixed random variable?
	- Can the diffusion limit approach be extended to models with **stochastic volatility**?

- *Context*:
	- The open question about stochastic parameters connects directly to the non-stationary bandit literature explored in [[PNOTES - Optimal Exploration-Exploitation in a Multi-Armed-Bandit Problem with Non-Stationary Rewards]]. In a non-stationary environment, the "true" parameter drifts over time, and the posterior never converges — the investor must perpetually re-estimate. Browne and Whitt's framework would need to be extended with a state-space model for the drift itself, leading to a *doubly stochastic* filtering problem.

---

# MATHEMATICS

The mathematical architecture of this paper consists of a multi-layered derivation chain that progresses from discrete-time gambling through Bayesian updating and diffusion limits to continuous-time stochastic control. The chain establishes two principal results: (1) the certainty equivalence of the optimal Bayesian Kelly rule under logarithmic utility, and (2) the quantification of the financial value of learning. The logical dependencies are: classical Kelly → Bayesian updating → RWIRE → diffusion limit → filtering → optimal control → value of information.

### 1. Multiplicative Wealth Dynamic

The fundamental discrete-time wealth process under proportional betting. The gambler wagers fraction $f_n$ of current wealth $V_{n-1}$ at each step $n$, with game increment $Z_n \in \{+1, -1\}$:

**Multiplicative Wealth ($V_n$):**

$$
V_n = V_{n-1} + f_n V_{n-1} Z_n = V_0 \prod_{i=1}^n (1 + f_i Z_i) \tag{1}
$$

where $V_0$ is initial wealth, $f_n$ is the wagered fraction at step $n$, and $Z_n$ is the payoff increment with $P(Z_n = 1) = \theta$ and $P(Z_n = -1) = 1 - \theta$.

The multiplicative structure means that percentage gains and losses compound — this is why the *logarithm* of wealth, not wealth itself, is the natural objective for long-run growth.

### 2. Classical Kelly Fraction

For constant betting $f_n = f$ with known $\theta$, the growth rate $G(f) = \theta \ln(1+f) + (1-\theta)\ln(1-f)$ is maximized by:

**Kelly Fraction ($f^*$):**

$$
f^* = 2\theta - 1 = E[Z] \tag{2}
$$

**Maximum Growth Rate ($\Lambda(\theta)$):**

$$
\Lambda(\theta) = \ln 2 + \theta \ln \theta + (1 - \theta)\ln(1-\theta) = \ln 2 - H(\theta) \tag{3}
$$

where $H(\theta) = -\theta \ln \theta - (1-\theta)\ln(1-\theta)$ is the binary entropy of the Bernoulli distribution with parameter $\theta$. The growth rate equals the informational advantage over a fair game.

### 3. Continuous-Time Wealth SDE

In the diffusion limit, with drift $\mu$ and volatility $\sigma$, wealth evolves as:

**Wealth SDE ($V_t$):**

$$
dV_t = f_t V_t \mu \, dt + \sigma f_t V_t \, dW_t \tag{7}
$$

where $f_t$ is the proportion invested and $W_t$ is a standard Brownian motion. The discrete-to-continuous scaling sends $\theta = \frac{1}{2}(1 + \mu/\Delta)$ and $\Delta \to 0$, giving the continuous-time Kelly fraction:

**Continuous-Time Kelly Fraction:**

$$
f^* = \frac{\mu}{\sigma^2} \tag{8}
$$

### 4. Continuous-Time Value Function

Under the optimal constant control $f^* = \mu/\sigma^2$, the expected log terminal wealth at horizon $T$ is:

**Non-Bayesian Value ($F(x)$):**

$$
F(x) = \ln x + \frac{T}{2}\frac{\mu^2}{\sigma^2} \tag{10}
$$

This is the baseline against which the Bayesian value function is compared. The growth rate $\mu^2/(2\sigma^2)$ is half the squared Sharpe ratio.

### 5. Value with Randomized Drift

If the drift $Z$ is a random variable with mean $\mu$ and variance $c$, and the investor has **perfect information** (knows $Z$ before trading), then:

**Perfect-Information Value ($F^*(x)$):**

$$
F^*(x) = \ln x + \frac{T}{2\sigma^2}(\mu^2 + c) \tag{12}
$$

The gain over the non-random case is $cT/(2\sigma^2)$, linear in both the prior variance $c$ and the horizon $T$. This establishes the maximum possible benefit of information.

### 6. Beta Prior and Posterior Update

When $\theta$ is unknown with prior $\theta \sim \text{Beta}(\alpha, \beta)$, the posterior after observing $n$ trials with sufficient statistic $Y_n$ (number of wins) is:

**Beta Posterior:**

$$
\theta \mid Y_n = y \sim \text{Beta}\left(\alpha + \frac{n+y}{2}, \, \beta + \frac{n-y}{2}\right) \tag{19-20}
$$

**Posterior Mean:**

$$
E(\theta \mid Y_n) = \frac{Y_n + n + 2\alpha}{2(n + \alpha + \beta)}
$$

The conjugacy of the Beta-Binomial model ensures the posterior remains in the Beta family after each observation, enabling closed-form updates. The posterior mean is a precision-weighted average of the prior mean $\alpha/(\alpha + \beta)$ and the sample success rate.

### 7. Discrete Bayesian Kelly Rule

The certainty equivalent principle applied to the discrete Kelly fraction yields:

**Bayesian Kelly Fraction (Discrete) ($f_{j+1}^*$):**

$$
f_{j+1}^*(S_j, j) = \frac{1}{\Delta} \cdot \frac{S_j + \alpha - \beta}{j + \alpha + \beta} \tag{26}
$$

where $S_j = \sum_{i=1}^j Z_i$ is the random walk position. This is identical to the standard Kelly fraction $f^* = (2\theta - 1)/\Delta$ with $\theta$ replaced by $E(\theta \mid S_j, j)$. The policy is myopic — it depends only on the current posterior, not on the remaining horizon.

### 8. Posterior Mean of Random Drift (Continuous-Time Filtering)

For the continuous-time model $dX_t = Z \, dt + \sigma \, dW_t$ with $Z \sim N(\mu, c)$ unobserved, the Gaussian filtering equations give:

**Posterior Distribution:**

$$
Z \mid \mathcal{F}_t^X \sim N\left(\frac{cX_t + \mu\sigma^2}{\sigma^2 + ct}, \, \frac{\sigma^2 c}{\sigma^2 + ct}\right) \tag{42}
$$

The posterior mean $\hat{Z}_t = (cX_t + \mu\sigma^2)/(\sigma^2 + ct)$ is a Kalman-type filter. The posterior variance $\sigma^2 c/(\sigma^2 + ct)$ decreases deterministically and is path-independent — the investor learns at a fixed rate regardless of what they observe.

### 9. Filtered Diffusion SDE

The FKK (Fujisaki-Kallianpur-Kunita) theorem converts the partially observed process into a fully observed innovations representation:

**Filtered SDE ($X_t$):**

$$
dX_t = \frac{cX_t + \mu\sigma^2}{\sigma^2 + ct} \, dt + \sigma \, dW_t' \tag{43}
$$

where $W_t'$ is the innovations Brownian motion adapted to the observation filtration $\mathcal{F}_t^X$. This SDE has **time-varying, state-dependent drift** — the drift coefficient changes as the posterior evolves.

### 10. Optimal Bayesian Control

The central result of the paper. The optimal fraction to invest in the risky opportunity at time $t$ is:

**Bayesian Kelly Fraction (Continuous) ($f_t^*$):**

$$
f_t^* = \frac{1}{\sigma^2}\left(\frac{cX_t + \mu\sigma^2}{\sigma^2 + ct}\right) = \frac{\hat{Z}_t}{\sigma^2} \tag{51}
$$

This is the certainty equivalent of the non-Bayesian rule $f^* = \mu/\sigma^2$: the unknown $Z$ is replaced by $\hat{Z}_t$, the posterior mean from §8. The Separation Principle holds — estimation and control decouple under logarithmic utility.

### 11. Bayesian Value Function

Under the optimal control from §10, the expected log terminal wealth is:

**Bayesian Value ($F^B(x)$):**

$$
F^B(x) = \ln x + \frac{\mu^2}{2\sigma^2}T + \frac{c}{2\sigma^2}T - \frac{1}{2}\ln\left(1 + \frac{c}{\sigma^2}T\right) \tag{56}
$$

The three terms beyond $\ln x$: (i) the known-drift growth $\mu^2 T/(2\sigma^2)$; (ii) the random-drift contribution $cT/(2\sigma^2)$; (iii) the learning penalty $-\frac{1}{2}\ln(1 + cT/\sigma^2)$.

### 12. Financial Value of Learning

Comparing the value functions yields the paper's key economic results:

**Bayesian Gain over Naive Strategy:**

$$
F^B(x) - F(x) = \frac{c}{2\sigma^2}T - \frac{1}{2}\ln\left(1 + \frac{c}{\sigma^2}T\right) \tag{57}
$$

**Cost of Learning (Perfect Information Gap):**

$$
F^P(x) - F^B(x) = \frac{1}{2}\ln\left(1 + \frac{c}{\sigma^2}T\right) \tag{58}
$$

The gain from having a random drift to exploit is $O(T)$ (linear), while the cost of having to learn that drift is $O(\ln T)$ (logarithmic). In the long run, the Bayesian investor captures nearly all of the value that a perfectly informed investor would achieve.

### 13. Bayesian Portfolio Control

For the portfolio problem with a risk-free rate $r$, the Bayesian Merton rule is:

**Bayesian Merton Fraction ($f_t^*$):**

$$
f_t^* = \frac{1}{\sigma^2}\left(\frac{cX_t + \mu\sigma^2}{\sigma^2 + ct} - r\right) \tag{63}
$$

For a general prior $G$ on the drift $Z$:

$$
f_t^* = \frac{1}{\sigma^2}\left(E[Z \mid \mathcal{F}_t^X] - r\right) \tag{64}
$$

This is the cleanest statement of the paper's message: **replace the unknown drift by its posterior mean, then apply the standard Merton/Kelly rule**.

### 14. Power Utility Breakdown

For power utility $u(x) = x^\eta$ with $\eta \neq 0$, the Bayesian optimal control in discrete time is governed by a **nonlinear recursion**:

**Power Utility Bayesian Control ($f^*$):**

$$
f^*(x, S_k, k) = \frac{1}{\Delta} \cdot \frac{[\theta_k(S_k) g_{n-k}(k+1, S_k+1)]^{1/(1-\eta)} - [(1-\theta_k(S_k)) g_{n-k}(k+1, S_k-1)]^{1/(1-\eta)}}{[\theta_k(S_k) g_{n-k}(k+1, S_k+1)]^{1/(1-\eta)} + [(1-\theta_k(S_k)) g_{n-k}(k+1, S_k-1)]^{1/(1-\eta)}} \tag{70}
$$

with auxiliary recursion:

$$
g_n(r, v) = \left[\theta_r(v) g_{n-1}(r+1, v+1)^\eta + (1-\theta_r(v)) g_{n-1}(r+1, v-1)^\eta\right]^{1/\eta} \tag{71}
$$

The dependence on $g_n$ — which encodes future posterior evolution — means the optimal policy is **not** a certainty equivalent. The investor must account for the *volatility of the estimate*, not just its current value. The Separation Principle fails.

---

### **1. Problem:**

The paper addresses the problem of **optimal investment** when the fundamental parameter governing asset returns — the drift — is **unobserved**. The classical Kelly criterion and Merton's portfolio rule assume this parameter is known, yielding clean, myopic policies. In practice, investors estimate parameters from data, introducing a gap between the assumed and true growth-optimal strategies. The paper fills this gap by embedding the estimation problem *within* the control problem, asking: what is the optimal investment policy when the investor must simultaneously learn and invest?

### **2. Setup:**

- **Discrete time**: A repeated binary gambling model where the gambler bets fractions of wealth on a biased coin with *unknown* success probability $\theta \sim \text{Beta}(\alpha, \beta)$.
- **Continuous time**: A diffusion model $dX_t = Z \, dt + \sigma \, dW_t$ where the drift $Z \sim N(\mu, c)$ is unobserved, and wealth is controlled via $dV_t = f_t V_t \, dX_t$.
- The variance $\sigma^2$ is assumed **known**; only the drift is uncertain.
- **Logarithmic utility** is the primary objective; power utility $u(x) = x^\eta$ is analyzed as a contrast case.
- Prior distributions are **conjugate** (Beta in discrete time, Normal in continuous time) for analytical tractability.

### **3. Key Idea:**

Under logarithmic utility, the optimal Bayesian investment policy is the **certainty equivalent** of the classical Kelly/Merton rule: replace the unknown drift with its posterior mean and invest as if that were the true drift. This **Separation Principle** — estimation decouples from control — yields a policy that is both myopic and globally optimal. The cost of learning (relative to perfect information) grows only logarithmically in time, while the benefit of having a random drift to exploit grows linearly.

### **4. Assumptions:**

**Explicit:**
- **Logarithmic utility** for the central results; power utility treated separately.
- **Known variance** $\sigma^2$; only the drift is uncertain.
- **I.I.D. increments** conditional on the unknown parameter (the parameter is fixed, not time-varying).
- **Conjugate priors** (Beta-Binomial, Normal-Normal) for closed-form solutions.

**Implicit:**
- **Frictionless markets**: no transaction costs, taxes, or market impact.
- **Infinite divisibility** of wealth: the investor can allocate any fraction, including leverage.
- **Unconstrained betting**: the optimal fraction can exceed 1 (borrowing) or be negative (shorting).
- **Single risky asset**: no multi-asset portfolio considerations.

### **5. Limitation:**

- The assumption of **known variance** is rarely satisfied in practice — real markets exhibit stochastic volatility with a well-documented negative correlation to returns (the leverage effect).
- The **fixed unknown drift** assumption excludes non-stationary environments where the drift itself evolves. In such settings, the posterior never converges and the "cost of learning" does not vanish.
- **Conjugate prior** restriction limits the explicit results; non-conjugate priors require numerical methods.
- The optimality proof relies on **Martingale arguments and diffusion limits** rather than a direct HJB verification for the continuous-time case.
- The **certainty equivalence breaks down for all non-logarithmic utility functions**, meaning the paper's most elegant result applies only to one specific utility function.

### **6. Relevance & Open Questions:**

This paper provides the theoretical foundation for any adaptive portfolio strategy that uses posterior estimates as plug-in parameters. For research on **Kelly-based RL agents**, the certainty equivalence result justifies using posterior mean estimates directly in the Kelly formula — but *only* under logarithmic utility. For risk-sensitive or distributional objectives (as explored in [[PNOTES - Distributional Robust Kelly Strategy Optimal Strategy under Uncertainty in the Long-Run]]), the failure of certainty equivalence means the agent must explicitly model estimation uncertainty in its policy.

**Open questions:**
- How does the optimal policy change when the drift is a **stochastic process** (e.g., Ornstein-Uhlenbeck) rather than a fixed random variable?
- Can the logarithmic cost-of-learning result be extended to **multi-asset** portfolios with correlated unknown drifts?
- What is the interaction between **parameter uncertainty** and **model uncertainty** (e.g., unknown volatility, unknown distribution family)?
- Does the certainty equivalence principle survive when the investor faces **portfolio constraints** (e.g., no leverage, long-only)?

---

### Integration:

* **Problem:** Browne and Whitt's formulation of Bayesian Kelly investing under unknown drift is the continuous-time theoretical backbone for any adaptive investment strategy that uses posterior estimates. The certainty equivalence result for log-utility provides both a tractable policy rule (plug in the posterior mean) and a benchmark (the logarithmic cost of learning) against which more sophisticated approaches can be measured. For thesis work on Kelly-based portfolio optimization under uncertainty, this paper defines the idealized case — the "best possible" outcome under strong assumptions — that motivates the need for robust or distributional corrections when those assumptions fail.

* **Limitation:** The most binding limitation is the **known variance** assumption combined with the **fixed drift** assumption. Real financial markets exhibit both stochastic volatility and regime changes in drift, violating both. The paper's strongest result (certainty equivalence for log-utility) is robust to the prior specification but fragile to the utility function choice — any departure from logarithmic utility breaks the Separation Principle and requires the investor to solve a coupled filtering-control problem. This motivates the approaches in [[PNOTES - Optimal Betting Under Parameter Uncertainty - Improving the Kelly Criterion]] (frequentist shrinkage) and [[PNOTES - Modified Kelly criteria]] (Bayesian loss-function estimation) as practical corrections for the gap between log-utility theory and real-world risk preferences.
