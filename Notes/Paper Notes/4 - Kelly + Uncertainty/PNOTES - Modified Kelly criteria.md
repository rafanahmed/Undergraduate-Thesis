[[Modified Kelly Criteria.pdf]]

# **Abstract:**

The paper considers an extension of the **Kelly criterion** used in sports wagering. By recognizing that the probability $p$ of placing a correct wager is *unknown*, modified Kelly criteria are obtained that take the uncertainty into account. Estimators are proposed from a **decision-theoretic framework**. The resultant betting fractions can differ markedly based on the choice of **loss function**. In all cases studied, the modified Kelly fractions are smaller than the original Kelly fraction.

- The standard Kelly criterion is mathematically proven to maximize the exponential rate of bankroll growth (Breiman, 1961), yet experienced gamblers consistently report that the Kelly fraction is *too high* and leads to financial loss.
	- The resolution: the input $p$ used in determining the Kelly fraction is an unknown quantity. Gamblers are often overly optimistic about their systems, and the true $p$ is less than the specified $p$.
- Rather than ad-hoc approaches such as **"half-Kelly"** or **"quarter-Kelly"** (which have no theoretical underpinning), this paper provides a *systematic* statistical approach that accounts for the uncertainty in $p$.
	- The resulting modified fractions are derived, not imposed. They are natural consequences of treating the betting fraction as a statistical estimation problem.

- *Context*:
	- The paper sits at the intersection of two literatures: the **Kelly criterion** (information theory, portfolio optimization) and **Bayesian decision theory** (statistical estimation under loss functions). The synthesis is that the betting fraction $f$ is not a deterministic optimization output but a *statistical estimator* of the true optimal $k(p)$, and the quality of this estimator should be evaluated through a loss function that reflects the Kelly desiderata — namely, expected log growth. This reframing is the paper's core intellectual contribution.
	- The distinction between the Kelly criterion as a *mathematical proof* and its *statistical implementation* is critical. The proof assumes $p$ is known; the implementation requires $p$ to be estimated from data. The gap between these two regimes is where ad-hoc strategies like half-Kelly have flourished — and where this paper provides a principled alternative.
	- This paper is a precursor to the distributional robust approach in [[PNOTES - Distributional Robust Kelly Strategy Optimal Strategy under Uncertainty in the Long-Run]], which addresses the same fundamental problem (Kelly under uncertainty) but uses a *distributional robust* framework rather than a Bayesian one. Where Chu et al. place a Beta prior on $p$, Sun et al. define an uncertainty set $\Xi$ over the full distribution and optimize against the worst case.

---

# 1. Introduction

The paper opens by establishing the historical tension between the Kelly criterion's theoretical optimality and its practical reputation.

## The Kelly Criterion and Its Discontents

- **Kelly (1956)** provides the optimal fraction of a bankroll that should be wagered, given a specified winning probability $p$ and European odds $\theta$.
- The Kelly criterion is optimal from several points of view:
	- It maximizes the **exponential rate of growth** of the bankroll.
	- It provides the **minimal expected time** to reach an assigned target balance (Breiman, 1961).
- Despite these proofs, gamblers frequently experience losses when applying Kelly. **Samuelson (1979)** provided influential criticism, and practitioners have long advised fractional Kelly approaches.

- *Context*:
	- Samuelson's objection was primarily philosophical — he argued against maximizing the mean of the logarithm of wealth on the grounds that it does not account for the gambler's risk preferences. The Kelly criterion implicitly assumes a log-utility function, which may not match an individual's actual risk aversion. However, the *practical* failures identified in this paper are more concrete: they arise from estimation error, not from utility mismatch.

## The Statistical Explanation

The paper's central insight: the **simple but often overlooked explanation** is that the input $p$ is unknown. Gamblers estimate $p$ from historical data, and because they are often overly optimistic, the true $p$ is frequently less than the estimated $\hat{p}$.

- The paper reframes the determination of $f$ from a deterministic optimization to a **statistical estimation problem** where $p$ is an unknown parameter.
- This produces "modified Kelly criteria" that are systematically justified, in contrast to ad-hoc fractional approaches.

## Relationship to Baker and McHale (2013)

The paper explicitly acknowledges **Baker and McHale (2013)** as a key predecessor:

- Baker and McHale (2013) also use a decision-theoretic framework and recognize the unknown nature of $p$.
- However, they impose an *assumed form* on the betting fraction: $ks^*(q)$, where $s^*(p)$ is the Kelly fraction, $q$ is an estimator of $p$, and $0 < k < 1$ is a shrinkage factor.
- Chu et al. make **no such assumption** on the betting fraction. The Bayes estimators they derive are optimal *without* imposing a parametric relationship between $f$ and $k(p)$.
- Baker and McHale (2016) extend their earlier work and argue for frequentist risk minimization. A major difference is that Chu et al. use **prior distributions** (Bayesian), whereas Baker and McHale use a frequentist approach.

- *Context*:
	- The distinction between "imposing shrinkage" and "deriving shrinkage" is methodologically important. Baker and McHale assume the betting fraction takes the form $f = k \cdot k(p)$ and then optimize over $k$. This is a one-dimensional search over a scalar multiplier. Chu et al. let the Bayes estimator take whatever functional form minimizes expected posterior loss — the resulting $f_0$ happens to resemble half-Kelly in many cases, but this is a *consequence* of the optimization, not an *assumption*.

---

# 2. Review of Sports Gambling

This section establishes the formal vocabulary and derives the Kelly criterion from first principles.

## Odds and Profitability

- **American odds** are the standard in U.S. sports betting. Negative odds (e.g., $-110$) indicate the wager required to win \$100. Positive odds (e.g., $+110$) indicate the profit on a \$100 wager.
- **European (decimal) odds** $\theta$ provide a simpler representation: a winning bet of \$x returns $x\theta$.
	- Example: $\theta = 1.909$ corresponds to American odds $-110$. A winning \$110 bet returns $\$110 \times 1.909 = \$210$.
- **Vigorish** is the bookmaker's built-in profit margin, created by the discrepancy between payout odds and true probabilities. At $-110$ on both sides, the bookie collects \$220 in wagers and pays out \$210, guaranteeing a \$10 profit regardless of outcome.

### Profitability Condition

A gambling system with European odds $\theta$ and winning probability $p$ is profitable if:

$$
p > \frac{1}{\theta} \tag{2}
$$

- At $\theta = 1.909$ (American $-110$), the system needs $p > 52.4\%$ to be profitable.

- *Context*:
	- The profitability condition (2) is deceptively simple. While $52.4\%$ seems easily achievable, bookmakers have survived for centuries precisely because consistently beating this threshold is extremely difficult. The vigorish creates an asymmetry: the gambler must be *better than break-even* just to avoid losing. The entire remainder of the paper is only applicable under the assumption that the gambler possesses a system satisfying (2). When (2) does not hold, gambling is a losing proposition regardless of money management strategy.

## Derivation of the Kelly Criterion

The paper provides the full proof of the Kelly criterion.

- Begin with bankroll $B_0$. Bet a fraction $f$ of the bankroll. Let $w = 1$ if the wager is won, $w = 0$ otherwise.
- The subsequent bankroll:

$$
B_1(f) = (1 - f + \theta f)^w (1 - f)^{1-w} B_0
$$

- The Kelly approach maximizes the **expected log growth**:

$$
E[\log(B_1(f)/B_0)] = p \log(1 - f + \theta f) + (1 - p) \log(1 - f) \tag{4}
$$

- Differentiating (4) with respect to $f$ and setting equal to zero yields the **Kelly criterion**:

$$
k(p) = \begin{cases} \frac{p\theta - 1}{\theta - 1} & p > 1/\theta \\ 0 & p \leq 1/\theta \end{cases} \tag{3}
$$

- *Context*:
	- The Kelly fraction $k(p)$ is the unique maximizer of the concave function (4). The expected log growth (4) has two terms: the first captures the log return from winning (weighted by $p$), and the second captures the log return from losing (weighted by $1 - p$). Because $\log(1 - f)$ is strictly decreasing in $f$, betting too much erodes the bankroll faster on losses than it grows it on wins — this is the fundamental source of the Kelly criterion's moderation. The criterion balances the upside of larger bets against the compounding cost of larger losses.
	- The expected log growth (4) is the objective that the loss function $l_0$ will later be built upon. See **MATHEMATICS §1–2** for the derivation chain.

---

# 3. Development of Modified Kelly Criteria

This is the paper's central methodological section. The determination of $f$ is reframed as a **statistical estimation problem** where $p$ is an unknown parameter.

## The Estimation Framework

- The data arises from historical wagering: a gambler proposes a system and uses past seasons to estimate $p$.
	- Example: betting on road teams when the home team plays its first game back from a road trip. The winning proportion $\hat{p}$ from past data estimates $p$.
- The **statistical model**:

$$
X \equiv \text{number of winning historical matches} \sim \text{Binomial}(n, p) \tag{5}
$$

- The problem reduces to estimating the unknown $k(p)$ by an estimator $f = f(x)$.

### The Natural Loss Function

To assess the quality of an estimator, a **loss function** is required. The paper introduces a loss function that is *natural* in the Kelly context — the ratio of optimal expected log growth to the growth under an alternative fraction $f$:

$$
l_0(f, p) = p \log\!\left(\frac{1 - k(p) + \theta k(p)}{1 - f + \theta f}\right) + (1 - p) \log\!\left(\frac{1 - k(p)}{1 - f}\right) \tag{6}
$$

- *Context*:
	- The loss function $l_0$ is the **pivot** of the entire paper. It measures the cost of using $f$ instead of the true optimal $k(p)$, expressed in units of expected log growth. This is not an arbitrary choice — it is the unique loss function that directly quantifies the forgone growth rate. Other loss functions (absolute error, squared error) measure the *distance* between $f$ and $k(p)$, but $l_0$ measures the *economic consequence* of that distance. Because the log growth function is concave, $l_0(f, p) \geq 0$ with equality only when $f = k(p)$.
	- The loss function is asymmetric around $k(p)$: overestimating $f$ (betting too much) incurs greater loss than underestimating $f$ (betting too little) because the expected posterior loss $G(f)$ increases more rapidly for $f > f_0$ than for $f < f_0$ (see Figure 1 in the paper). This asymmetry is what drives the modified Kelly fractions to be *conservative* — the penalty for overbetting exceeds the penalty for underbetting.

## 3.1 Minimax Estimators

The paper first considers the **minimax approach**, which minimizes the worst-case risk.

- The **risk function** of an estimator $f$ is:

$$
R_f(p) = \sum_{x=0}^{n} l_0(f, p) \, p(x \mid p) \tag{7}
$$

where $p(x \mid p)$ is the binomial probability mass function from (5).

- The **minimax estimator** minimizes the supremum of the risk:

$$
S(f) = \sup_p R_f(p)
$$

- The minimax approach is **overly conservative**: the minimizer of $S(f)$ is $f(x) = 0$ for all $x$. The reasoning:
	- $l_0(f, p) \geq 0$ for all $f$ and $p$.
	- Therefore, $R_f(p) = 0$ is achieved by $f = 0$ — never betting.
	- Mathematically, the supremum of $R_f(p)$ is attained at $p = 1/\theta$ (the boundary of profitability), where the worst-case term simplifies and is minimized by $f = 0$.

- *Context*:
	- The minimax result is instructive rather than useful. It demonstrates that a strategy optimized against the *absolute worst case* degenerates to zero action — because the worst case is always that the system is not profitable, and any bet incurs non-zero loss. This motivates the Bayesian approach: by introducing prior information (even weak prior information), the gambler avoids the pathological conservatism of minimax while still accounting for uncertainty.
	- The connection to the **Distributional Robust Kelly Problem (DRKP)** in [[PNOTES - Distributional Robust Kelly Strategy Optimal Strategy under Uncertainty in the Long-Run]] is notable: the DRKP can be seen as a *bounded* minimax approach where the uncertainty set $\Xi$ prevents the adversary from placing all probability at the worst-case $p = 1/\theta$. The DRKP achieves positive betting fractions precisely because the uncertainty set constrains how adversarial the distribution can be.

## 3.2 Bayes Estimators

The paper introduces a **prior density function** $\pi(p)$ to describe uncertainty in $p$.

### Bayes Risk

With the prior, the **Bayes risk** is:

$$
r_f = \int_0^1 R_f(p) \, \pi(p) \, dp \tag{9}
$$

Bayes estimators minimize the Bayes risk (9), which is equivalent to minimizing the **expected posterior loss**:

$$
G(f) = \int_0^1 l_0(f, p) \, \pi(p \mid x) \, dp \tag{10}
$$

### Beta Prior and Posterior

The **Beta prior** $p \sim \text{Beta}(a, b)$ is a natural choice:

- Defined on $p \in (0, 1)$, matching the parameter space.
- For $a > 1$ and $b > 1$, the prior density is concave — unimodal with a peak at a most-likely value $p_0$.
- The gambler can specify $(a, b)$ through mean and variance: $E(p) = a/(a + b)$, $\text{Var}(p) = ab/((a + b)^2(a + b + 1))$.

The **conjugacy** of the Beta prior with the Binomial likelihood yields the posterior:

$$
p \mid x \sim \text{Beta}(x + a, \, n - x + b) \tag{11}
$$

with density:

$$
\pi(p \mid x) = \frac{\Gamma(n + a + b)}{\Gamma(x + a)\,\Gamma(n - x + b)} \, p^{x+a-1}(1 - p)^{n - x + b - 1} \tag{12}
$$

- *Context*:
	- The Beta-Binomial conjugacy is the mathematical engine that makes the entire framework tractable. Conjugacy means the posterior is in the same family as the prior, so updating beliefs with data is a simple parameter update: $a \to x + a$, $b \to (n - x) + b$. The posterior mean $\hat{p} = (x + a)/(n + a + b)$ is a *shrinkage estimator* — it pulls the sample proportion $x/n$ toward the prior mean $a/(a + b)$, with the degree of shrinkage controlled by the ratio of prior "pseudo-counts" $(a + b)$ to total data $(n + a + b)$. As $n \to \infty$, the posterior concentrates at the true $p$, and the shrinkage vanishes.

### Derivation of the Modified Kelly Criterion $f_0$

Based on (3), (6), (10), and (12), the Bayes estimator minimizes:

$$
G(f) = \int_{1/\theta}^{1} \left[p \log\!\left(\frac{1 - k(p) + \theta k(p)}{1 - f + \theta f}\right) + (1 - p) \log\!\left(\frac{1 - k(p)}{1 - f}\right)\right] \pi(p \mid x) \, dp + \int_0^{1/\theta} \left[p \log\!\left(\frac{1}{1 - f + \theta f}\right) + (1 - p) \log\!\left(\frac{1}{1 - f}\right)\right] \pi(p \mid x) \, dp \tag{13}
$$

The key simplification: since $f$ does not appear in the numerators of the logarithms in (13), minimizing (13) is equivalent to maximizing:

$$
Q(f) = \hat{p} \log(1 - f + \theta f) + (1 - \hat{p}) \log(1 - f) \tag{14}
$$

where $\hat{p} = (x + a)/(n + a + b)$ is the **posterior mean** of $p$.

- *Context*:
	- The transition from (13) to (14) is the paper's most elegant result. The complicated double integral collapses to a function of a single sufficient statistic — the posterior mean $\hat{p}$. This happens because the terms containing $f$ in (13) factor out of the integral and recombine into precisely the expected log growth function (4), evaluated at the posterior mean. The integral over $p$ of $p \cdot \pi(p \mid x)$ is simply $E[p \mid x] = \hat{p}$.
	- The paper credits an anonymous reviewer for recognizing this simplification, which led to the analytic expression (15).

Maximizing $Q(f)$ is identical to the original Kelly formulation (4) with $\hat{p}$ replacing $p$. Therefore, the **Modified Kelly criterion**:

$$
f_0 = \begin{cases} \frac{\hat{p}\theta - 1}{\theta - 1} & \hat{p} > 1/\theta \\ 0 & \hat{p} \leq 1/\theta \end{cases} \tag{15}
$$

where $\hat{p} = (x + a)/(n + a + b)$.

- *Context*:
	- The Modified Kelly criterion $f_0$ has the same functional form as the original Kelly criterion $k(p)$, but with the posterior mean $\hat{p}$ replacing the unknown $p$. Since $\hat{p}$ is a shrinkage estimator that pulls $x/n$ toward the prior mean (typically $0.5$), $f_0 < k(\hat{p}_{\text{MLE}})$ whenever the prior is conservative. This provides a *theoretical justification* for fractional Kelly wagering. In many practical cases, $f_0 \approx k(\hat{p})/2$ — half-Kelly — but this is a derived result, not an assumption.
	- The convergence property is important: as $n \to \infty$, the posterior mean $\hat{p} \to x/n \to p$, and therefore $f_0 \to k(p)$. The modified criterion converges to the original Kelly criterion as the gambler accumulates more data. The rate of convergence is governed by the prior strength $(a + b)$ relative to the sample size $n$.

---

# 4. Alternative Loss Functions

The paper investigates how the choice of loss function affects the resulting betting fraction. Three alternative loss functions are considered, all using the same **Beta**$(a, b)$ prior.

## Absolute Error Loss ($l_1$)

$$
l_1(f, p) = |f - k(p)|
$$

The Bayes estimator under absolute error loss is the **posterior median** of $k(p)$:

$$
f_1 = \begin{cases} \frac{\tilde{p}\theta - 1}{\theta - 1} & \tilde{p} > 1/\theta \\ 0 & \tilde{p} \leq 1/\theta \end{cases} \tag{16}
$$

where $\tilde{p}$ is the **posterior median** of $p$ from (11).

- *Context*:
	- For large, roughly symmetric posteriors (when $x + a$ and $n - x + b$ are large and comparable), the posterior mean $\hat{p}$ and median $\tilde{p}$ are nearly identical, so $f_0 \approx f_1$. The difference between the two becomes noticeable only when the posterior is markedly skewed — which happens when $x + a$ and $n - x + b$ are small or very different.

## Squared Error Loss ($l_2$)

$$
l_2(f, p) = (f - k(p))^2
$$

The Bayes estimator under squared error loss is the **posterior mean** of $k(p)$:

$$
f_2 = \int_{1/\theta}^{1} \left(\frac{p\theta - 1}{\theta - 1}\right) \pi(p \mid x) \, dp \tag{17}
$$

- *Context*:
	- The estimator $f_2$ is *fundamentally different* from $k(p)$, $f_0$, and $f_1$. With those three, there are always scenarios where the estimator is zero (when the posterior mean or median falls below $1/\theta$). With $f_2$, the integral in (17) is strictly positive whenever the posterior assigns *any* probability to $p > 1/\theta$ — which it almost always does. Therefore, $f_2$ never recommends zero betting. This is a consequence of the squared error loss penalizing deviations symmetrically and quadratically, without the "floor at zero" that the Kelly-natural loss and absolute error loss induce.

## General Asymmetric Loss ($l_3$)

The paper introduces a flexible loss function that nests $l_1$ and $l_2$ as special cases and allows **asymmetric penalization** of over- and under-estimation:

$$
l_3(f, p) = (c_1 \mathbb{I}_{f > k(p)} + c_2) |f - k(p)|^k \tag{18}
$$

where $c_1 > 0$ and $c_2 > 0$ control the asymmetric penalty, and $1 < k < 2$ interpolates between absolute and squared error. The indicator $\mathbb{I}_{f > k(p)}$ activates the additional penalty $c_1$ when the estimator *overestimates* the Kelly fraction.

- The paper considers two parameterizations:
	- **(19):** $c_1 = 1, c_2 = 1, k = 1.5$ — overestimation penalty is double underestimation.
	- **(20):** $c_1 = 1, c_2 = 2, k = 1.5$ — overestimation penalty is 1.5 times underestimation.
- The Bayes estimator $f_3$ is obtained by minimizing:

$$
G(f) = \int_0^1 l_3(f, p) \, \pi(p \mid x) \, dp \tag{21}
$$

- No analytic solution exists. The paper uses **Simpson's rule** for quadrature and a **brute-force grid search** over $f \in [0, 1]$ in steps of $0.001$.

- *Context*:
	- The loss function $l_3$ is motivated by the common complaint that the Kelly fraction is too large. By setting $c_1 > 0$, the loss function *penalizes overestimation more heavily than underestimation*, which biases the resulting estimator toward smaller fractions. This is a principled way to encode the asymmetric real-world costs: overbetting risks ruin, while underbetting merely forgoes some growth. The parameterization (19) is described as "extreme" because the overestimation penalty is double; (20) is more moderate at 1.5 times.

---

# 5. Examples

The paper presents five examples that validate the modified criteria and explore sensitivity to odds, sample size, prior strength, and real-world data.

## Example 5.1: Hypothetical NBA ($\theta = 1.952$)

- European odds $\theta = 1.952$ (American $-105$), $x = 100$, $n = 180$, so $\hat{p}_{\text{MLE}} = 0.556$.
- Standard Kelly: $k(\hat{p}) = 0.089$ — bet 9% of bankroll per wager.
- **Prior specification:** $a = b = 50$, giving $E(p) = 0.5$, $\text{SD}(p) \approx 0.05$. Roughly 95% of prior probability falls in $(0.4, 0.6)$.
	- This prior reflects the efficient markets intuition: gambling systems picking winners at rates above 60% are extremely rare.
- **Modified Kelly fractions:** $f_0 = 0.048$, $f_1 = 0.048$, $f_2 = 0.056$, $f_{3a} = 0.034$, $f_{3b} = 0.041$.
- All are smaller than $k(\hat{p}) = 0.089$. In particular, $f_0 \approx k(\hat{p})/2$ — **half-Kelly has a theoretical rationale**.
- Consistent ordering: $f_{3a} < f_{3b} < f_0 \approx f_1 < f_2$.

- *Context*:
	- The ordering $f_{3a} < f_{3b} < f_0 \approx f_1 < f_2$ is not coincidental — it reflects the relative conservatism of the loss functions. The general loss $l_3$ with strong asymmetric penalty produces the smallest fractions; the natural loss $l_0$ and absolute error $l_1$ produce moderate fractions; the squared error $l_2$ produces the largest because it never assigns zero and penalizes deviations symmetrically without the Kelly-specific economic weighting.

## Example 5.2: Increased Vigorish ($\theta = 1.909$)

- Same setup but with $\theta = 1.909$ (American $-110$), the industry standard.
- Kelly: $k(\hat{p}) = 0.067$. Modified: $f_0 = 0.025$, $f_1 = 0.025$, $f_2 = 0.039$, $f_{3a} = 0.018$, $f_{3b} = 0.024$.
- All fractions are *smaller* than in Example 5.1 — higher vigorish (lower $\theta$) compresses the margin, requiring greater conservatism.
- **Figure 1** shows $G(f)$ is convex, with the loss increasing more rapidly for $f > f_0$ than for $f < f_0$, confirming the asymmetric cost of overbetting.

## Example 5.3: Doubled Sample Size

- $\theta = 1.909$, $a = b = 50$, but now $x = 200$, $n = 360$ (same winning proportion, doubled data).
- Kelly remains $k(\hat{p}) = 0.067$.
- Modified fractions increase: $f_0 = 0.041$, $f_1 = 0.041$, $f_2 = 0.047$, $f_{3a} = 0.030$, $f_{3b} = 0.036$.
- The modified fractions move *toward* the original Kelly as $n$ increases. **As $n \to \infty$, modified Kelly converges to original Kelly.**

- *Context*:
	- This convergence is the key feature distinguishing the Bayesian approach from ad-hoc shrinkage. Half-Kelly always bets half the Kelly fraction regardless of how much data supports the system. The modified Kelly criterion, by contrast, becomes progressively less conservative as more data arrives — the prior's influence is "washed out" by the accumulating evidence. This is a direct consequence of the posterior mean $\hat{p} = (x + a)/(n + a + b) \to x/n$ as $n \to \infty$.

## Example 5.4: Prior Sensitivity and Simulation

- $\theta = 1.909$, $x = 100$, $n = 180$, but now $a = b = 100$ (stronger prior, more doubtful).
	- $\text{SD}(p) \approx 0.035$, so 95% of prior probability is in $(0.43, 0.57)$.
- Kelly remains $k(\hat{p}) = 0.067$. Modified: $f_0 = 0.005$, $f_1 = 0.005$, $f_2 = 0.024$, $f_{3a} = 0.008$, $f_{3b} = 0.012$.
- A stronger prior dramatically reduces the betting fractions.

### Simulation Results

- **Setup:** True $p = (p_{\text{orig}} + p_0)/2 = 0.541$ (midpoint between the probabilities implied by original Kelly and modified Kelly). Initial bankroll $B_0 = \$1000$, 200 consecutive wagers.
- **Five-run visualization (Figure 2):** Original Kelly shows greater variability — higher peaks and deeper troughs. In 3 of 5 runs, modified Kelly outperforms by the end.
- **1,000-run density (Figure 3):**
	- Modified Kelly $f_0$ was profitable in **65%** of runs.
	- Original Kelly was profitable in only **53%** of runs.
- **Variable season length (Figure 4):** Over seasons from 100 to 5,000 wagers, modified Kelly achieves a higher percentage of profitable outcomes, a higher percentage of doubled bankrolls, and a lower percentage of halved bankrolls (especially at longer horizons).

- *Context*:
	- The simulation results in Example 5.4 are the paper's strongest empirical evidence. The key finding is not that modified Kelly produces higher *expected* returns — it does not (original Kelly has higher expected log growth when $p$ is known). Rather, modified Kelly produces higher *probability of profit* because it avoids the catastrophic overbetting that occurs when $\hat{p}$ overestimates $p$. The 65% vs. 53% profitability gap is substantial and arises directly from the conservative shrinkage of the Bayesian estimator.
	- The asymptotic convergence visible in Figure 4 is theoretically guaranteed: as the season length grows, both the original and modified Kelly strategies converge to optimal behavior (because more data pins down $p$ more precisely). But at realistic season lengths (100–500 wagers), the modified approach has a significant safety advantage.

## Example 5.5: Real-World NBA Playoffs

- Based on **Professor MJ's "bounce-back" system**: bet on a playoff team to cover the spread after losing badly (margin exceeding 12.5 points). Historical record: $n = 484$, $x = 271$, $\hat{p} = 0.560$.
- $\theta = 1.909$, $a = b = 50$.
- Kelly: $k(\hat{p}) = 0.076$. Modified: $f_0 = 0.054$, $f_1 = 0.054$, $f_2 = 0.056$, $f_{3a} = 0.042$, $f_{3b} = 0.047$.
	- $f_0$ is roughly **3/4 Kelly** — closer to the original because $n = 484$ is relatively large, so the posterior mean is closer to the MLE.

### Out-of-Sample Performance

- 14 qualifying games in the remainder of the 2017 playoffs. Results: 4 wins, 8 losses, 2 pushes (sequence: WLLWPLLWLLWLLP).
- Starting with $B_0 = \$1000$:
	- **Original Kelly:** final balance $\$694.10$.
	- **Modified Kelly $f_0$:** final balance $\$776.91$.
- The cautious modified approach mitigated losses during a period of severe system underperformance.

- *Context*:
	- The out-of-sample result is small ($n = 14$) and should be interpreted cautiously, but it illustrates the protective mechanism: when the system underperforms its historical rate (4/12 = 33% wins, far below the historical 56%), betting smaller fractions limits the damage. The modified Kelly approach lost 22.3% of the bankroll versus 30.6% for original Kelly — a meaningful difference when compounded over many such adverse stretches.

---

# 6. Discussion

The paper synthesizes the results and identifies limitations and extensions.

- Modified Kelly criteria require specifying a **Beta prior** on $p$. When priors are conservative (placing less probability on $p > 1/\theta$), the modified criteria are consistently smaller than original Kelly.
- The choice of **loss function** has a significant impact on the betting fraction, though all loss functions studied produce fractions smaller than original Kelly.
- **Variable $p$ wagering:** For gamblers without a fixed system (where $p$ varies across wagers), the framework faces a stumbling block in specifying $x$, $n$, $a$, and $b$. The paper suggests $x = n = 0$ as a default (using only the prior), with $(a, b)$ specified via constraint argumentation.
- **Future work:** Modified Kelly fractions for **simultaneous wagers** (where capital is allocated across multiple concurrent events) remain an open problem.

- *Context*:
	- The limitation regarding variable $p$ is practically important. In real sports betting (and especially in financial portfolio management), the winning probability is not constant across opportunities. Each trade or wager may have a different estimated edge. The paper's framework assumes a single, fixed $p$ estimated from a single historical dataset. Extending to variable $p$ would require either (a) a hierarchical Bayesian model where each wager's $p_i$ is drawn from a population distribution, or (b) a regression-based approach where $p$ depends on observable features.

---

# Appendix: R Implementation

The paper provides functional **R code** for computing the Bayes estimator $f_3$ under the general loss function $l_3(f, p)$ from (18). The implementation:

- Uses the `cubature` package for numerical integration (Simpson's rule via `adaptIntegrate`).
- Handles the discontinuity in $l_3$ at $p = (f(\theta - 1) + 1)/\theta$ by splitting the integral into two parts.
- Performs brute-force optimization over $f \in [0, 1]$ in steps of $0.001$.

- *Context*:
	- The step size of $0.001$ means the maximum discretization error in $f$ is $\pm 0.0005$. For betting fractions in the range $0.01$–$0.10$, this represents a relative error of 0.5%–5%. While sufficient for the examples in the paper, higher-precision applications (e.g., high-frequency automated wagering) would require finer grids or gradient-based optimization.

---

# MATHEMATICS

The mathematical framework of this paper proceeds in three phases. First, the Kelly criterion is derived from expected log growth maximization. Second, the estimation framework introduces a loss function that reframes the betting fraction as a statistical estimator, and the Bayesian machinery (Beta prior, conjugate posterior) yields an analytic Bayes estimator. Third, alternative loss functions produce alternative estimators. The derivation chain follows: odds model → expected log growth → Kelly criterion → natural loss function → decision-theoretic risk → Bayes estimation with Beta prior → modified Kelly fractions.

### 1. Profitability Condition and European Odds

A gambling system with European odds $\theta$ returns $x\theta$ on a winning wager of $x$. With winning probability $p$, expected profit per wager of $\$x$ is:

$$
(-x)(1 - p) + (x\theta - x)p = x(p\theta - 1)
$$

This is positive if and only if:

**Profitability Condition:**

$$
p > \frac{1}{\theta} \tag{2}
$$

For $\theta = 1.909$ (American $-110$), the threshold is $p > 0.524$.

### 2. Expected Log Growth and the Kelly Criterion

Starting with bankroll $B_0$ and betting fraction $f$, the bankroll after one wager is:

$$
B_1(f) = (1 - f + \theta f)^w (1 - f)^{1-w} B_0
$$

where $w \in \{0, 1\}$ is the win indicator. The **expected log growth** is:

$$
E[\log(B_1/B_0)] = p \log(1 - f + \theta f) + (1 - p)\log(1 - f) \tag{4}
$$

Differentiation with respect to $f$:

$$
\frac{d}{df} E[\log(B_1/B_0)] = \frac{p(\theta - 1)}{1 - f + \theta f} - \frac{1 - p}{1 - f}
$$

Setting equal to zero and solving:

$$
p(\theta - 1)(1 - f) = (1 - p)(1 - f + \theta f)
$$

$$
p\theta - p - p\theta f + pf = 1 - p - f + \theta f - p + p f - p\theta f + p\theta f
$$

After simplification:

**Kelly Criterion ($k(p)$):**

$$
k(p) = \begin{cases} \frac{p\theta - 1}{\theta - 1} & p > 1/\theta \\ 0 & p \leq 1/\theta \end{cases} \tag{3}
$$

The second derivative is negative for $f \in (0, 1)$, confirming $k(p)$ is a maximum. The Kelly fraction increases linearly in $p$ for $p > 1/\theta$, with slope $\theta/(\theta - 1)$.

### 3. The Natural Loss Function

The loss incurred by using fraction $f$ instead of the true optimal $k(p)$ is defined as the difference in expected log growth:

**Natural Loss ($l_0$):**

$$
l_0(f, p) = p \log\!\left(\frac{1 - k(p) + \theta k(p)}{1 - f + \theta f}\right) + (1 - p) \log\!\left(\frac{1 - k(p)}{1 - f}\right) \tag{6}
$$

This is the expected log growth at $k(p)$ minus the expected log growth at $f$. Since $k(p)$ uniquely maximizes (4), $l_0(f, p) \geq 0$ with equality iff $f = k(p)$.

### 4. Risk Function, Bayes Risk, and Expected Posterior Loss

The **risk function** averages the loss over the sampling distribution:

$$
R_f(p) = \sum_{x=0}^{n} l_0(f, p) \binom{n}{x} p^x (1 - p)^{n-x} \tag{7}
$$

The **Bayes risk** averages risk over the prior:

$$
r_f = \int_0^1 R_f(p) \, \pi(p) \, dp \tag{9}
$$

The Bayes estimator minimizes the **expected posterior loss** (equivalent to minimizing Bayes risk):

$$
G(f) = \int_0^1 l_0(f, p) \, \pi(p \mid x) \, dp \tag{10}
$$

### 5. Beta Prior, Conjugate Posterior, and the Posterior Mean

The prior is $p \sim \text{Beta}(a, b)$. Combined with the binomial likelihood $X \sim \text{Binomial}(n, p)$, the posterior is:

**Beta Posterior:**

$$
p \mid x \sim \text{Beta}(x + a, \, n - x + b) \tag{11}
$$

The posterior mean is:

$$
\hat{p} = \frac{x + a}{n + a + b} \tag{P}
$$

This is a weighted average of the sample proportion $x/n$ and the prior mean $a/(a + b)$, with weights proportional to $n$ and $(a + b)$, respectively.

### 6. Derivation of the Modified Kelly Criterion $f_0$

Substituting (6) and (12) into (10) and splitting the integral at $p = 1/\theta$:

$$
G(f) = \int_{1/\theta}^{1}\left[p \log\!\left(\frac{1 - k(p) + \theta k(p)}{1 - f + \theta f}\right) + (1 - p)\log\!\left(\frac{1 - k(p)}{1 - f}\right)\right]\pi(p \mid x)\,dp + \int_0^{1/\theta}\left[p \log\!\left(\frac{1}{1 - f + \theta f}\right) + (1 - p)\log\!\left(\frac{1}{1 - f}\right)\right]\pi(p \mid x)\,dp \tag{13}
$$

The terms involving $f$ appear only in the denominators of the logarithms. Therefore, minimizing $G(f)$ is equivalent to maximizing:

$$
Q(f) = \int_0^1 \left[p \log(1 - f + \theta f) + (1 - p)\log(1 - f)\right] \pi(p \mid x)\,dp = \hat{p}\log(1 - f + \theta f) + (1 - \hat{p})\log(1 - f) \tag{14}
$$

where the last equality uses $\int_0^1 p \, \pi(p \mid x)\,dp = \hat{p}$. Since $Q(f)$ is identical to (4) with $p$ replaced by $\hat{p}$, the maximizer is:

**Modified Kelly Criterion ($f_0$):**

$$
f_0 = \begin{cases} \frac{\hat{p}\theta - 1}{\theta - 1} & \hat{p} > 1/\theta \\ 0 & \hat{p} \leq 1/\theta \end{cases} \tag{15}
$$

where $\hat{p} = (x + a)/(n + a + b)$.

### 7. Alternative Bayes Estimators

**Absolute Error Loss ($f_1$):** The Bayes estimator under $l_1(f, p) = |f - k(p)|$ is the posterior median of $k(p)$:

$$
f_1 = \begin{cases} \frac{\tilde{p}\theta - 1}{\theta - 1} & \tilde{p} > 1/\theta \\ 0 & \tilde{p} \leq 1/\theta \end{cases} \tag{16}
$$

where $\tilde{p}$ is the posterior median of $p \mid x \sim \text{Beta}(x + a, n - x + b)$.

**Squared Error Loss ($f_2$):** The Bayes estimator under $l_2(f, p) = (f - k(p))^2$ is the posterior mean of $k(p)$:

$$
f_2 = \int_{1/\theta}^{1} \frac{p\theta - 1}{\theta - 1} \, \pi(p \mid x) \, dp \tag{17}
$$

This integral is always positive when the posterior assigns any probability to $p > 1/\theta$, so $f_2$ never equals zero.

**General Asymmetric Loss ($f_3$):** Under $l_3(f, p) = (c_1 \mathbb{I}_{f > k(p)} + c_2)|f - k(p)|^k$, the Bayes estimator minimizes:

$$
G(f) = \int_0^1 l_3(f, p) \, \pi(p \mid x) \, dp \tag{21}
$$

No closed form exists. The paper computes $G(f)$ numerically by splitting the integral at $p = (f(\theta - 1) + 1)/\theta$ (the discontinuity point of the indicator) and evaluating via adaptive quadrature. The minimum is found by brute-force search over $f \in \{0, 0.001, 0.002, \dots, 1\}$.

---

### **1. Problem:**

The standard Kelly criterion assumes that the winning probability $p$ is *known*, but in practice $p$ must be estimated from finite historical data. This estimation error causes the Kelly fraction to be systematically too large, leading to excessive volatility and frequent losses. The paper addresses the gap between the Kelly criterion's mathematical optimality (under known $p$) and its practical fragility (under estimated $\hat{p}$), providing a principled Bayesian alternative to ad-hoc fractional Kelly rules.

### **2. Setup:**

The model assumes a **Binomial** data-generating process: $X \sim \text{Binomial}(n, p)$, where $X$ counts winning wagers in $n$ historical trials. European odds $\theta$ are fixed and known. The parameter space is $p \in (0, 1)$ with the profitability condition $p > 1/\theta$. A **Beta**$(a, b)$ prior is placed on $p$, yielding a conjugate **Beta**$(x + a, n - x + b)$ posterior. The betting fraction $f \in [0, 1]$ is treated as a point estimator of the unknown $k(p)$, evaluated under various loss functions ($l_0, l_1, l_2, l_3$) within a **decision-theoretic framework**.

### **3. Key Idea:**

The paper reframes the determination of the betting fraction from a deterministic optimization (maximize log growth given known $p$) to a **statistical estimation problem** (estimate $k(p)$ under parameter uncertainty using a loss function). Using a **natural loss function** $l_0$ that measures forgone expected log growth and a **Beta prior** for $p$, the Bayes estimator takes the same functional form as the Kelly criterion but with the posterior mean $\hat{p} = (x + a)/(n + a + b)$ replacing $p$. The resulting "shrinkage" toward the prior mean provides a theoretically grounded, data-adaptive replacement for ad-hoc fractional Kelly strategies.

### **4. Assumptions:**

**Explicit:**
- Wager outcomes follow a **Binomial**$(n, p)$ distribution — each wager is an independent Bernoulli trial.
- The true probability $p$ is constant over the historical data period and the future wagering period (**stationarity**).
- European odds $\theta$ are fixed and known.
- Uncertainty in $p$ is adequately represented by a **Beta**$(a, b)$ prior.
- The gambling system satisfies the profitability condition $p > 1/\theta$.

**Implicit:**
- **Independence of wagers:** The Binomial model requires independent outcomes. In sports, this is violated by factors such as team fatigue, injuries, scheduling clusters, and psychological momentum.
- **Static odds:** $\theta$ is assumed constant, but real odds vary between games and bookmakers.
- **Single sequential wagers:** The framework considers one wager at a time. Simultaneous wagers are explicitly excluded.
- **Prior quality:** The choice of $(a, b)$ is treated as a principled but ultimately subjective input. Misspecification of the prior can bias the estimator.

### **5. Limitation:**

- **Prior sensitivity:** The modified fractions depend substantially on $(a, b)$. Example 5.4 demonstrates that doubling the prior strength from $(50, 50)$ to $(100, 100)$ reduces $f_0$ from $0.025$ to $0.005$ — a fivefold reduction. No objective method for prior specification is provided beyond heuristic argumentation.
- **Stationarity assumption:** Real-world winning probabilities drift over time due to changes in team composition, coaching, league rules, and bookmaker algorithms. The model provides no mechanism for adapting to a changing $p$.
- **No simultaneous wagers:** The framework handles only sequential, one-at-a-time bets. Multi-event portfolio allocation (the more common case in both sports betting and finance) remains unaddressed.
- **Computational limitation:** The loss function $l_3$ requires numerical optimization via brute-force grid search with step size $0.001$, which is computationally expensive and limited in precision.
- **No finite-time guarantees:** The convergence of $f_0$ to $k(p)$ is asymptotic. No bounds are provided on the finite-sample regret or the probability of ruin for finite $n$.

### **6. Relevance & Open Questions:**

This paper provides the foundational Bayesian treatment of Kelly under parameter uncertainty. For any research program exploring Kelly-based objectives in sequential decision-making or portfolio RL, it establishes the key insight that the betting fraction is a *statistical estimator* whose quality depends on the volume of historical data and the strength of prior beliefs.

- **Connection to distributional robustness:** The Bayesian approach in this paper and the distributional robust approach in [[PNOTES - Distributional Robust Kelly Strategy Optimal Strategy under Uncertainty in the Long-Run]] are complementary. The Bayesian approach assumes a parametric prior on $p$ and derives estimators; the DRKP approach defines a non-parametric uncertainty set and optimizes against the worst case. A hybrid approach — using the Bayesian posterior to *construct* the uncertainty set for a DRKP — is a natural synthesis.
- **Extension to continuous returns:** The Binomial model is specific to binary (win/lose) outcomes. Extending to continuous return distributions (as in portfolio optimization) would require replacing the Beta-Binomial conjugacy with a more general Bayesian model for return distributions.
- **Online adaptation:** How should the Bayesian framework adapt when evidence suggests $p$ is drifting? A natural extension is a **dynamic Beta model** where the prior at time $t$ is the posterior from time $t - 1$, possibly with discount factors to downweight stale data.
- **Multi-asset Kelly:** The simultaneous wagers problem identified in the Discussion is directly relevant to portfolio optimization, where capital must be allocated across multiple assets with correlated returns.

---

### Integration:

* **Problem:** This paper provides the essential statistical correction to the Kelly criterion's core vulnerability: its assumption that the winning probability is known. For any RL agent using log-wealth maximization as its reward signal, the agent's estimate of future returns plays the role of $\hat{p}$, and the modified Kelly framework quantifies exactly how much the agent should *discount* its own estimates based on the volume and reliability of its training data. The posterior mean $\hat{p} = (x + a)/(n + a + b)$ is a concrete mechanism for implementing *epistemic humility* in sequential capital allocation. This bridges directly to the distributional robust framework in [[PNOTES - Distributional Robust Kelly Strategy Optimal Strategy under Uncertainty in the Long-Run]], which addresses the same problem from a worst-case rather than Bayesian perspective.

* **Limitation:** The most consequential limitations for integration into a dynamic RL setting are the stationarity assumption and the restriction to single sequential wagers. Financial environments are non-stationary, and portfolio decisions involve simultaneous allocation across many assets. The paper's framework is a starting point — it establishes that Bayesian shrinkage of the Kelly fraction is theoretically justified and empirically beneficial — but extending it to the multi-asset, non-stationary, continuous-return setting required for modern portfolio RL remains open work. The prior sensitivity issue is also relevant: in an RL context, the "prior" corresponds to the agent's initial beliefs about environmental dynamics, and misspecification of these beliefs can lead to either excessive conservatism or insufficient caution.
