[[Optimal Betting Under Parameter Uncertainty - Improving the Kelly Criterion.pdf]]

# **Abstract:**

The paper demonstrates that the standard **Kelly betting criterion** performs suboptimally when the true winning probability $p$ is replaced by an estimate, because the *maximized expected utility* is an upward-biased estimator of out-of-sample performance. The authors show that **shrinking** the bet size — multiplying the Kelly-optimal stake $s^*(Q)$ by a scaling factor $k < 1$ — consistently improves expected out-of-sample utility under logarithmic utility. They propose several methods for estimating the shrinkage factor: a numerical solution using a **Beta distribution** approximation of the sampling distribution, and a closed-form **first-order approximation** that yields a simple correction formula. For general risk-averse utility functions, they establish sufficient conditions under which shrinkage (rather than swelling) is guaranteed, using the **Arrow-Pratt measure of relative risk aversion**. Empirical validation comes from Monte-Carlo simulations and an analysis of 15,794 ATP tennis matches. The paper concludes by proposing the **"Fortunate Formula"** — an improved iteration of the Kelly criterion that accounts for parameter uncertainty.

- *Context*:
	- This paper is a direct predecessor to the Bayesian framework in [[PNOTES - Modified Kelly criteria]], which cites Baker and McHale (2013) as the key prior work. Where Baker and McHale take a *frequentist* approach — modeling the sampling distribution of $\hat{p}$ and optimizing a scalar shrinkage factor $k$ — Chu et al. (2018) reframe the problem as a *Bayesian estimation* problem, treating the Kelly fraction as a statistical estimator under various loss functions. The two papers address the same fundamental vulnerability (Kelly under parameter uncertainty) but from complementary methodological directions.
	- The paper also connects to the distributional robust approach in [[PNOTES - Distributional Robust Kelly Strategy Optimal Strategy under Uncertainty in the Long-Run]], which handles uncertainty by defining non-parametric uncertainty sets over the full distribution rather than modeling the sampling distribution of a point estimate.
	- The intellectual lineage traces to **Kan and Zhou (2007)**, who showed that Markowitz portfolio choice could be modified for higher out-of-sample expected utility. Baker and McHale adapt this insight from the vector setting (portfolio weights across multiple assets) to the scalar setting (a single bet proportion), demonstrating that the logic of "optimizing the optimized utility" applies broadly in decision theory.

---

# 1. Introduction

## The Problem of Parameter Risk

The paper opens by distinguishing between two regimes: optimizing expected utility when the probability $p$ is *known*, and optimizing when $p$ must be *estimated*. This distinction — trivial in statement but consequential in practice — is the paper's motivating concern.

- **Parameter risk** is defined as the performance degradation that arises when decisions based on limited experience are applied out of sample. This is a general problem in statistics, but it takes a specific and damaging form in utility-maximized betting.
- The expected utility $E(u(s))$ may be an unbiased estimator of the true expected utility for any fixed $s$. However, the *maximized* expected utility $u(s^*(\hat{p}))$ is **not** an unbiased estimator of out-of-sample maximized utility. It will in general **overestimate** it.
	- This is because $s^*(\hat{p})$ is chosen to maximize $u$ at the estimated probability, not at the true probability. The optimization introduces an upward bias — the estimate looks better than it will perform.
- *Context*:
	- This overestimation bias is a specific instance of the **optimizer's curse** (Smith and Winkler, 2006), which arises whenever a decision is selected from among alternatives based on noisy estimates of their value. The selected option's estimated value is systematically higher than its true value, precisely because the selection process favors upward noise. In the Kelly context, $s^*(\hat{p})$ is the "selected option" — it is the bet size that looks best given $\hat{p}$ — and the optimizer's curse predicts that its in-sample utility will exceed its out-of-sample utility.

## The Plug-In Approach: Bayesian and Frequentist Equivalence

The paper notes that the Bayesian and frequentist approaches are *equivalent* in the simple case where utility is linear in $p$:

- In the **Bayesian approach**, the expected utility is $E(p)u_w + (1 - E(p))u_l$, where $E(p)$ is taken over the prior distribution.
- In the **frequentist approach**, the expected utility is $\hat{p}u_w + (1 - \hat{p})u_l$.
- Because utility is *linear* in $p$, the expected utility depends only on the first moment of the distribution of $p$ — either the prior mean $E(p)$ or the point estimate $\hat{p}$.
- The problem arises when utility is *maximized* with respect to a decision variable $s$. The maximization introduces a nonlinear transformation that breaks the linearity, making the first moment insufficient.

- *Context*:
	- The equivalence breaks down precisely at the point of optimization. For a fixed bet size $s$, the expected utility is indeed linear in $p$, and the plug-in approach is harmless. But the Kelly criterion does not use a fixed $s$ — it *optimizes* $s$ as a function of $\hat{p}$. This optimization creates a nonlinear relationship between $\hat{p}$ and the resulting expected utility, and Jensen's inequality then implies the plug-in utility is biased upward. The entire paper flows from this observation.

## Related Work

- **Kelly (1956)** introduced the growth-optimal strategy. **Thorp (2006)** extended it extensively to blackjack and stock markets. **Poundstone (2005)** popularized it as "Fortune's Formula."
- **Half-Kelly** and fractional Kelly strategies are widely used heuristics:
	- **Thorp (2006)** uses them in blackjack.
	- **MacLean, Ziemba, and collaborators** have extensively studied fractional Kelly in investment contexts (1992, 1999, 2004, 2005, 2010, 2011, 2012).
	- **Kadane (2011)** shows that half-Kelly does not correspond exactly to any utility function, but approximates a constant relative risk aversion utility with $\theta \approx 1/\lambda$, where $\lambda$ is the Kelly fraction.
- **Medo et al. (2008)** consider limited information in Kelly games — the "insider" who knows $p$ vs. the "outsider" who does not. The finding is intuitive: insiders accrue superior returns. However, this setup does not involve *estimation error* in $p$ — both players know their respective probabilities exactly. Baker and McHale clarify that simply having imperfect information does not necessitate shrinkage; *estimation error* in $\hat{p}$ does.
- **McCardle and Winkler (1992)** study optimal strategy under repeated gambles with unknown coin bias. Their setup is similar but they did not consider logarithmic utility.
- **Kan and Zhou (2007)** is the direct intellectual precursor. They showed that Markowitz optimal portfolio choice could be modified for higher out-of-sample expected utility, and that the modified investment dominated the Bayesian choice. Baker and McHale adapt this logic to the scalar Kelly setting.

- *Context*:
	- The distinction between Medo et al.'s "imperfect information" and Baker and McHale's "estimation error" is crucial. Medo's outsider knows $p = 0.5$ (or whatever the unconditional probability is) with certainty — there is no *uncertainty about the uncertainty*. Baker and McHale's bettor has an estimate $\hat{p}$ with a *sampling distribution* — the variance $\sigma^2$ of this distribution quantifies how wrong $\hat{p}$ might be. It is this second-order uncertainty (uncertainty about the probability estimate itself) that drives the need for shrinkage.

---

# 2. Bet Shrinkage

This section contains the paper's core theoretical contribution: the proof that bet shrinkage improves out-of-sample utility, and the methods for computing the optimal shrinkage factor.

## Setup and Notation

- Let $p$ be the true probability of an event, and $Q$ a random variable representing the bettor's estimate, with pdf $f(q)$, mean $p$, and variance $\sigma^2$.
- The bookmaker offers **fractional odds** $b$: a successful bet of one unit returns $b$ units of profit.
	- The connection to European odds $\theta$ (used in [[PNOTES - Modified Kelly criteria]]) is $\theta = b + 1$.
- With known $p$, the bettor maximizes **logarithmic expected utility**:

$$
E(u(s)) = p \ln(1 + bs) + (1 - p) \ln(1 - s)
$$

- The maximizer is the **Kelly betting formula**:

$$
s^*(p) = \frac{(b + 1)p - 1}{b}
$$

- *Context*:
	- In Chu et al.'s notation, $s^*(p) = ((b+1)p - 1)/b = (\theta p - 1)/(\theta - 1) = k(p)$. The two papers use different symbols for the same quantity. Baker and McHale use $s^*$ for the optimal bet proportion and $k$ for the shrinkage factor. Chu et al. use $k(p)$ for the optimal Kelly fraction and $f$ for the estimator. Be careful not to conflate Baker's $k$ (shrinkage) with Chu's $k(p)$ (Kelly fraction).

## Expected Maximized Utility Under Parameter Uncertainty (Equation 1)

When $p$ is estimated as $Q$, the expected maximized utility becomes an integral over the sampling distribution:

$$
E(u^*) = \int_0^1 f(q) \left\{ p \ln(1 + bs^*(q)) + (1 - p) \ln(1 - s^*(q)) \right\} dq \tag{1}
$$

- The integrand is maximized only when $s^*(q) = s^*(p)$, i.e., when the estimate equals the true probability. For any other value, the integrand is strictly less than the maximum.
- Therefore, $E(u^*)$ is *lower* than the naive estimate $p \ln(1 + bs^*(p)) + (1 - p) \ln(1 - s^*(p))$.
- This gap between in-sample and out-of-sample performance is the problem that bet shrinkage addresses.

- *Context*:
	- Equation (1) is the foundational equation of the paper. It computes the *true* expected utility when the bettor uses the Kelly-optimal bet for each estimated probability $q$, but the actual probability is $p$. The integral averages over all possible estimates $q$, weighted by how likely each estimate is under $f(q)$. The key insight: because the Kelly bet $s^*(q)$ is optimized for $q$, not for $p$, it will generally be wrong — sometimes too large, sometimes too small — and the concavity of the logarithm means these errors reduce expected utility. This is precisely Jensen's inequality in action. **See MATHEMATICS §2** for the formal treatment.

## Theorem 1: Shrinkage Always Improves Utility

The paper introduces a **scaling factor** $k$, replacing $s^*(Q)$ with $ks^*(Q)$, so the expected utility becomes:

$$
E(u^*) = \int_0^1 f(q) \left\{ p \ln(1 + bks^*(q)) + (1 - p) \ln(1 - ks^*(q)) \right\} dq \tag{2}
$$

**Theorem 1:** Expected maximized utility $E(u^*)$ can be increased by shrinking the bet size.

- **Proof sketch:** Evaluate $dE(u^*)/dk$ at $k = 1$:

$$
\left.\frac{dE(u^*)}{dk}\right|_{k=1} = \int_0^1 f(q) s^*(q) \left\{ \frac{pb}{1 + bs^*(q)} - \frac{1 - p}{1 - s^*(q)} \right\} dq
$$

- This reduces to $1 - (1/(b+1))E\{p/Q + (1-p)b/(1 - Q)\}$.
- Since $1/q$ and $1/(1 - q)$ are **convex functions**, Jensen's inequality gives $E(1/Q) > 1/E(Q) = 1/p$ and $E(1/(1-Q)) > 1/(1 - p)$.
- Therefore, $dE(u^*)/dk|_{k=1} < 0$ for any non-degenerate $f(q)$ (i.e., whenever $\sigma^2 > 0$).
- The second derivative $d^2E(u^*)/dk^2 < 0$ (concavity) and $dE(u^*)/dk|_{k=0} > 0$, so there exists a unique maximum at some $0 < k^* < 1$.

- *Context*:
	- The proof is elegant in its simplicity: the entire argument rests on Jensen's inequality applied to the convex functions $1/q$ and $1/(1-q)$. The convexity means that the average of $1/Q$ exceeds $1/\text{average}(Q)$ — in other words, the nonlinear interaction between the estimate $Q$ and the utility function creates a systematic drag on performance. Shrinkage reduces this drag by pulling the bet toward zero, which reduces the sensitivity of utility to estimation error. The proof also reveals that shrinkage is *always* beneficial under log utility — there is no regime where $k^* = 1$ (unless $\sigma^2 = 0$, i.e., the probability is known exactly).

## Finding the Optimal Scaling Factor $k^*$ (Equation 3)

The optimal $k^*$ satisfies the first-order condition:

$$
\frac{dE(u^*)}{dk} = \int_0^1 f(q) s^*(q) \left\{ \frac{pb}{1 + bks^*(q)} - \frac{1 - p}{1 - ks^*(q)} \right\} dq = 0 \tag{3}
$$

- This equation can be solved numerically using **Newton-Raphson iteration**:

$$
k_{n+1} = k_n - \frac{dE(u^*)/dk}{d^2E(u^*)/dk^2}
$$

- To evaluate the integral, a specific functional form for $f(q)$ is required.

## The Beta Method

The **Beta distribution** is the natural choice for approximating $f(q)$:

- It is defined on $(0, 1)$, matching the parameter space of $p$.
- With only two free parameters, it is specified by $p$ (mean) and $\sigma^2$ (variance).
- Parameters: $\alpha = p\{p(1-p)/\sigma^2 - 1\}$, $\beta = (1-p)\{p(1-p)/\sigma^2 - 1\}$.
- The maximum attainable $\sigma$ is $1/2$ (corresponding to a probability that is 0 or 1 with equal chance). A uniform distribution gives $\sigma \approx 0.289$.

- *Context*:
	- In the Bayesian framework of [[PNOTES - Modified Kelly criteria]], the Beta distribution serves as a *prior* on $p$, encoding the bettor's beliefs. Here, in Baker and McHale's frequentist framework, the Beta distribution approximates the *sampling distribution* of the estimator $\hat{p}$. The parametric form is the same, but the philosophical interpretation differs: Bayesian prior belief vs. frequentist sampling variability. The practical consequence is similar — both lead to bet shrinkage — but the mechanisms are distinct.

## First-Order Approximation (Equation 5)

By expanding $u^*$ in a Taylor series about $s^*(p)$ and keeping first-order terms in $\sigma^2$, the paper derives:

$$
k^* \approx \frac{s^*(p)^2}{s^*(p)^2 + \left(\frac{b+1}{b}\right)^2 \sigma^2} \tag{5}
$$

- This is the "back of the envelope" correction. It requires only $s^*(p)$ (the Kelly bet at the estimated probability) and $\sigma^2$ (the variance of the estimate).
- The half-Kelly method ($k = 1/2$) is optimal when $\sigma \approx p - 1/(1+b)$.
	- This provides the first *quantitative* justification for half-Kelly: it is the correct shrinkage when the standard error of the probability estimate roughly equals the *edge* (the excess probability above the break-even threshold).

- *Context*:
	- Equation (5) is arguably the paper's most practically useful result. It converts the abstract concept of "accounting for parameter uncertainty" into a simple multiplicative correction. A bettor who can estimate the standard error $\sigma$ of their probability estimate can immediately compute the adjusted bet size. The formula has a natural interpretation: $k^*$ is the ratio of the signal (squared Kelly bet) to the signal plus noise (squared Kelly bet plus scaled variance). As $\sigma \to 0$, $k^* \to 1$ (no shrinkage needed). As $\sigma$ grows, $k^* \to 0$ (the estimate is too noisy to act on).
	- The formula also reveals that shrinkage depends on the *relative* magnitude of estimation error to edge. A small edge with even modest error requires severe shrinkage; a large edge tolerates more error. This explains why aggressive Kelly betting is especially dangerous when the perceived edge is small — precisely the regime where most real-world betting systems operate.

---

# 3. Some Complications

## Negative Bets and Short-Selling

- If $Q < 1/(1+b)$, the Kelly solution gives $s^*(Q) < 0$ — a negative bet, analogous to **short-selling** in financial markets.
- In betting exchanges, one can "lay" bets (bet against an outcome), making negative bets feasible. The analysis in Section 2 implicitly assumes this.
- At the racetrack, negative bets are impossible. When $Q < 1/(1+b)$, the bettor simply does not bet, and the expected utility for that realization is zero.
- The inability to place negative bets:
	- Does **not** affect the approximate formula (5), which is valid for small $\sigma$.
	- Does invalidate the general proof that $k^* < 1$ for any unbiased $f(q)$. The modified integral (with lower limit $1/(1+b)$ instead of $0$) could theoretically permit $k^* > 1$, though the authors report never finding such a case empirically.

## Two-Player Match Decision Tree

When betting on a match between two players with odds $b_1, b_2$:

- If $Q < 1/(1+b_1)$ and $1-Q < 1/(1+b_2)$: **no bet**.
- If $Q > 1/(1+b_1)$ and the utility from betting on player 1 exceeds that of player 2: **bet on player 1**.
- Otherwise: **bet on player 2**.

The integrand of Equation (3) is modified to reflect this decision tree, with the bet size and direction changing across different regions of $q$.

- *Context*:
	- The decision tree introduces piecewise structure into the integral (3), complicating the numerical solution. The bettor effectively has three regimes: bet on player 1 (for high $q$), do not bet (for intermediate $q$), and bet on player 2 (for low $q$). The shrinkage factor $k$ applies to all bets uniformly — it does not distinguish between bets on different players. This is a limitation: ideally, the shrinkage factor would be different for each regime, but solving for regime-specific shrinkage factors would require a more complex optimization.

---

# 4. Bet Rescaling for General Risk-Averse Utility Functions

The paper generalizes from logarithmic utility to arbitrary risk-averse utility functions $u$, asking: under what conditions does shrinkage (rather than swelling) occur?

## General Setup

For a general utility function, the expected utility at bet $s^*$ is:

$$
E(u) = pu(1 + bs^*) + (1 - p)u(1 - s^*)
$$

The scaling analysis examines $dE(u^*)/dk|_{k=1}$ as before. Shrinkage occurs if this derivative is negative.

Using the identity $bqu'(1 + bs^*(q)) = (1 - q)u'(1 - s^*(q))$ — which is the first-order condition for $s^*(q)$ under any differentiable utility — the derivative can be expressed as:

$$
\left.\frac{dE(u^*)}{dk}\right|_{k=1} = -E\left\{(Q - p) \frac{s^*(Q) u'(1 - s^*(Q))}{Q}\right\} \tag{7}
$$

A sufficient condition for shrinkage is that the function $h(Q) = s^*(Q) u'(1 - s^*(Q))/Q$ is increasing, so that $E\{(Q - p)h(Q)\} > 0$.

## Theorem 2: Derivative Condition

**Theorem 2:** The condition $s_1^* > s^*/p$ is sufficient for shrinkage, where $s_1^* = ds^*/dp$.

- From Equation (7), since $u'' < 0$ (risk aversion), the condition $s_1^* > s^*/p$ ensures all terms have the correct sign for shrinkage.
- This condition is satisfied for logarithmic utility (where $s_1^* = (b+1)/b$ and $s^*/p = ((b+1)p - 1)/(bp)$, and $(b+1)/b > ((b+1)p - 1)/(bp)$ for $p < 1$).
- The condition is of limited direct use because it is not expressed in terms of the utility function.

## Theorem 3: Arrow-Pratt Condition

**Theorem 3:** The condition

$$
R_r(1 + bs^*) < \frac{b^2}{b^2 - 1/(1 - p)} \tag{10}
$$

is sufficient for shrinkage, where $R_r(x) = -xu''(x)/u'(x)$ is the **Arrow-Pratt measure of relative risk aversion**.

- This condition is satisfied for **logarithmic utility** (where $R_r = 1$) and for **isoelastic utility** $u(x) = (x^\alpha - 1)/\alpha$ with $0 < \alpha < 1$ (where $R_r \leq 1$).
- It is **not** satisfied for **exponential utility** $u(x) = -\exp(-\lambda x)$, where $R_r = \lambda x$ can be arbitrarily large.

- *Context*:
	- The Arrow-Pratt measure $R_r$ quantifies how much a decision-maker dislikes proportional gambles. For utility functions with low relative risk aversion ($R_r$ small), the penalty for overbetting is large relative to the penalty for underbetting, and shrinkage is always optimal. For utility functions with high relative risk aversion ($R_r$ large), the situation is more complex — the utility function may penalize underbetting so severely that swelling is preferred. The threshold in Equation (10) quantifies exactly when this transition occurs.

## Bet Swelling Under Exponential Utility

For exponential utility $u(x) = -\exp(-\lambda x)$, the optimal bet is:

$$
s^*(p) = \frac{\ln(bp/(1 - p))}{(b + 1)\lambda} \tag{11}
$$

- The bet proportion can exceed unity (the bettor can borrow), since exponential utility is defined for negative wealth.
- When the odds are very favorable ($b \gg 1$), the analysis shows $dE(u^*)/dk|_{k=1} > 0$, meaning **bet swelling** ($k^* > 1$) is optimal.
	- Intuitively, exponential utility does not penalize losses as severely as logarithmic utility (it avoids the $\ln(0) = -\infty$ catastrophe), so when odds are excellent, it is worth *increasing* exposure to capture the upside despite estimation error.

- *Context*:
	- Bet swelling is a surprising theoretical result. It means that for certain utility functions and very favorable odds, the optimal response to uncertainty is to bet *more*, not less. This is the opposite of the conventional wisdom. However, the authors emphasize that this situation is practically negligible: "excellent odds of winning are not often on offer." The result is important theoretically because it shows that shrinkage is not a universal consequence of parameter uncertainty — it depends on the interaction between the utility function's curvature and the odds.

## Theorem 4: Probability Bound Condition

**Theorem 4:** The conditions $u''' > 0$ and

$$
p \leq \frac{2b^2 + b - 1}{b^3 + b^2 - b - 1}
$$

are sufficient for shrinkage under any smooth risk-averse utility function.

- For $b = 2$, this yields $p \leq 1$ — shrinkage is **always** required for $b \leq 2$.
- The condition $u''' > 0$ (the third derivative of utility is positive) is satisfied by all standard utility functions (logarithmic, isoelastic, exponential).

- *Context*:
	- Theorem 4 is the strongest result in the paper. It establishes that for any standard utility function, shrinkage is guaranteed whenever the odds are not too favorable. Since most real-world odds satisfy $b \leq 2$ (even money or worse), bet shrinkage is a near-universal recommendation in practical settings. The condition $u''' > 0$ is sometimes called **prudence** in the economics literature — it implies that the decision-maker has a precautionary motive for saving (or, here, for betting less). The paper notes that "all utility functions known to us have this property."

## General Rescaling Formula (Equations 14–15)

Constructing the bootstrapped estimator $E(u^*(ks^*(Q)))$, expanding by Taylor series about $s^*(p)$, and maximizing with respect to $k$:

$$
k^* = \frac{s^*(p) E(s^*(Q))}{E(s^*(Q)^2)} \tag{14}
$$

This is valid for small variance $\sigma^2$. Expanding $s^*(Q)$ in a Taylor series about $p$:

$$
k^* = \frac{s^*(p)\{s^*(p) + \tfrac{1}{2} s_2^*(p) \sigma^2\}}{s^*(p)^2 + (s_1^*(p)^2 + s^*(p) s_2^*(p)) \sigma^2} \tag{15}
$$

- The form (14) does **not** depend explicitly on the form of $u$, but does so implicitly through $s^*(Q)$, which depends on $u$.
- For logarithmic utility, Equation (14) reduces to the approximation (5).

- *Context*:
	- Equations (14–15) are the general-purpose tools of the paper. They allow a practitioner using *any* risk-averse utility function to compute an approximate shrinkage factor, requiring only the ability to evaluate $s^*(Q)$ (the optimal bet under their utility function for a given probability estimate). This generalizes the "back of the envelope" formula (5) beyond logarithmic utility. The key insight: the shrinkage factor is the ratio of $s^*(p) \cdot E(s^*(Q))$ (a product of first moments) to $E(s^*(Q)^2)$ (a second moment). When the variance of $s^*(Q)$ is large relative to its mean, $k^*$ is small — more shrinkage is needed.

---

# 5. Examples

## 5.1 Simulated Gambling Study

### Setup

- A bettor receives even odds ($b = 1$) on an event with true probability $p = 1 - (5/6)^5 \approx 0.57$ (whether a six can be thrown in 5 tosses of a die).
- The bettor does **not** know $p$ and estimates it by performing sets of tosses.
- **10,000 simulations**, each containing **100 sequential bets**.
- In half the simulations, the number of toss sets is drawn uniformly between 8 and 20 (creating varying levels of estimation error). In the other half, the bettor knows the exact probability.

### Results — Favorable Bet (Table 1)

| Method | Mean Final Bankroll | Median | S.D. | Mean Utility |
|---|---|---|---|---|
| Kelly (known $p$) | 44.5 | 7.48 | 189.5 | 1.98 |
| Half-Kelly (known $p$) | 6.88 | 4.41 | 8.05 | 1.47 |
| Kelly (estimated $p$) | 57.1 | 2.1 | 669.0 | 0.682 |
| Half-Kelly (estimated $p$) | 7.4 | 3.5 | 14.3 | 1.238 |
| Approx. shrinkage | 26.3 | 2.7 | 189.9 | 0.936 |
| Beta shrinkage | 27.2 | 3.0 | 183.1 | 1.061 |
| Binomial shrinkage | 26.5 | 3.1 | 171.8 | 1.090 |

- When $p$ is known exactly, raw Kelly outperforms half-Kelly (utility 1.98 vs. 1.47), as guaranteed by theory.
- When $p$ is estimated, raw Kelly's mean utility drops to **0.682** — below the no-betting threshold of 1.0 (in log-bankroll terms, the bettor is worse off than if they had not bet at all).
- The Beta shrinkage method achieves utility **1.061** (above the no-betting threshold) and the binomial shrinkage method achieves **1.090**.
- Raw Kelly produces a **deceptively high mean bankroll** (57.1) driven by a few rare massive winners, while the **median bankroll** (2.1) reveals that most bettors lose.

- *Context*:
	- The mean vs. median discrepancy in the raw Kelly results is a classic signature of *heavy-tailed* outcomes. Under Kelly betting with estimation error, a few lucky bettors who overestimate a winning probability compound their gains explosively, while many bettors who overestimate end up overbetting and losing their bankrolls. The shrinkage methods eliminate the extreme winners (reducing the mean) but protect the majority of bettors (increasing the median from 2.1 to 3.0–3.1). For a risk-sensitive decision-maker, the median is a much better indicator of typical outcomes than the mean.

### Results — Unfavorable Bet (Table 2)

- The bet is changed to $p = 1 - (5/6)^3 \approx 0.42$ (whether a six can be thrown in 3 tosses). At even odds, this is an **unfavorable** bet — the Kelly fraction is zero.
- With known $p$, no method bets (utility = 0, bankroll = 1).
- With estimated $p$, raw Kelly yields mean utility **-0.698** and mean bankroll **0.709**. Shrinkage methods reduce the losses: approximate shrinkage gives **-0.296**, beta shrinkage gives **-0.322**, and half-Kelly gives **-0.253**.
- The half-Kelly method performs well in this regime, consistent with findings by Grant and Johnstone (2010) that a 40% Kelly bet works well for Australian Football League matches.

## 5.2 Tennis Betting

### Data and Model

- **15,794 ATP Tour matches** from 2005–2011, sourced from tennis-data.co.uk.
- Probability estimation: **logistic regression** using the natural logarithm of the ratio of players' world ranking points as the sole covariate.
- Training: 2005–2008 (in-sample). Testing: 2009 (out-of-sample).
- For each prediction, 1,000 matches were randomly sampled from the in-sample to fit the model, reducing bias.

### Error Estimation

- The error variance $\sigma^2$ was estimated by:
	1. Computing log-odds ratios for both bookmaker and model predictions.
	2. Partitioning the error equally between bookmaker and model: $\hat{\sigma}^2_{\text{log-odds}} = \tfrac{1}{2}\text{Var}(\text{log-odds}_{\text{model}} - \text{log-odds}_{\text{bookie}})$.
	3. Using the **delta method** to convert the log-odds variance to a variance on $\hat{p}$.
- The authors acknowledge this is crude — the bookmaker's predictions may be "sharper" (conditioned on more information), so some of the disagreement reflects information asymmetry rather than model error. Using the Hessian from the logistic regression would "greatly underestimate $\sigma^2$" because it assumes the model is correctly specified.

- *Context*:
	- The error estimation problem is the Achilles' heel of the entire framework. The shrinkage factor depends critically on $\sigma^2$, but $\sigma^2$ is itself uncertain. If $\sigma^2$ is underestimated, the bet will be insufficiently shrunk; if overestimated, the bet will be excessively conservative. The authors' pragmatic solution — splitting the bookmaker-model disagreement — is a reasonable heuristic, but it illustrates a fundamental limitation: one needs a reliable estimate of the estimation error, which is a higher-order statistical problem. This difficulty is acknowledged in the Conclusions as a key challenge.

### Results (Table 3)

| Method | Mean Bankroll | Mean Utility | % Player 1 | % Player 2 | % No Bet |
|---|---|---|---|---|---|
| Kelly | 0.984 | -0.0482 | 30.85 | 49.65 | 19.49 |
| Half-Kelly | 0.992 | -0.0161 | 30.85 | 49.65 | 19.49 |
| Approx. shrinkage | 1.001 | -0.0024 | 30.85 | 49.65 | 19.49 |
| Beta shrinkage | 1.000 | -0.0027 | 9.98 | 35.85 | 54.17 |

- Raw Kelly yields negative mean utility (-0.0482) and a mean bankroll below 1 (0.984) — the bettor loses money on average.
- The approximate shrinkage method achieves a mean bankroll of **1.001** (marginally above break-even) and the best mean utility (-0.0024).
- The beta method achieves similar utility but makes far fewer bets (54.17% no-bet vs. 19.49%) because it sometimes shrinks the bet to zero (when $k^*$ is negative, the feasible optimum is zero).
- **Brier scores**: the ranking model (0.2271) was less sharp than the bookmaker's implied probabilities (0.1886). The authors did not "beat the bookie," but the shrinkage methods successfully mitigated losses.

- *Context*:
	- The failure to beat the bookmaker is not the point — the paper's claim is comparative: given the *same* probability model, the shrinkage methods outperform raw Kelly. The ranking model is deliberately simple (a single covariate), and the bookmaker's odds reflect far richer information. The contribution is methodological, not predictive. A bettor with a *better* model (lower Brier score) could apply the same shrinkage framework to achieve actual profits while maintaining the protective benefits of shrinkage.

---

# 6. Conclusions

## The Fortunate Formula

The paper closes by proposing an improved iteration of the Kelly criterion, derived from the first-order approximation (5):

$$
s^* = \frac{((b+1)p - 1)^3}{b\left\{((b+1)p - 1)^2 + (b+1)^2 \sigma^2\right\}}
$$

This is the product $k^* \cdot s^*(p)$, where $k^*$ is from Equation (5) and $s^*(p) = ((b+1)p - 1)/b$. The authors dub this the **"Fortunate Formula"** (a play on Poundstone's "Fortune's Formula").

- *Context*:
	- The Fortunate Formula is the paper's recommended practical tool. It takes two inputs beyond the standard Kelly criterion: the fractional odds $b$ and the estimated probability $p$ (which the standard Kelly also requires), *plus* the standard error $\sigma$ of the probability estimate. The formula automatically scales the bet: when $\sigma$ is small relative to the edge $((b+1)p - 1)$, the bet is close to the standard Kelly; when $\sigma$ is large, the bet shrinks toward zero. This makes it a *smooth* generalization of the Kelly criterion that degenerates to the original formula when uncertainty vanishes.
	- Compared to the approach in [[PNOTES - Modified Kelly criteria]], the Fortunate Formula is:
		- *Simpler:* it is a closed-form expression rather than a posterior computation requiring prior specification.
		- *Less principled:* it is a first-order approximation rather than an exact Bayes estimator.
		- *Frequentist:* it does not require a prior distribution on $p$, only an estimate of $\sigma$.
	- For a practitioner, the Fortunate Formula is the fastest path from "I have an edge estimate and an error estimate" to an adjusted bet size.

## General Implications for Decision Theory

- The paper identifies a **general implication**: whenever utility is maximized with respect to a decision variable and the parameters entering the utility are uncertain, the resulting decision can be improved by rescaling.
- This applies beyond betting: portfolio optimization (Kan and Zhou, 2007), insurance decisions, and any setting where utility maximization uses plug-in estimates.
- The direction of rescaling (shrinkage vs. swelling) depends on the curvature of the utility function, but for standard risk-averse utility functions and realistic parameters, shrinkage is almost always the correct adjustment.
- Extensions to more complex betting scenarios (spread betting, each-way horse racing, simultaneous bets on multiple events) are identified as future work.

---

# MATHEMATICS

The mathematical framework of this paper proceeds in three stages. First, the standard Kelly criterion is derived from expected log growth maximization under known probability. Second, the parameter uncertainty problem is formalized: when the true probability $p$ is unknown and replaced by a random estimate $Q$, the expected maximized utility decreases. The paper proves that this decrease can be partially recovered by scaling the bet size, derives exact and approximate formulas for the optimal scaling factor, and extends the analysis to arbitrary risk-averse utility functions. The derivation chain follows: Kelly objective → Kelly optimal bet → expected utility under uncertainty → scaling framework → Jensen's inequality shrinkage proof → first-order approximation → general utility rescaling → sufficient conditions for shrinkage → Fortunate Formula.

### 1. Kelly Expected Utility and Optimal Bet

The bettor wagers a proportion $s$ of their bankroll at fractional odds $b$ (profit per unit wagered on a win). The expected log utility is:

**Kelly Expected Utility ($E(u(s))$):**

$$
E(u(s)) = p \ln(1 + bs) + (1 - p) \ln(1 - s) \tag{K}
$$

where $p$ is the true probability of winning and $s \in [0, 1]$ is the bet proportion. Differentiating with respect to $s$ and setting equal to zero:

$$
\frac{dE(u)}{ds} = \frac{pb}{1 + bs} - \frac{1 - p}{1 - s} = 0
$$

Solving for $s$ yields the **Kelly betting formula**:

**Kelly Betting Formula ($s^*(p)$):**

$$
s^*(p) = \frac{(b + 1)p - 1}{b} \tag{K*}
$$

The bet is positive (and the formula is applicable) only when $(b+1)p > 1$, i.e., $p > 1/(b+1)$. The second derivative of (K) evaluated at $s^*(p)$ is:

$$
\frac{d^2E(u)}{ds^2}\bigg|_{s^*} = -\frac{p b^2}{(1 + bs^*)^2} - \frac{1 - p}{(1 - s^*)^2} < 0
$$

confirming that $s^*(p)$ is a maximum.

### 2. Expected Maximized Utility Under Parameter Uncertainty

When $p$ is unknown and estimated by a random variable $Q$ with pdf $f(q)$, mean $E(Q) = p$, and variance $\text{Var}(Q) = \sigma^2$, the expected maximized utility is:

**Expected Maximized Utility ($E(u^*)$):**

$$
E(u^*) = \int_0^1 f(q) \left\{ p \ln(1 + bs^*(q)) + (1 - p) \ln(1 - s^*(q)) \right\} dq \tag{1}
$$

The integrand is the *true* expected utility (evaluated at $p$) of the Kelly bet $s^*(q)$ computed at the *estimated* probability $q$. Since $s^*(q) = s^*(p)$ only when $q = p$, and the integrand is strictly concave in $s^*(q)$ around this maximum, the integral $E(u^*)$ is strictly less than the naive expected utility $p \ln(1 + bs^*(p)) + (1 - p) \ln(1 - s^*(p))$ for any non-degenerate $f(q)$.

### 3. The Scaling Framework and Theorem 1

Introduce a scaling factor $k$ so the bet becomes $ks^*(Q)$:

**Scaled Expected Utility:**

$$
E(u^*) = \int_0^1 f(q) \left\{ p \ln(1 + bks^*(q)) + (1 - p) \ln(1 - ks^*(q)) \right\} dq \tag{2}
$$

Differentiating with respect to $k$ and evaluating at $k = 1$:

$$
\left.\frac{dE(u^*)}{dk}\right|_{k=1} = \int_0^1 f(q) s^*(q) \left\{ \frac{pb}{1 + bs^*(q)} - \frac{1 - p}{1 - s^*(q)} \right\} dq
$$

Substituting $s^*(q) = ((b+1)q - 1)/b$ and simplifying:

$$
\left.\frac{dE(u^*)}{dk}\right|_{k=1} = 1 - \frac{1}{b+1} E\left\{\frac{p}{Q} + \frac{(1 - p)b}{1 - Q}\right\}
$$

Since $1/q$ and $1/(1-q)$ are convex functions of $q$, Jensen's inequality gives $E(1/Q) > 1/E(Q) = 1/p$ and $E(1/(1-Q)) > 1/(1 - p)$. Therefore:

$$
\frac{1}{b+1}\left(\frac{p}{p} + \frac{(1-p)b}{1-p}\right) = \frac{1}{b+1}(1 + b) = 1
$$

and the Jensen strict inequality yields $dE(u^*)/dk|_{k=1} < 0$ for non-degenerate $f$. Combined with $dE(u^*)/dk|_{k=0} > 0$ and negative second derivative, there exists a unique $k^* \in (0, 1)$ maximizing $E(u^*)$.

### 4. First-Order Condition for Optimal Scaling

The optimal scaling factor $k^*$ satisfies:

**Optimal Scaling FOC:**

$$
\int_0^1 f(q) s^*(q) \left\{ \frac{pb}{1 + bk^*s^*(q)} - \frac{1 - p}{1 - k^*s^*(q)} \right\} dq = 0 \tag{3}
$$

This nonlinear equation in $k^*$ is solved numerically via Newton-Raphson iteration:

$$
k_{n+1} = k_n - \frac{dE(u^*)/dk}{d^2E(u^*)/dk^2}
$$

where $f(q)$ is taken as a Beta distribution with parameters $\alpha = p\{p(1-p)/\sigma^2 - 1\}$ and $\beta = (1-p)\{p(1-p)/\sigma^2 - 1\}$.

### 5. Taylor Approximation and First-Order Shrinkage Factor

Expanding $E(u^*)$ about $s^*(p)$ via Taylor series:

$$
E(u^*) \approx E(u(s^*(p))) + \frac{1}{2} \frac{\partial^2 E(u(x))}{\partial x^2}\bigg|_{x=s^*(p)} \int_0^1 (ks^*(q) - s^*(p))^2 f(q) \, dq \tag{4}
$$

The first-order term vanishes because $\partial E(u(x))/\partial x|_{x=s^*(p)} = 0$ (by definition of the optimum). Differentiating (4) with respect to $k$ and equating to zero:

**First-Order Shrinkage Factor ($k^*$):**

$$
k^* \approx \frac{s^*(p)^2}{s^*(p)^2 + \left(\frac{b+1}{b}\right)^2 \sigma^2} \tag{5}
$$

This formula requires only the Kelly bet $s^*(p)$, the odds $b$, and the estimation variance $\sigma^2$. The half-Kelly strategy ($k = 1/2$) is recovered when $s^*(p)^2 = ((b+1)/b)^2 \sigma^2$, which simplifies to $\sigma \approx s^*(p) \cdot b/(b+1) = p - 1/(b+1)$ — the standard error equals the edge.

### 6. General Utility Functions and Bet Rescaling

For a general risk-averse utility $u$ (with $u'' < 0$), the identity from the first-order condition of $s^*(q)$ is:

$$
bq \, u'(1 + bs^*(q)) = (1 - q) \, u'(1 - s^*(q)) \tag{6}
$$

The derivative of expected maximized utility with respect to scaling is:

$$
\left.\frac{dE(u^*)}{dk}\right|_{k=1} = -E\left\{(Q - p) \frac{s^*(Q) \, u'(1 - s^*(Q))}{Q}\right\} \tag{7}
$$

For the small-$\sigma$ case, this becomes:

$$
\left.\frac{dE(u^*)}{dk}\right|_{k=1} = -\frac{\sigma^2 s_1^*}{p}\left\{u'(1 - s^*) - s^* u''(1 - s^*) - \frac{s^* u'(1 - s^*)}{p}\right\} \tag{9-approx}
$$

where $s^*$ and $s_1^* = ds^*/dp$ are evaluated at $p$.

### 7. Sufficient Conditions for Shrinkage (Theorems 2–4)

**Theorem 2:** $s_1^* > s^*/p$ is sufficient for shrinkage. This follows directly from the sign of Equation (7) given $u'' < 0$.

**Theorem 3 (Arrow-Pratt condition):** Shrinkage is guaranteed if:

$$
R_r(1 + bs^*) < \frac{b^2}{b^2 - 1/(1 - p)} \tag{10}
$$

where $R_r(x) = -xu''(x)/u'(x)$ is the Arrow-Pratt measure of relative risk aversion. This condition is satisfied for all utility functions with $R_r \leq 1$ (including logarithmic and isoelastic).

**Theorem 4 (Probability bound):** If $u''' > 0$ (prudence) and:

$$
p \leq \frac{2b^2 + b - 1}{b^3 + b^2 - b - 1}
$$

then shrinkage is guaranteed. For $b \leq 2$, this yields $p \leq 1$, so shrinkage is *always* required for any prudent utility function at even-money or worse odds.

### 8. General Rescaling Formula

For arbitrary utility, the Taylor-series approximation of $k^*$ generalizes (5) to:

**General Rescaling Formula ($k^*$):**

$$
k^* = \frac{s^*(p) \, E(s^*(Q))}{E(s^*(Q)^2)} \tag{14}
$$

This does not depend explicitly on the utility function $u$, but does so implicitly through the bet function $s^*(Q)$. Expanding $s^*(Q)$ in Taylor series about $p$:

$$
k^* = \frac{s^*(p)\{s^*(p) + \tfrac{1}{2} s_2^*(p) \sigma^2\}}{s^*(p)^2 + (s_1^*(p)^2 + s^*(p) s_2^*(p)) \sigma^2} \tag{15}
$$

where $s_1^* = ds^*/dp$ and $s_2^* = d^2s^*/dp^2$.

### 9. The Fortunate Formula

Multiplying the first-order shrinkage factor (5) by the Kelly bet $s^*(p) = ((b+1)p - 1)/b$:

**Fortunate Formula ($s^*$):**

$$
s^* = k^* \cdot s^*(p) = \frac{((b+1)p - 1)^3}{b\left\{((b+1)p - 1)^2 + (b+1)^2 \sigma^2\right\}} \tag{FF}
$$

When $\sigma = 0$, this reduces to the standard Kelly formula $s^*(p) = ((b+1)p - 1)/b$. As $\sigma$ increases, the formula shrinks the bet toward zero. The cubic numerator and quadratic-plus-variance denominator encode the interaction between edge and uncertainty: the bet scales as the *cube* of the edge divided by the *square* of the edge plus the *square* of the scaled uncertainty.

---

### **1. Problem:**

The standard Kelly criterion treats the estimated winning probability $\hat{p}$ as if it were the true probability $p$, ignoring the estimation error inherent in any finite-sample estimate. This **plug-in approach** causes the Kelly-optimal bet to be systematically too large, because the maximized expected utility is an upward-biased estimator of out-of-sample performance. The paper addresses the question: how should the Kelly bet be rescaled to account for this parameter uncertainty, and by how much?

### **2. Setup:**

The model assumes a **single binary bet** at fractional odds $b$, with true winning probability $p$ unknown to the bettor. The bettor has an unbiased estimate $Q$ with sampling distribution $f(q)$, mean $p$, and variance $\sigma^2$. The decision variable is the proportion $s \in [0, 1]$ of the bankroll to wager. The primary utility function is **logarithmic** ($u(s) = \ln(\text{wealth})$), which is the basis of the Kelly criterion. The analysis is extended to general risk-averse utility functions $u$ with $u'' < 0$. The optimization is over a scalar **shrinkage factor** $k$ applied multiplicatively to the Kelly bet: $s = k \cdot s^*(Q)$.

### **3. Key Idea:**

By treating the Kelly-optimal bet $s^*(Q)$ as a random quantity (because $Q$ is random), the paper shows that the expected out-of-sample utility is an integral over the sampling distribution of $Q$, and this integral is strictly less than the plug-in utility at $k = 1$. Using Jensen's inequality, the authors prove that reducing $k$ below 1 always improves expected utility under log utility (Theorem 1). The optimal shrinkage factor $k^*$ balances the cost of betting suboptimally (too small) against the cost of parameter-sensitive overbetting (too large), yielding a first-order closed-form approximation that can be applied by any bettor who can estimate the standard error of their probability estimate.

### **4. Assumptions:**

**Explicit:**
- **Logarithmic utility** is the primary utility function (extended to general risk-averse utilities in Section 4).
- The estimate $Q$ is **unbiased**: $E(Q) = p$.
- The sampling distribution $f(q)$ can be adequately approximated by a **Beta distribution** (for the Beta Method).
- **Fractional odds** $b$ are fixed and known at the time of the bet.
- The bet is a **single event** — no simultaneous wagers.

**Implicit:**
- **Quantifiable estimation error**: the bettor can accurately estimate $\sigma^2$. If the model generating $\hat{p}$ is structurally misspecified, $\sigma^2$ will be underestimated.
- **Static environment**: the bet does not affect the odds (no market impact).
- **Independent events**: the sequential bets in the examples are treated as independent draws from the same probability $p$.
- **Constant $p$**: the true probability does not change over time (stationarity).

### **5. Limitation:**

- **Uncertainty about the uncertainty**: the framework requires $\sigma^2$ as an input, but estimating $\sigma^2$ is itself a difficult problem. Model misspecification leads to underestimation of $\sigma^2$, resulting in insufficient shrinkage. The paper acknowledges: "it is when $\sigma$ is largest that it is likely to be hardest to estimate."
- **No negative bets**: when short-selling is impossible, the general proof of $k^* < 1$ breaks down. The modified integral may theoretically permit $k^* > 1$, though this was never observed empirically.
- **Single-event restriction**: the framework handles only one bet at a time. Multi-event portfolios (the simultaneous betting case) introduce correlations and a multivariate optimization that is left as future work.
- **First-order approximation**: the closed-form shrinkage factor (Equation 5) and the Fortunate Formula are first-order Taylor approximations valid for small $\sigma^2$. For large estimation errors, higher-order terms may be significant.
- **No dominance proof**: unlike Kan and Zhou (2007), the authors were unable to prove that the shrunken bet dominates the Bayesian solution for all parameter values. Dominance is demonstrated empirically but not theoretically.
- **Static probability**: no mechanism for adapting to time-varying $p$, which limits applicability in non-stationary environments such as financial markets.

### **6. Relevance & Open Questions:**

This paper provides the foundational frequentist treatment of Kelly betting under parameter uncertainty, complementing the Bayesian approach in [[PNOTES - Modified Kelly criteria]] and the distributionally robust approach in [[PNOTES - Distributional Robust Kelly Strategy Optimal Strategy under Uncertainty in the Long-Run]]. The three papers together form a trilogy: frequentist shrinkage (Baker & McHale), Bayesian estimation (Chu et al.), and worst-case robust optimization (Sun et al.).

- **Shrinkage as epistemic humility:** The paper establishes that any agent maximizing expected utility under parameter uncertainty should bet *less* than the plug-in optimum. For an RL agent using log-wealth as a reward signal, this means the policy should incorporate a multiplicative discount on position sizes proportional to the agent's confidence in its return predictions.
- **The Fortunate Formula as a baseline:** The closed-form Fortunate Formula provides a simple, interpretable baseline against which more sophisticated uncertainty-adjusted strategies (Bayesian, robust) can be compared.
- **Open question — multi-asset generalization:** How does the scalar shrinkage factor $k$ generalize to a vector of shrinkage factors across multiple simultaneous bets? This is the portfolio version of the problem and connects to Kan and Zhou's (2007) work on Markowitz optimization.
- **Open question — adaptive shrinkage:** Can the shrinkage factor be made dynamic, adapting to a changing $\sigma^2$ as the agent accumulates more data and its model improves?
- **Open question — model misspecification:** What happens when the model generating $\hat{p}$ is not just noisy but systematically biased? The unbiasedness assumption $E(Q) = p$ is crucial; relaxing it introduces a bias-variance tradeoff in the shrinkage factor.

---

### Integration:

* **Problem:** Baker and McHale (2013) formalize the most basic failure mode of Kelly betting in practice: the plug-in estimator for $p$ produces an overconfident bet because optimization under uncertainty inflates the perceived edge. This is the same problem addressed by Chu et al. (2018) from a Bayesian perspective and Sun et al. (2021) from a robust perspective. For a research program exploring Kelly-based objectives in portfolio RL, the core takeaway is that *any* Kelly-derived reward signal must be discounted by a factor reflecting the agent's estimation uncertainty. The Fortunate Formula $s^* = ((b+1)p - 1)^3 / \{b[((b+1)p - 1)^2 + (b+1)^2\sigma^2]\}$ provides the simplest instantiation of this principle, converting an edge estimate and an error estimate into an uncertainty-adjusted position size.

* **Limitation:** The most consequential limitation for integration into portfolio RL is the restriction to single binary bets with constant probability. Real portfolio environments involve multiple assets with continuous returns, time-varying expected returns, and correlated outcomes. The scalar shrinkage factor $k$ must generalize to a matrix of shrinkage factors (or equivalent) in the multi-asset case. Furthermore, the requirement for an accurate $\sigma^2$ estimate maps onto the RL challenge of epistemic uncertainty quantification: the agent must not only estimate expected returns but also estimate *how wrong* those estimates might be — a meta-estimation problem that compounds the difficulty of the original prediction task. The paper's reliance on the unbiasedness assumption ($E(Q) = p$) is also restrictive; RL agents trained on finite data may have systematically biased value estimates, introducing a bias-variance interaction not captured by the shrinkage framework.
