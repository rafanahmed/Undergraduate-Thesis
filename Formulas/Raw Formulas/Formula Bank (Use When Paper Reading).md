# Mathematical Foundations of Multiplicative Growth and Optimal Asset Allocation

[[Deep Reinforcement Learning for Optimal Asset Allocation Using DDPG with TiDE.pdf]]

---


## Log-Utility Maximisation Identity (Kelly, 1956)

The architecture of modern portfolio management is shifting from classical Mean-Variance Optimisation (MVO) toward Reinforcement Learning (RL) frameworks grounded in the **Kelly Criterion**. Traditional MVO is fundamentally limited by *plug-in estimation*, where sample estimates are substituted for true population statistics — a process that often fails to converge in high-dimensional asset spaces. By reframing asset allocation as a sequential decision-making task within a **Markov Decision Process (MDP)**, we can develop dynamic policies that adapt to multivariate time-series states without the suboptimal constraints of rigid distributional assumptions.

The core identity that anchors this entire framework is:

$$
\max\; \mathbb{E}\!\left[\ln U(W)\right] = \max\; \mathbb{E}\!\left[\ln(1 + r_t)\right]
$$

Where,

| $\mathbb{E}[\cdot]$ | Expectation operator over the distribution of market outcomes.                                                    |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| $\ln(\cdot)$         | Natural logarithm function.                                                                                       |
| $U(W)$               | Utility function evaluated at wealth $W$. Under log-utility, $U(W) = \ln(W)$.                                    |
| $W$                  | Total investor wealth.                                                                                            |
| $r_t$                | Rate of return at time $t$. The gross return is $1 + r_t$.                                                        |
| $\max$               | Maximisation operator — the agent seeks the allocation policy that maximises this quantity.                        |

This identity bridges the gap between investor-specific utility and the market's maximum expected return. It establishes that maximising the expected utility of wealth is **mathematically equivalent** to maximising the expected logarithmic growth rate. This provides the foundation for **logarithmic growth optimisation**, ensuring the strategy prioritises long-term compounding over volatile, short-term additive gains. Every subsequent formula in this framework is ultimately in service of this identity.

---

## Optimal Investment Fraction — Merton/Kelly (Merton, 1969)

Modelling wealth as a *multiplicative* process is architecturally essential for long-term capital preservation. In financial environments, investment outcomes compound: a loss of 10% requires an 11.1% gain to recover — a reality that additive models fail to capture. To determine the *magnitude* of the investment action — defined here as the portfolio weight $\omega$ — we utilise the **Optimal Investment Fraction**.

$$
\pi^* = \frac{\mu - r_f}{R_a \, \sigma^2}
$$

Where,

| $\pi^*$  | Optimal fraction (weight $\omega$) of capital allocated to the risky asset (e.g., the CRSP index). |
| -------- | -------------------------------------------------------------------------------------------------- |
| $\mu$    | Expected return of the risky asset.                                                                |
| $r_f$    | Risk-free rate of return.                                                                          |
| $\sigma$ | Standard deviation (volatility) of the risky asset's returns.                                      |
| $R_a$    | Coefficient of constant relative risk aversion (CRRA parameter).                                   |
| $-$      | Subtraction — the numerator $\mu - r_f$ is the *risk premium*.                                     |

This formula serves as the primary **bet-sizing mechanism**. It calibrates the allocation magnitude by balancing the risk premium $\mu - r_f$ against volatility $\sigma^2$, scaled by the investor's risk tolerance $R_a$. When $R_a = 1$ (log-utility), this reduces to the classic **Kelly fraction**. The formula encodes a fundamental trade-off: higher expected excess return increases the optimal bet, but higher variance decreases it quadratically. This is the mechanism that prevents the agent from over-leveraging in volatile markets and under-investing in calm ones.

---

## Cumulative Wealth Transition

The evolution of wealth over a sequence of trading periods is inherently *multiplicative*. Each period's outcome scales the existing capital base, meaning that the order and magnitude of returns compound into the final result. This formula represents the **state dynamics** of wealth in a multiplicative environment.

$$
W_T = W_0 \prod_{t=1}^{T} (1 + r_t)^{\gamma}
$$

Where,

| $W_T$            | Total wealth at terminal time $T$.                                                                                                           |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| $W_0$             | Initial wealth at the start of the investment horizon.                                                                                       |
| $\prod_{t=1}^{T}$ | Product operator — iterates over all trading periods from $t = 1$ to $T$, multiplying each period's growth factor.                           |
| $r_t$             | Return at time $t$.                                                                                                                          |
| $1 + r_t$         | Gross return — the multiplicative growth factor for period $t$. A return of 5% gives a factor of 1.05.                                       |
| $\gamma$           | Discount factor used to weigh the significance of future rewards. Controls how strongly the agent values near-term vs. long-term compounding. |
| $T$               | Total number of trading periods in the horizon.                                                                                              |

This transition formula represents the **state dynamics** in a multiplicative environment. By incorporating the discount factor $\gamma$, the framework acknowledges that current state transitions influence the base for all subsequent compounding, highlighting the **high-stakes nature** of sequential allocation. A single poor decision early in the sequence permanently reduces the base from which all future growth compounds. This is the fundamental asymmetry of multiplicative dynamics that additive models fail to capture, and it is the reason log-utility (see Log-Utility Maximisation Identity) is the correct objective for long-horizon agents.

---

## CRRA Utility Function

The **Constant Relative Risk Aversion (CRRA)** utility function is the institutional standard for modelling professional investment objectives. It ensures the agent maintains a *consistent risk profile* regardless of total wealth scale — a billionaire and a retail investor with the same $R_a$ will allocate the same *fraction* of their wealth to risky assets.

$$
\begin{aligned}
U(W) &= \frac{W^{1-R_a}}{1-R_a} \\[6pt]
\ln U(W) &= (1 - R_a)\,\ln(W) - \ln(1 - R_a)
\end{aligned}
$$

Where,

| $U(W)$  | Utility of wealth. A concave function that encodes diminishing marginal utility — each additional unit of wealth is worth less than the last. |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| $W$     | Total investor wealth.                                                                                                                        |
| $R_a$   | Relative risk aversion parameter. When $R_a = 1$, the utility function reduces to $U(W) = \ln(W)$ (log-utility, the Kelly case).             |
| $\ln$   | Natural logarithm function.                                                                                                                   |
| $1-R_a$ | Exponent controlling the curvature of the utility function. Higher $R_a$ means stronger aversion to large losses.                             |

The logarithmic reduction demonstrates that maximising $\max\, \mathbb{E}[\ln U(W)]$ simplifies to maximising $\max\, \mathbb{E}[\ln(W)]$ (up to constants that do not affect the optimisation). This simplification is critical for RL agents — it replaces complex, non-linear utility targets with a stable objective in logarithmic space. This also resolves the **Sharpe Ratio Paradox**: while many researchers attempt to maximise the Sharpe Ratio directly as a reward, the Sharpe Ratio is *non-additive* and *time-lagged*. Log-utility is the superior reward signal because it is **additive** and **mathematically consistent** with long-term growth.

---

## Final Wealth Maximisation Objective

The terminal objective ties all previous formulations together. It expresses the agent's goal in its most complete form: maximise the expected log of terminal wealth, which is the sum of all per-period reward signals added to the log of initial capital.

$$
\max\; \mathbb{E}\!\left[\ln W_0 + \sum_{t=1}^{T} \text{Reward}_t\right] = \max\; \mathbb{E}\!\left[\ln W_T\right]
$$

Where,

| $\max$                | Maximisation operator over the agent's policy.                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| $\mathbb{E}[\cdot]$   | Expectation over market outcomes.                                                                                                    |
| $\ln W_0$             | Log of initial wealth — a constant that does not affect the optimal policy but anchors the scale.                                    |
| $\sum_{t=1}^{T}$      | Summation over all trading periods.                                                                                                  |
| $\text{Reward}_t$     | Per-period reward signal, defined as $\gamma \ln(1 + r_t)$ (see Logarithmic Reward).                                                |
| $\ln W_T$             | Log of terminal wealth — the quantity whose expectation the agent maximises.                                                         |
| $T$                   | Total number of trading periods.                                                                                                     |

This equation ensures the agent prioritises **sustainable capital growth**. By optimising for the log of terminal wealth, the agent naturally avoids catastrophic drawdowns, as the logarithmic function heavily penalises returns that approach zero (i.e., $\ln(x) \to -\infty$ as $x \to 0$). The equivalence between the left and right sides confirms that summing per-period log-rewards is a lossless decomposition of the terminal wealth objective — making the problem tractable for RL algorithms that operate on per-step reward signals.

---

## Logarithmic Reward

Standard RL algorithms — including those utilising the **Bellman equation** — are designed for *additive* reward signals. To optimise a multiplicative financial environment, we must engineer additive signals that faithfully represent compounding growth. The logarithmic reward is the transformation that achieves this.

$$
\text{Reward}_t = \gamma \, \ln(1 + r_t)
$$

Where,

| $\text{Reward}_t$ | Additive reward signal for period $t$. This is the per-step quantity the RL agent accumulates.      |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| $\gamma$            | Discount factor controlling the weight of future rewards relative to present ones.                  |
| $\ln(\cdot)$       | Natural logarithm function — converts the multiplicative growth factor into an additive increment.  |
| $1 + r_t$          | Gross return for period $t$ (see Cumulative Wealth Transition).                                     |
| $r_t$              | Net return at time $t$.                                                                             |

The log transformation converts the *product* of returns into a *sum* of rewards. This makes the multiplicative financial environment compatible with standard RL update mechanisms, allowing the agent to use simple summation to represent complex multiplicative compounding. Without this transformation, applying Bellman-style temporal difference updates to a multiplicative process would produce biased value estimates. The discount factor $\gamma$ allows the framework to smoothly interpolate between myopic ($\gamma \to 0$) and fully long-horizon ($\gamma \to 1$) optimisation.

---

## Sum of Rewards to Wealth Identity

This identity formally proves the equivalence between the agent's additive reward accumulation and the multiplicative growth of wealth. It is the mathematical bridge that guarantees an RL agent optimising summed log-rewards is, in fact, optimising for long-term compounding.

$$
\sum_{t=1}^{T} \text{Reward}_t = \ln \prod_{t=1}^{T} (1 + r_t)^{\gamma}
$$

Where,

| $\sum_{t=1}^{T}$    | Summation over all trading periods — the additive accumulation used by the RL agent.                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------ |
| $\text{Reward}_t$    | Per-period logarithmic reward, defined as $\gamma \ln(1 + r_t)$ (see Logarithmic Reward).                          |
| $\ln(\cdot)$         | Natural logarithm function.                                                                                        |
| $\prod_{t=1}^{T}$   | Product operator — the multiplicative accumulation of gross returns that defines actual wealth growth.              |
| $(1 + r_t)^{\gamma}$ | Discounted gross return for period $t$ (see Cumulative Wealth Transition).                                         |
| $T$                  | Total number of trading periods.                                                                                   |

This identity allows the RL agent to **optimise for the Kelly Criterion through additive updates**. It transforms a complex multiplicative growth problem into a manageable summation problem, aligning the agent's iterative learning procedure with long-term wealth accumulation goals. The identity holds because $\sum \gamma \ln(1+r_t) = \gamma \sum \ln(1+r_t) = \ln \prod (1+r_t)^\gamma$, by the standard logarithmic property $\ln(a^b) = b\ln(a)$ and $\sum \ln(a_t) = \ln \prod a_t$. This is not an approximation — it is an exact algebraic equivalence, which is why the log-reward formulation introduces no information loss relative to the true multiplicative objective.

---

## Bellman Equation — Q-Learning Update

For discrete allocation, the framework employs a **K-means clustered state-space discretisation** ($k = 50$) and solves for the optimal policy using the standard Q-learning update rule. The state space comprises 11 macroeconomic predictors (e.g., dividend-price ratio, inflation, T-bill rates) and lagged returns, mapped to a discrete action $a$ representing the portfolio weight $\omega$.

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
$$

Where,

| $Q(s, a)$                | Action-value function — the expected cumulative reward from taking action $a$ in state $s$ and following the optimal policy thereafter. |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| $\leftarrow$             | Update assignment — the left side is overwritten with the right side's value.                                                          |
| $\alpha$                 | Learning rate controlling the step size of each update. Determines how much new information overrides the old estimate.                 |
| $r$                      | Immediate reward received after taking action $a$ in state $s$ (see Logarithmic Reward).                                              |
| $\gamma$                  | Discount factor weighting the importance of future rewards relative to the present.                                                    |
| $\max_{a'}$              | Maximisation over all possible actions $a'$ in the next state — selects the best future action.                                        |
| $Q(s', a')$              | Estimated value of taking action $a'$ in the successor state $s'$.                                                                     |
| $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ | Temporal difference (TD) error — the discrepancy between the current estimate and the bootstrapped target.            |
| $s, s'$                  | Current state and successor state, respectively.                                                                                       |
| $a, a'$                  | Current action and candidate successor action, respectively.                                                                           |

While Q-learning provides **stability** through tabular convergence guarantees, it is limited by the granularity of discrete bins (e.g., $\omega \in \{0.0, 0.1, \dots, 1.0\}$). For institutional-grade precision, the framework transitions to the **Deterministic Policy Gradient** for continuous allocation (see Deterministic Policy Gradient). The Bellman equation remains foundational, however, as it defines the recursive value decomposition that all deep RL methods — including DDPG — build upon.

---

## Deterministic Policy Gradient (Silver et al., 2014)

For continuous allocation — where the portfolio weight $\omega$ takes any value in $[0, 1.5]$ (including leverage) — the framework uses the **Deep Deterministic Policy Gradient (DDPG)** architecture integrated with a **Time-series Dense Encoder (TiDE)**. TiDE is specifically designed to handle temporal dependencies in multivariate time series more effectively than LSTMs or CNNs.

$$
\nabla_\theta J = \mathbb{E}\!\left[\nabla_a Q_\phi(s, a)\,\nabla_\theta \pi_\theta(s)\right]
$$

Where,

| $\nabla_\theta J$       | Gradient of the objective function $J$ with respect to the actor's parameters $\theta$. This is the direction in which $\theta$ should be updated to improve the policy.  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| $\mathbb{E}[\cdot]$     | Expectation over the state distribution induced by the current policy.                                                                                                    |
| $\nabla_a Q_\phi(s, a)$ | Gradient of the critic's action-value function $Q_\phi$ with respect to the action $a$. Tells the actor *which direction* to adjust the action to increase expected value. |
| $Q_\phi$                 | Critic network parameterised by $\phi$, which estimates the expected cumulative reward for a given state-action pair. Uses a soft-updated target critic ($\tau = 0.005$).  |
| $\nabla_\theta \pi_\theta(s)$ | Gradient of the actor network's output with respect to its parameters $\theta$. Tells the optimiser how $\theta$ changes the action.                                |
| $\pi_\theta(s)$         | Actor network (the deterministic policy) that maps states $s$ directly to continuous actions $a$.                                                                         |
| $s$                     | Current state — the multivariate time-series observation (macroeconomic predictors, lagged returns).                                                                      |
| $a$                     | Action — the continuous portfolio weight $\omega \in [0, 1.5]$.                                                                                                           |

This architecture enables the agent to map complex states directly to **continuous weights** $\omega \in [0, 1.5]$. This model-free approach is superior to model-based methods, which often suffer from the *suboptimality* of a two-step "prediction then optimisation" process. The chain rule structure $\nabla_a Q \cdot \nabla_\theta \pi$ is what makes end-to-end gradient flow possible: the critic provides the value landscape, and the actor navigates it. The inclusion of leverage ($\omega > 1.0$) introduces a specific architectural trade-off: it increases absolute portfolio value but typically results in a lower Sharpe Ratio due to increased volatility.

---

## Logarithmic Utility — Evaluation Metric

To validate the framework's performance, the primary optimisation objective is measured directly as the sum of log-returns across the evaluation period. This metric captures **additive growth in log-space**, which is the faithful representation of multiplicative wealth accumulation.

$$
LU = \sum_{t=1}^{T} \ln(1 + r_t)
$$

Where,

| $LU$             | Logarithmic Utility — the cumulative log-return over the evaluation horizon. Higher values indicate stronger compounding performance.  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| $\sum_{t=1}^{T}$ | Summation over all trading periods in the evaluation window.                                                                          |
| $\ln(\cdot)$     | Natural logarithm function.                                                                                                           |
| $1 + r_t$        | Gross return for period $t$.                                                                                                          |
| $r_t$            | Net return at time $t$.                                                                                                               |
| $T$              | Total number of evaluation periods.                                                                                                   |

This is the **primary optimisation objective** expressed in evaluation form. Because $LU = \ln(W_T / W_0)$, it directly measures how many times the initial capital has been multiplied in log-space. It is additive across periods, making it straightforward to compute and compare across different evaluation windows. Unlike the Sharpe Ratio, $LU$ is *time-consistent* — the $LU$ of a combined period equals the sum of the $LU$ values for each sub-period.

---

## Portfolio Value — Evaluation Metric

Portfolio Value measures *total wealth accumulation* as a simple ratio of terminal to initial capital. It is the most intuitive performance metric and directly reflects the success of the compounding strategy.

$$
PV = \frac{W_{T}}{W_0}
$$

Where,

| $PV$   | Portfolio Value — the multiplicative factor by which initial wealth has grown. $PV = 2.0$ means the capital doubled. |
| ------ | -------------------------------------------------------------------------------------------------------------------- |
| $W_T$  | Terminal wealth at time $T$ (see Cumulative Wealth Transition).                                                      |
| $W_0$  | Initial wealth at the start of the evaluation period.                                                                |

Portfolio Value is the **raw multiplicative outcome** of the investment strategy. While $LU$ is the quantity the agent *optimises*, $PV$ is the quantity investors ultimately *care about*. The relationship between them is $PV = e^{LU}$, confirming that maximising log-utility (see Logarithmic Utility — Evaluation Metric) is equivalent to maximising portfolio value. $PV$ is reported alongside $LU$ because it provides immediate economic intuition: a $PV$ of 1.5 means a 50% total return, regardless of how many periods it took.

---

## Sharpe Ratio — Evaluation Metric (Sharpe, 1966)

The Sharpe Ratio evaluates **risk-adjusted performance** by normalising excess returns by their volatility. It answers the question: *how much return did the agent earn per unit of risk taken?*

$$
SR = \frac{\mathbb{E}_t\!\left[\rho_t - \rho_f\right]}{\sqrt{\text{Var}_t\!\left[\rho_t - \rho_f\right]}}
$$

Where,

| $SR$                                              | Sharpe Ratio — a dimensionless measure of risk-adjusted return. Higher is better.                                                          |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| $\mathbb{E}_t[\cdot]$                             | Time-series mean over the evaluation window.                                                                                               |
| $\rho_t$                                           | Portfolio return at time $t$.                                                                                                              |
| $\rho_f$                                           | Risk-free rate of return (e.g., T-bill rate).                                                                                              |
| $\rho_t - \rho_f$                                  | Excess return — the portfolio's return above the risk-free benchmark.                                                                      |
| $\text{Var}_t[\cdot]$                              | Time-series variance of excess returns over the evaluation window.                                                                         |
| $\sqrt{\cdot}$                                     | Square root — converts variance to standard deviation, giving the denominator the same units as the numerator.                             |

Empirical results for the DDPG-TiDE framework demonstrate a Sharpe Ratio of **1.13** (without leverage) and **0.99** (with leverage), both outperforming the passive Buy-and-Hold benchmark of **0.95**. The decrease in $SR$ with leverage ($\omega > 1$) illustrates a key architectural trade-off: leverage amplifies both returns *and* volatility, increasing Portfolio Value but degrading risk-adjusted performance. The Sharpe Ratio is evaluated over a **12-month rolling window**. While $SR$ is the industry-standard evaluation metric, it should *not* be used as the RL reward signal — it is non-additive and time-lagged, making it incompatible with temporal difference learning. Log-utility (see Logarithmic Utility — Evaluation Metric) is the correct training objective; $SR$ is used only for post-hoc evaluation.

---
