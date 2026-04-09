
[[Deep Reinforcement Learning for Optimal Asset Allocation Using DDPG with TiDE.pdf]]
# **Abstract**:
**Constraints**:
This paper highlights the problem that comes with optimally allocating capital between risky and risk-free assets and this is due to the inherent volatility regimes in financial markets.

Some conventional methods rely on
- Strict Distributional assumptions
	- *Context*:
		- This means you pick a math formula upfront (usually a bell curve / Gaussian) and assume all future returns will follow it.
		- The bell curve only needs two numbers:
			- the average return
			- how spread out it is.
		- Real markets, however, have extreme crashes and rallies way more often than a bell curve predicts (fat tails), returns can be lopsided (skewness), and calm/chaotic periods cluster together (volatility clustering).
			- So any portfolio built on the bell-curve assumption looks great on paper but breaks in practice. Model-free RL sidesteps this entirely because it learns what to do from experience, no assumed distribution needed.
- Non additive reward ratios
	- *Context*:
		- Think of the Sharpe Ratio $\frac{R_p - R_f}{\sigma_p}$ :
			- it's a ratio is: return divided by risk.
			- You can't just add ratios across time periods the way you can add up scores.
				- Ex. If you got a Sharpe of 1.5 in January and 2.0 in February, the combined Sharpe is NOT 3.5 because ratios don't work that way.
				- RL needs rewards it can sum up over time ($\sum_t r_t$), so ratio-based metrics are fundamentally incompatible.
				- The fix: **use log-returns instead**:
					- $\log(W_T) = \sum_t \log(1 + r_t)$ - this IS additive. You can sum them across periods and it just works with RL.

The issue with this is that they are not robust enough and are not exactly applicable to the wide range of investment goals an individual agent or a trading entity has.

**Addressing the constraints**:
In the paper, R. Liu et al., formulates an optimal two-asset allocation problem as:
- *Context*:
	- "Two-asset allocation" = you have two buckets:
		- one risky (e.g., stocks)
		- one safe (e.g., cash/T-bills)
			- Each period you decide what fraction $w \in [0,1]$ goes into the risky bucket, and the rest stays safe.
		- This is the simplest meaningful portfolio problem:
			- Every real allocation question ultimately reduces to some version of "how much risk do I take on vs. keep safe?"
- A sequential decision-making task
	- Within a Markov Decision Process (MDP)

This framework enables application of reinforcement learning (RL) in developing:
- Dynamic policies based on simulated financial scenarios regardless of prerequisites

**Method**:
This paper utilizes the **Kelly Criterion** to balance:
- Immediate reward signals against long-term investment objectives
	- *Context*:
		- After each portfolio rebalance, the agent gets a single number telling it "how did that one move go?"
			- that's the immediate reward.
			- Here it's $r_t = \log(W_{t+1}/W_t)$, derived from the Kelly Criterion.
				- "Immediate" just means this one step, as opposed to the total cumulative goal.
				- What makes Kelly special: it naturally prevents the agent from being too greedy (going all-in and risking ruin) while still pushing it to grow wealth; the mathematical sweet spot between aggression and survival.

With the novel approach in integrating the following RL framework for continuous decision-making:
- Time-series Dense Encoder (TiDE)
	- Into the Deep Deterministic Policy Gradient (DDPG)

This DDPG x TiDE synthesis and its results are compared with 2 things:
- a simple discrete Q-Learning RL framework
- a passive buy-and-hold investment strategy

**Results**:
DDPG-TiDE:
- Outperformed Q-learning
- Generated higher risk adjusted returns than buy-and-hold strategy

**Conclusion**
"These findings suggest that tackling the optimal asset allocation problem by integrating TiDE within a DDPG reinforcement learning framework is a fruitful avenue for further exploration."

# Intro
This paper presents something alternative to an "efficient frontier" or a  "determined exact tangency point of the Capital Market Line"
- *Context*:
	- Imagine plotting every possible portfolio on a graph:
		- x-axis is risk, y-axis is return.
		- The **efficient frontier** is the top edge of that cloud:
			- the best return you can get for any given risk level.
		- Now draw a straight line from the risk-free rate $R_f$ that just barely touches (is tangent to) this curve:
			- that's the **Capital Market Line (CML)**.
		- The point where it touches is the single "best" risky portfolio.
			- In theory, every investor should just mix that one portfolio with cash.
				- However, computing that tangent point requires knowing the exact distribution of returns, which we don't have.
				- This paper's alternative suggests that we should let an RL agent figure out the optimal mix by learning from experience instead.
- The focus is rather on identifying the investor-specific optimal allocation of capital in order to maximize long-term portfolio performance (assuming market portfolio is observable and broadly representative of assets available to be invested into)

Classical optimal portfolio methods are mean variance optimization (MVO), which was proposed to construct portfolios.
- This has been used to estimate the distribution of returns on assets and to select valuable portfolios.
	- *Context*:
		- "Distribution of returns" = if you could see every possible outcome for a stock and how likely each one is, that's the distribution.
		- Formally it's the density function $f(r)$, described by its moments (mean, variance, skewness, kurtosis, ...).
		- If you assume a bell curve, only mean and variance matter
			- However, real markets have richer structure (fat tails, asymmetry) that gets thrown away under that assumption.
	- However, because the true distribution of returns is unknown, the population is estimated and substituted by the sample, and then the optimal portfolio weights are selected based on the conclusions derived from there.
		- *Context*:
			- The "true" distribution is what you'd see with infinite data under perfectly stable conditions, a theoretical ideal we can never actually observe.
			- All we have is a limited chunk of historical data (the sample). So we compute the average return and covariance from that sample and pretend it's the truth and this is what we call the **plug-in estimation**.
				- The problem: averages computed from small samples are noisy, so the "optimal" portfolio you derive is mostly fitting noise, not real patterns.
				- This is exactly why model-free RL is attractive since it learns what actions to take directly from experience, no distribution estimation needed.
	- So, portfolio weights that maximize utility depends on:
		- First-order and second-order returns and their moments
			- *Context*:
				- First-order moment = the average return $\mu_i = E[R_i]$ — "where is the center?"
				- Second-order moments = how spread out and how correlated: variance $\sigma_i^2$ (how much one asset bounces around) and covariance $\sigma_{ij} = \text{Cov}(R_i, R_j)$ (do two assets move together or opposite?).
					- MVO needs both to compute optimal weights: $w^* = f(\mu, \Sigma, \lambda)$. Both must be estimated from limited data however, and average returns are notoriously hard to pin down with small samples.
		- Investor's risk aversion coefficient.
- This method is known as plug-in estimation.
	- The limitation of this approach is that it assumes a consistent estimate of population statistics,
		- But the estimation results will not converge when the assets in the portfolio gradually increase.

*NOTE: Can an analogy in current RL be made with this or something equivalent to this? What is a paper that goes into that?*
- Exact same thing happens in RL.
	- Model-based RL tries to learn a full map of "if I'm in state $s$ and do action $a$, where do I end up?"
		- the transition dynamics $P(s'|s,a)$.
		- As the state/action space gets bigger, the number of things to estimate explodes, and limited experience can't keep up.
			- Model-free RL (like policy gradients) dodges this by skipping the map entirely and just learning "what action works best here?" directly from trial and error. Relevant references:
				- Jiang et al. (2017) — argues RL bypasses distributional estimation for portfolio management
				- Moerland et al. (2023), "Model-Based RL: A Survey" — model error compounding sections are directly analogous
				- Henderson et al. (2018), "Deep RL that Matters" — estimation fragility in deep RL

Hyperparameters in ML models can be tuned to maximize expected out-of-sample utility OR minimize a predefined risk function, given
- The first-order and second-order moments and
- Functional form of the return distribution are known
- *Context*:
	- Two ways to frame the goal:
		- **Maximize out-of-sample utility**:
			- pick weights that perform best on future data the model hasn't seen, using a utility function $U(W)$ like log-utility or exponential utility to score outcomes. This guards against overfitting to the past.
		- **Minimize a predefined risk function**:
			- instead of chasing returns, directly minimize a risk measure:
				- variance, Value-at-Risk (VaR), CVaR, or max drawdown.
			- One says "maximize the good," the other says "minimize the bad." :
				- Both need you to know the moments and distribution shape, which circles back to the same plug-in estimation problems.

To do this, a two step approach is done:
1. A class of equations with hyperparameters is selected, the samples are trained and then the hyperparameters are then estimated.
2. Hyperparameters are then tuned with data that is out-of-sample on the validation set in order to get the best performance.

Portfolio selection methods in ML considers integrating:
- Utility maximization into statistical problems.
	- This method is known as the one-step procedure or parametric portfolio weight approach.
	- It is not necessary to know the distribution assumptions of the returns and, instead, only the investment utility function and the weight function are needed.
- Ex. Maximum Sharpe Ratio Regression (MSRR)

In the reinforcement learning context:
- RL uses reward signals from a simulated market environment to iteratively guide learning; widely used in asset allocation problems
- In this paper, the novel introduction:
	- Utility equation based on the Kelly Criterion
	- In which the rewards is determines through formulaic derivation
		- From these, the optimal policy problem within the MDP framework is given
- Additionally the paper then incorporates the Time- series Dense Encoder (TiDE) into the Deep Deterministic Policy Gradient (DDPG) mechanism to enhance the handling of temporal relationships in multivariate time series within the state space.

# Methods and Materials:

## Kelly Strategy & MDP Simulation

1971 paper by [Merton](linkinghub.elsevier.com/retrieve/pii/002205317190038X) explored trading stocks bonds within his portfolio model to optimize the utility of investors

2010 paper by [MacLean](https://www.worldscientific.com/doi/abs/10.1142/9789814293501_0038) equated the following:
- Investor's optimal expected utility = market's maximum expected return
	- The investment objective that was labeled for the portfolio was the **Kelly Criterion**:
	- *Context*:
		- The Kelly Criterion says: if you want the best long-run growth, maximize the *expected log* of your wealth each period.
			- This is because log turns multiplicative compounding into additive sums, and maximizing the sum of logs = maximizing the geometric growth rate.
				- See **MATHEMATICS §1** for the formal expression.

Based on derivation using **Itô's Lemma**, the optimal investment fraction in the risky asset is $\pi^*$
- *Context*:
	- This is the closed-form Kelly fraction under Gaussian returns.
		- Higher excess return $(\mu - r_f)$ → invest more; higher variance $(\sigma^2)$ or higher risk aversion $(R_a)$ → invest less.
			- See **MATHEMATICS §1** for the formal derivation.
	- **Limitation**: the distribution of risky assets in real financial markets is complex and unknown, and indicators are not limited to price data.
		- So the optimal solution $\pi^*$ is limited by various conditions and cannot adapt to dynamically adjusting markets.
	- **Solution**: Use MDP + RL algorithms for the practical implementation of the RL Kelly strategy.

**MDP Definition**:
An MDP provides a mathematical framework for sequential decision making under uncertainty, defined by the tuple $(S, A, P, R, \gamma)$:
- $S$ = state space
- $A$ = action space
- $P(s' | s, a)$ = transition probabilities
- $R(s, a)$ = reward function
- $\gamma$ = discount factor

In financial markets, the MDP maps as follows:
- **States**: market conditions (historical macro factors, excess returns, volatilities)
- **Actions**: trading decisions (buy, sell, hold, or adjusting risk-asset weights)
- **Rewards**: long-term value (e.g., log-return)

The agent explores strategies through repeated simulated trials, continuously refining portfolio allocations through reward feedback, striving to maximize cumulative logarithmic returns while respecting risk constraints. RL avoids strict rule-based heuristics and adapts dynamically to changing market conditions.

## CRRA Utility and Reward Setting

The reward in this RL framework must serve two purposes:
- Provide **immediate feedback** on the chosen action
- Embody the **longer-term risk-return trade-off**

This is encoded through the **CRRA (Constant Relative Risk Aversion) utility function**:
- *Context*:
	- CRRA is a widely used assumption in economic models and is assumed in Merton's model.
		- When risk aversion $R_a \to 1$, CRRA reduces to logarithmic utility $U(W) = \ln(W)$.
		- The paper shows that maximizing CRRA utility is equivalent to maximizing expected log-wealth:
			- The $-\ln(1-R_a)$ constant drops out of the optimization, and since $(1-R_a) > 0$ the sign doesn't flip.
				- So: **max CRRA utility = max expected log-wealth**.
					- This is the mathematical bridge between CRRA utility theory and the **Kelly Criterion**.
						- See **MATHEMATICS §2** for the full derivation (Equations 1–3).

The **reward** is then set as a discounted log-return: $\text{Reward}_t = \gamma \ln(1 + r_t)$
- *Context*:
	- This is the key derivation of the paper.
		- Optimizing the expected sum (or discounted sum) of rewards aligns with maximizing the expected future log wealth = the **Kelly Criterion**.
			- Each action's return-risk balance (via the immediate log-return feedback) contributes to the long-run objective of sustainable capital growth.
				- This **unifies immediate step-by-step reward signals with a principled, long-horizon utility measure**.
					- See **MATHEMATICS §3** for the full derivation chain from reward to wealth maximization (Equations 4–7).

## Q-Learning and Bellman Equation

Since the true transition dynamics of time series is typically unknown, **Q-learning** uses **temporal difference (TD) learning** to iteratively approximate the optimal action-value function $Q^*(s,a)$ using sampled experience $(s, a, r, s')$.
- *Context*:
	- The **Bellman equation** is the foundation of Q-learning.
		- It provides a recursive relationship: the value of a state-action pair = immediate reward + expected discounted return of the optimal future actions.
			- The Q-learning update rule uses:
				- a learning rate $\alpha$ that controls magnitude
				- a discount factor $\gamma$ that weighs future rewards
				- a **TD target** $r + \gamma \max_{a'} Q(s', a')$ representing the optimal future return estimate
			- This update ensures Q-learning converges to $Q^*(s,a)$ with adequate exploration and appropriate scheduling of learning rates.
				- See **MATHEMATICS §4** for the Bellman equation and update rule (Equations 8–9).

**Implementation in this paper**:
- **State space**: K-means clustering ($k=50$) on key market features → finite set of cluster labels as observed states
	- *Context*:
		- Instead of feeding raw 18-dimensional feature vectors, they quantize the state space into 50 discrete clusters.
			- This makes the Q-table tractable but loses nuance.
- **Action space**: Discretized portfolio weights $\omega \in \{0.0, 0.1, \ldots, 1.0\}$
	- *Context*:
		- 11 possible weight values, finer than buy/sell/hold but still coarse.
			- Each weight $\omega$ determines the fraction in the risky asset.
- **Baseline**: Buy-and-hold strategy — invest all funds in the market index at the beginning and hold until the end — as a passive baseline with fixed $\omega = 1.0$.

## DDPG with TiDE

Q-learning is a value-based RL algorithm designed for discrete state-action spaces, but it suffers from two major challenges:
- The Q-table becomes impractically large as the number of clusters increases
- Action selection by maximizing Q-values entails a costly search over all discrete bins
	- *Context*:
		- To address these limitations, policy-based methods were introduced, allowing RL agents to learn a policy $\pi(a|s)$ directly rather than relying on Q-values.

**DDPG** (Deep Deterministic Policy Gradient) extends Q-learning ideas by:
- Combining value function learning (Q-learning) with policy optimization
- Suitable for **continuous action spaces**
- Learning a **deterministic policy** $a = \pi_\theta(s)$ that maps each multivariate time series state directly to a continuous action
	- Avoids expensive search for action selection

**Actor-Critic Framework**:
- **Actor network** $\pi_\theta(s)$: outputs portfolio weights
- **Critic network** $Q_\phi(s, a)$: evaluates them
	- *Context*:
		- The actor is updated using the **deterministic policy gradient**, ensuring selected actions maximize the critic's Q-value.
			- See **MATHEMATICS §5** for the policy gradient formula (Equation 10).
		- The critic is updated using Bellman's equation, employing a **soft-updated target critic** $Q_{\phi'}$ ($\tau = 0.005$) to enhance stability.
			- Soft update means: $\phi' \leftarrow \tau \phi + (1-\tau)\phi'$.
				- Instead of hard-copying the network periodically, you slowly blend the learned weights into the target network.
					- This prevents the target from changing too fast, which stabilizes training.

**TiDE Architecture Integration**:
- Historical features are passed to a **two-layer TiDE encoder**:
	- Flattens inputs → projects into a latent representation using fully connected layers
	- Followed by **residual blocks** with ReLU activation and **layer normalization**
- **TiDE Actor**: Maps encoded state → scalar action via final linear transformation + **sigmoid activation** → portfolio weights $\omega \in [0, 1.0]$
- **TiDE Critic**: Concatenates TiDE encoder output with the TiDE actor's action → processes through two fully connected layers with ReLU activations → outputs Q-value
	- *Context*:
		- TiDE (Time-series Dense Encoder) is the novel contribution.
			- It replaces standard MLP layers in DDPG with a time-series-aware encoder that captures temporal dependencies among heterogeneous financial indicators.
				- This is the first known integration of TiDE within a DDPG RL framework.

**Exploration**: DDPG incorporates **Ornstein-Uhlenbeck (OU) noise** to facilitate exploration, since the policy network is inherently deterministic.

**Replay Buffer**: A multi-step replay buffer stores transitions $(s, a, r, s')$, stabilizes training by breaking correlations in consecutive data and allowing mini-batch updates.

**Hyperparameter Tuning**: Grid-search is used to explore the candidate range of hyperparameters.

## Dataset and Experiments

**Dataset**:
- Monthly time series on:
	- Market return rate $r_t$ (Mkt)
	- Risk-free asset return rate $r_f$ (RF)
- Period: **1927 to 2019**
- Split:
	- Training: **1927–1957**
	- Validation: **1958–1988**
	- Test: **1989–2019**
- Daily returns are retrieved to compute the **monthly realized volatilities**
- Source: [Fama/French website](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) (Fama/French 3 Factors csv)

**Features (18 total per timestep)**:
At each timestep, the agent observes:
1. The 18 features including the current month and previous 11 months
2. Logarithmic returns of the past 12 months

**11 Macroeconomic and Financial Predictors** (Table 1):

| Predictor | Description |
|-----------|-------------|
| dp | dividend-price ratio |
| ep | earnings-price ratio |
| bm | book-to-market ratio |
| ntis | net equity expansion |
| tbl | Treasury-bill rate |
| tms | term spread |
| dfy | default spread |
| infl | inflation |
| corpr | high-quality corporate bond rate |
| ltr | long-term rate of return |
| svar | stock variance |

Additional variables:
- One-month lagged excess return
- One-to-three-month lags of an enhanced measure of the **payout yield** (from [Goyal et al., 2024](https://sites.google.com/view/agoyal145))
- One-to-three-month lags of the realized squared monthly volatilities

**Models Compared**:
1. **Q-learning** (discrete state + discrete action):
	- Without leverage: $\omega \in \{0.0, 0.1, \ldots, 1.0\}$
	- With 50% leverage: $\omega \in \{0.0, 0.15, \ldots, 1.5\}$
2. **DDPG-TiDE** (continuous action):
	- Without leverage: $\omega \in [0.0, 1.0]$
	- With 50% leverage: $\omega \in [0.0, 1.5]$
3. **Buy-and-hold**: constant action $\omega = 1.0$

**Performance Metrics** (Table 2):

| Metric | Calculation |
|--------|-------------|
| Logarithmic Utility (cumulative rewards) | $LU = \ln(W_{t+1}) = \sum_{t=1}^{T} \ln(1+r_t) = \sum_{t=1}^{T} \text{reward}_t$ |
| Portfolio Value (cumulative returns) | $PV = \frac{W_{t+1}}{W_0}$ |
| Sharpe Ratio (12-month rolling window) | $SR = \frac{E_t[\rho_t - \rho_f]}{\sqrt{\text{var}_t[\rho_t - \rho_f]}}$ |

- *Context*:
	- Logarithmic Utility and Portfolio Value follow identical trends since the reward is a logarithmic transformation of wealth.
	- The Sharpe Ratio is calculated over a 12-month rolling window, providing a time-varying measure of risk-adjusted performance.

# Results and Discussion (Test Period: 1989–2019)

**Portfolio Value** (Fig. 1):
- **Q-learning** strategies: steady, shallow increase over long periods (especially before 2008)
	- Final wealth without leverage: **3.88**
	- Final wealth with 50% leverage: **4.65**
	- Much lower than buy-and-hold
- **Buy-and-hold**: final portfolio value = **17.77**
- **DDPG-TiDE** strategies: closely follow buy-and-hold
	- Without leverage: **14.85** (slightly below buy-and-hold)
	- With 50% leverage: **21.06** (slightly above buy-and-hold)
- *Context*:
	- Q-learning largely follows a **risk-averse strategy** (predominantly investing in the risk-free asset)
	- DDPG-TiDE largely follows a **risky strategy** (predominantly investing in the market index)

**Sharpe Ratio** (Fig. 2 — 12-month rolling window):
- **Q-learning**: much lower average Sharpe ratios; zero for long periods (e.g., 1992–2007)
- **DDPG-TiDE**: closely tracks buy-and-hold
	- Without leverage: average Sharpe ratio = **1.13**
	- With leverage: average Sharpe ratio = **0.99**
	- **Buy-and-hold**: average Sharpe ratio = **0.95**
- Both DDPG-TiDE values are **slightly higher than buy-and-hold**, indicating active DDPG-TiDE strategies generate slightly better **risk-adjusted returns** than a passive strategy.
- *Context*:
	- While leverage allows DDPG-TiDE to accumulate the highest returns, the increased risk results in a slightly lower average Sharpe ratio compared to the no-leverage version.

**Portfolio Weights** (Fig. 3):
- **Q-learning**: invests fully in the risk-free asset for long periods (e.g., 1992–2007) — coincides with zero Sharpe ratio periods and shallow returns
- **DDPG-TiDE**: often invests fully in the risky asset during the same period → more erratic Sharpe ratio that correlates with buy-and-hold
	- With leverage: often takes values $\omega = 1.5$ to take advantage of the bull market post-2008 → outstrips buy-and-hold returns

# Conclusions

**Key findings**:
- RL is used as a **model-free approach** to optimize portfolio allocation between a risk-free asset and a market index
- **Novel contribution**: introduction of a TiDE-based architecture into the DDPG RL framework
- **DDPG-TiDE outperforms Q-learning** in all metrics
- DDPG-TiDE is able to **generate more profits than buy-and-hold** by applying leverage during bull periods
- DDPG-TiDE generates the **highest average Sharpe ratio**

**Future work**:
- Incorporate a wider range of RL frameworks and methodologies
- Practical **trading costs** will be taken into account
- **Penalty mechanisms** will be introduced in both algorithm design and code implementation
- Long-term goal: **multi-agent modeling approach** to develop intelligent collectives of RL agents that offer **user-specific personalized investment strategies** (AI for Collective Intelligence research strategy)

**Funding**: UKRI EPSRC Grant No. EP/Y028392/1: AI for Collective Intelligence (AI4CI)

---
# MATHEMATICS

In the mathematical framework of this paper, the central challenge is bridging classical portfolio theory (Kelly criterion, CRRA utility) with reinforcement learning (MDP rewards, Bellman equations). The derivation chain establishes that maximizing RL cumulative rewards is mathematically equivalent to maximizing long-term log-wealth — the Kelly criterion. This section traces that derivation from utility theory through reward design to the policy optimization that drives the DDPG-TiDE agent.

### 1. The Kelly Criterion and Optimal Risky Fraction

The Kelly criterion defines the portfolio's investment objective as maximizing expected log-utility of wealth:

$$\max\, E[\ln U(W)] = \max\, E[\ln \text{Return}]$$

where $\text{Return} = 1 + r_t$ and $r_t$ is the rate of risky asset return $\sim N(\mu, \sigma^2)$. This formulation, established by MacLean [15], equates the investor's optimal expected utility with the market's maximum expected return.

Using **Itô's Lemma**, the optimal investment fraction in the risky asset is derived as:

**Optimal Risky Fraction ($\pi^*$):**
$$\pi^* = \frac{\mu - r_f}{R_a \sigma^2}$$

where $\mu$ is the expected return of the risky asset, $r_f$ is the return rate of the risk-free asset, $\sigma$ is the standard deviation, and $R_a$ is the coefficient of constant risk aversion (CRRA). This closed-form solution represents the theoretical ideal: higher excess return $(\mu - r_f)$ drives larger allocation to the risky asset, while higher variance $(\sigma^2)$ or greater risk aversion $(R_a)$ reduces it. However, since the true distribution of risky assets is complex and unknown, this analytical solution cannot adapt to dynamically adjusting market environments — motivating the MDP-based RL approach.

### 2. CRRA Utility and the Log-Utility Bridge

The Constant Relative Risk Aversion (CRRA) utility function encodes both immediate feedback and the longer-term risk-return trade-off:

**CRRA Utility Function:**
$$U(W) = \frac{W^{1-R_a}}{1-R_a} \tag{1}$$

where $W$ is wealth and $R_a$ is relative risk aversion (constant). CRRA is a widely used assumption in economic models and is assumed in Merton's model [18]. When $R_a \to 1$, this reduces to logarithmic utility $U(W) = \ln(W)$.

When this expression reduces to logarithmic utility:

$$\ln U(W) = (1-R_a)\ln(W) - \ln(1-R_a) \tag{2}$$

Maximizing the expectations of both sides, since $0 < R_a < 1$:

$$\max\, E[\ln U(W)] = \max\, E[(1-R_a)\ln(W) - \ln(1-R_a)] = \max\, E[\ln(W)] \tag{3}$$

The term $-\ln(1-R_a)$ is a constant that drops out of the optimization. Since $(1-R_a) > 0$, the sign does not flip. This establishes the critical equivalence: **maximizing CRRA utility = maximizing expected log-wealth**. This is the mathematical bridge between CRRA utility theory and the Kelly Criterion.

### 3. Reward Design and Wealth Maximization

Building on the log-utility equivalence, the per-step reward is defined as a discounted log-return:

**Reward Signal:**
$$\text{Reward}_t = \gamma \ln(1 + r_t) \tag{4}$$

The total accumulated reward over $T$ periods converts from a sum of logs to the log of a product:

**Total Reward:**
$$\sum_{t=1}^{T} \text{Reward}_t = \sum_{t=1}^{T} \ln(1+r_t)\gamma = \ln \prod_{t=1}^{T} (1+r_t)^\gamma \tag{5}$$

This gives the wealth at time $T$:

**Terminal Wealth:**
$$W_0 \prod_{t=1}^{T}(1+r_t)^\gamma = W_T, \quad \text{assuming } W_0 = 1 \tag{6}$$

Therefore, to maximize expected wealth:

**Wealth Maximization Objective:**
$$\max\, E\left[\ln W_0 + \sum_{t=1}^{T} \text{Reward}_t\right] = \max\, E\left[\ln W_0 \prod_{t=1}^{T}(1+r_t)^\gamma\right] = \max\, E[\ln W_T] \tag{7}$$

This corresponds exactly to the log-utility perspective $\max\, E[\ln U(W)]$. Optimizing the expected sum of rewards aligns with maximizing the expected future log wealth — the Kelly Criterion. Each action's return-risk balance, via the immediate log-return feedback, contributes to the long-run objective of sustainable capital growth. This derivation chain is the paper's key contribution: it **unifies immediate step-by-step reward signals with a principled, long-horizon utility measure**.

### 4. The Bellman Equation and Q-Learning Updates

Since the true transition dynamics of time series are typically unknown, Q-learning uses temporal difference (TD) learning to iteratively approximate the optimal action-value function. The Bellman equation provides the recursive foundation:

**Bellman Equation:**
$$Q^*(s,a) = E\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right] \tag{8}$$

The Bellman equation expresses the value of a state-action pair as the immediate reward plus the expected discounted return of the optimal future actions. The practical Q-learning update rule follows:

**Q-Learning Update:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s', a') - Q(s,a)\right] \tag{9}$$

where $\alpha$ is the learning rate controlling the magnitude of each update, $\gamma$ is the discount factor weighing future rewards, and $r + \gamma \max_{a'} Q(s', a')$ is the TD target representing the optimal future return estimate. This iterative process ensures convergence to $Q^*(s,a)$ with adequate exploration and appropriate scheduling of learning rates.

### 5. The Deterministic Policy Gradient

Within the DDPG actor-critic framework, the actor network $\pi_\theta(s)$ outputs portfolio weights and the critic network $Q_\phi(s,a)$ evaluates them. The actor is updated using the deterministic policy gradient:

**Deterministic Policy Gradient:**
$$\nabla_\theta J = E\left[\nabla_a Q_\phi(s,a) \nabla_\theta \pi_\theta(s)\right] \tag{10}$$

This gradient ensures that the actor's selected actions move in the direction that maximizes the critic's Q-value. The critic is updated using the Bellman equation with a soft-updated target critic $Q_{\phi'}$ ($\tau = 0.005$), where $\phi' \leftarrow \tau \phi + (1-\tau)\phi'$, to enhance stability by slowly blending learned weights into the target network rather than hard-copying periodically.

---
### **1. Problem:**
This paper addresses the **optimal two-asset allocation problem** — how to dynamically distribute capital between a risk-free asset and a market index (CRSP) to maximize long-term portfolio performance. It highlights that conventional methods rely on strict distributional assumptions and non-additive reward ratios (e.g., Sharpe Ratio), which limit robustness. The paper proposes a **model-free reinforcement learning** approach that bypasses these limitations by learning directly from simulated market interactions.

### **2. Setup:**
* **Multiplicative:** The framework involves multiplicative wealth dynamics where portfolio value compounds over time: $W_T = W_0 \prod_{t=1}^{T}(1+r_t)^\gamma$. The reward is designed as log-return to convert this multiplicative process into an additive one compatible with RL.
* **Continuous:** DDPG-TiDE operates in a **continuous action space** where portfolio weights $\omega \in [0, 1.0]$ (or $[0, 1.5]$ with leverage) are selected directly by a deterministic policy, rather than discretized into bins.

### **3. Key Idea:**
The key idea is the integration of the **Time-series Dense Encoder (TiDE)** into the **DDPG** actor-critic framework for portfolio optimization. The reward is grounded in the **Kelly Criterion** via CRRA utility, ensuring that maximizing cumulative RL rewards is mathematically equivalent to maximizing long-term log-wealth. TiDE captures temporal dependencies in multivariate financial time series within the state space, enabling richer representation of market conditions than standard MLP architectures.

### **4. Assumptions:**
* **Stationary:** The framework assumes the MDP environment (trained on 1927–1957 historical data) generalizes to out-of-sample periods, though the agent adapts through learned policies.
* **Observable Market Portfolio:** The CRSP index is taken as a representative proxy for the aggregate U.S. equity market — the true market portfolio remains unobservable.
* **No Transaction Costs:** The current implementation does not account for trading costs, slippage, or market impact from the agent's actions.
* **CRRA Utility:** Assumes constant relative risk aversion ($0 < R_a < 1$), which reduces to log-utility for the Kelly Criterion derivation.

### **5. Limitation:**
The method does not incorporate **transaction costs or penalty mechanisms**, which limits the practicality of the strategy in real-world deployment. The study is restricted to a **two-asset allocation** (one risky, one risk-free), which does not generalize to multi-asset portfolio problems. The Q-learning baseline uses a coarse K-means discretization ($k=50$) that loses nuance in the state representation. Additionally, the framework assumes **no market impact** from the agent's actions and does not address **non-stationary environments** where market dynamics shift fundamentally over time.

### **6. Relation to my idea:**


---
### Integration:

* **Problem:** The RL framework for two-asset allocation (like this paper) is a **good fit for understanding how Kelly-optimal rewards interact with multiplicative wealth dynamics**, but it does not focus on **adaptive decision sizing under uncertainty** — which may be a gap to address.

* **Limitation:** The primary limitation is that the paper assumes **stationary environments**, uses a **simplified two-asset setup**, and does not integrate **transaction costs or penalty mechanisms** for real-world practicality.
