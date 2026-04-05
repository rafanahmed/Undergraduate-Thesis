Real-world trading incurs commission fees on both buying and selling. The Transaction Remainder Factor quantifies the fraction of portfolio value that survives after paying these fees during a rebalancing event. It acts as a multiplicative penalty on the portfolio's gross return.
$$
\mu_t = \frac{1}{1 - c_p \, w_{t,0}} \left[ 1 - c_p \, w'_{t,0} - (c_s + c_p - c_s \, c_p) \sum_{i=1}^{m} \left( w'_{t,i} - \mu_t \, w_{t,i} \right)_+ \right]
$$

Where,

| $\mu_t$          | Transaction remainder factor for period $t$. A scalar in $(0, 1]$ representing the fraction of portfolio value retained after paying commissions. $\mu_t = 1$ means no fees were incurred. |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| $c_s$            | Commission rate for selling an asset (e.g., 0.0025 for 0.25%).                                                                                                                             |
| $c_p$            | Commission rate for purchasing an asset.                                                                                                                                                   |
| $w_{t,0}$        | Target weight of the cash (risk-free) asset in the new allocation $\mathbf{w}_t$.                                                                                                          |
| $w'_{t,0}$       | Evolved weight of the cash asset after market movements (see [[Evolved (Post-Market) Portfolio Weights (Z. Jiang et al., 2017)]]).                                                         |
| $w'_{t,i}$       | Evolved weight of risky asset $i$ after market movements.                                                                                                                                  |
| $w_{t,i}$        | Target weight of risky asset $i$ in the new allocation.                                                                                                                                    |
| $(\cdot)_+$      | ReLU (rectified linear unit) function: $\max(0, x)$. Activates only when the evolved weight exceeds the target weight scaled by $\mu_t$, meaning the asset must be sold.                   |
| $m$              | Number of risky assets in the portfolio.                                                                                                                                                   |
| $\sum_{i=1}^{m}$ | Summation over all risky assets.                                                                                                                                                           |

The Transaction Remainder Factor is the mechanism by which the framework **penalizes excessive trading**. Its most notable mathematical property is that it is **self-referential**: 
- $\mu_t$ appears on both sides of the equation because the amount that must be sold depends on the post-fee portfolio value, which itself depends on how much was sold. 
	- This circular dependency means $\mu_t$ **cannot be solved analytically** and must instead be computed via an iterative fixed-point method where $\mu_t^{(k)} = f(\mu_t^{(k-1)})$, converging to a stable value. 
	- The ReLU function $(x)_+$ ensures that only assets being *sold* (where $w'_{t,i} > \mu_t w_{t,i}$) contribute to the commission cost. 
		- This factor is embedded directly into the reward signal (see [[Log Return with Transaction Costs (Z. Jiang et al., 2017)]]), forcing the RL agent to learn policies that balance return-seeking behavior against the real-world cost of portfolio turnover, effectively discouraging the *churning effect* where frequent rebalancing erodes capital.


**References**:
- [[NOTES - A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem]]
- [[A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem.pdf]]
- [[Evolved (Post-Market) Portfolio Weights (Z. Jiang et al., 2017)]]
- [[Log Return with Transaction Costs (Z. Jiang et al., 2017)]]
