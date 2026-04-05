The idealized log-return (see [[Logarithmic Rate of Return (Z. Jiang et al., 2017)]]) ignores trading fees. In practice, every rebalancing event erodes portfolio value. This cost-adjusted version of the log-return embeds the Transaction Remainder Factor directly into the reward signal, forcing the agent to internalize the trade-off between potential gains and frictional losses.
$$
r_t = \ln\!\left(\mu_t \; \mathbf{y}_t \cdot \mathbf{w}_{t-1}\right)
$$

Where,

| $r_t$              | Cost-adjusted logarithmic return for period $t$. This is the *actual* reward signal used during training.            |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| $\ln(\cdot)$       | Natural logarithm function.                                                                                          |
| $\mu_t$            | [[Transaction Remainder Factor (Z. Jiang et al., 2017)]]. Scales the gross return down to reflect commission losses. |
| $\mathbf{y}_t$     | Price relative vector at period $t$ (see [[Price Relative Vector (Z. Jiang et al., 2017)]]).                         |
| $\mathbf{w}_{t-1}$ | Portfolio weight vector at the beginning of period $t$.                                                              |
| $\cdot$            | Dot product between $\mathbf{y}_t$ and $\mathbf{w}_{t-1}$.                                                           |

This is the **production-grade reward function** used during RL training. By embedding $\mu_t$ inside the logarithm, the agent receives a strictly lower reward whenever it incurs transaction costs. 

The logarithmic wrapper ensures the penalty is proportional: large trades that trigger heavy commissions produce a significant drop in $r_t$, while minor rebalancing has a negligible effect. 

This formulation forces the agent to learn a **cost-aware policy**. It must weigh the expected benefit of rebalancing against the guaranteed cost of doing so. 

In high-friction environments (e.g., cryptocurrency markets with 0.25% commission), this distinction is critical: traditional models like PAMR and OLMAR yielded negative returns precisely because they ignored or underweighted transaction costs. 

The framework's strong backtest performance is directly attributable to this cost-adjusted reward signal.


**References**:
- [[NOTES - A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem]]
- [[A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem.pdf]]
- [[Logarithmic Rate of Return (Z. Jiang et al., 2017)]]
- [[Transaction Remainder Factor (Z. Jiang et al., 2017)]]
- [[Price Relative Vector (Z. Jiang et al., 2017)]]