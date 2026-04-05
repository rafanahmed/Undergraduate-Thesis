Wealth accumulation is inherently a multiplicative process, but gradient-based optimization algorithms achieve greater numerical stability when operating on additive signals. 

The log-return transformation converts the path-dependent multiplicative growth into a path-independent additive quantity, which is essential for effective Back-propagation Through Time (BPTT) in recurrent networks. 

In simple terms, instead of working with products that can quickly become very large or very small, we take the logarithm so we are just summing numbers. This makes training neural networks more stable and lets us treat each period’s growth in a way that is easier for the learning algorithms to handle.
$$
r_t := \ln\frac{p_t}{p_{t-1}} = \ln\!\left(\mathbf{y}_t \cdot \mathbf{w}_{t-1}\right)
$$

Where,

| $r_t$              | Logarithmic rate of return (log-return) for period $t$. This is the per-period reward signal for the RL agent.                                               |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| $\ln(\cdot)$       | Natural logarithm function.                                                                                                                                  |
| $p_t / p_{t-1}$    | Gross return — the ratio of portfolio value at the end of period $t$ to its value at the start (see [[Portfolio Value Transition (Z. Jiang et al., 2017)]]). |
| $\mathbf{y}_t$     | Price relative vector at period $t$ (see [[Price Relative Vector (Z. Jiang et al., 2017)]]).                                                                 |
| $\mathbf{w}_{t-1}$ | Portfolio weight vector at the beginning of period $t$ (see [[Portfolio Value Transition (Z. Jiang et al., 2017)]]).                                         |
| $\cdot$            | Dot product between $\mathbf{y}_t$ and $\mathbf{w}_{t-1}$.                                                                                                   |

The log-return $r_t$ serves as the **immediate reward signal** for the reinforcement learning agent. 

Using the logarithm has three critical benefits: 
- It converts multiplicative compounding into additive accumulation, so maximizing $\sum r_t$ is equivalent to maximizing cumulative wealth.
- It stabilizes the reward signal by preventing the exponential blow-up common in compounding sequences
- It maximizes the sum of log-returns is mathematically equivalent to maximizing the **geometric mean** of returns 
	- This is the theoretically optimal criterion for long-term growth under the **Kelly criterion**. (idealized version without transaction costs; the cost-adjusted version appears in [[Log Return with Transaction Costs (Z. Jiang et al., 2017)]]).


**References:**
- [[NOTES - A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem]]
- [[A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem.pdf]]
- [[Price Relative Vector (Z. Jiang et al., 2017)]]
- [[Portfolio Value Transition (Z. Jiang et al., 2017)]]
