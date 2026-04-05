The global objective function aggregates all per-period cost-adjusted log-returns into a single scalar that the agent maximizes via deterministic policy gradients. This is the quantity that drives the entire training process.
$$
R = \frac{1}{t_f} \sum_{t=1}^{t_f+1} r_t
$$

Where,

| $R$                  | Average logarithmic accumulated return over the entire trading session. This is the scalar objective the policy gradient maximises.                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $t_f$                | Total number of trading periods in the session.                                                                                                                                                                                             |
| $\sum_{t=1}^{t_f+1}$ | Summation over all trading periods from the first to the last.                                                                                                                                                                              |
| $r_t$                | Per-period log-return (either the idealised version from [[Logarithmic Rate of Return (Z. Jiang et al., 2017)]] or the cost-adjusted version from [[Log Return with Transaction Costs (Z. Jiang et al., 2017)]], depending on the setting). |

While final wealth $p_f$ is the ultimate economic goal, $R$ is the mathematical proxy the agent actually optimizes. The averaging by $\frac{1}{t_f}$ normalizes the objective across sessions of different lengths, making hyper-parameters and learning rates transferable. 

Under [[A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem.pdf]]'s Hypothesis 2 (**Zero Market Impact**) — the assumption that the agent's allocations $\mathbf{w}_t$ do not influence future price relatives $\mathbf{y}_{t+1}$:
- The environment becomes stationary with respect to the agent's actions. 
	- In this regime, the reward at each time step depends *only* on the current action and state, not on the agent's history.
	- This independence property is what enables the use of a **deterministic policy gradient** (rather than stochastic policy gradients typical of environments where actions alter future states) and allows the **fully-exploiting** training scheme: 
		- Each period's reward can be maximized independently, and the sum of individually maximized log-returns equals the maximum of the sum. 
		- This is the mathematical foundation that makes the Online Stochastic Batch Learning (OSBL) schema tractable.


**References**:
- [[NOTES - A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem]]
- [[A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem.pdf]]
- [[Logarithmic Rate of Return (Z. Jiang et al., 2017)]]
- [[Log Return with Transaction Costs (Z. Jiang et al., 2017)]]