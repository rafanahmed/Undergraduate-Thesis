[Paper](obsidian://open?vault=undergrad-thesis&file=1706.10059v2.pdf)
### Problem, Method, and Results Overview:

In **quantitative finance**, traditional strategies often rely on existing **financial models** and historical data to craft trading strategies. 

This paper presents a **"financial-model-free" reinforcement learning (RL) framework**, which adopts a **deep machine learning approach** to solve the **portfolio management problem**. The portfolio management problem involves continuously redistributing capital across various financial assets to **maximize returns** while managing **risk** in a **stochastic** environment, such as financial markets.

The proposed framework uses a **model-free** approach, avoiding the need for traditional **price-prediction models**. Instead, it directly maps market states to **portfolio weights**, providing a more robust and adaptable solution that takes into account **transaction costs** and **market volatility**. 

This **financial-model-free** method overcomes many limitations of classical model-based approaches, particularly in high-friction environments like financial markets.

### Framework Breakdown:

1. **Topology**:
    
    - The **data topology** utilized in this framework is an **Ensemble of Identical Independent Evaluators (EIIE)**. In this setup, multiple **identical models** (evaluators) are trained independently. Their outputs are then aggregated using techniques like **voting**, **averaging**, or **weighting**, which ensures **model diversity** and mitigates the risk of **overfitting**.
        
    - Each evaluator functions as an **independent point** within a larger **topological structure**. The interactions between these evaluators create a **network-like system**, where their outputs, weights, and **correlations** form the basis of connections. This topology helps overcome the issue of **asset identity bias**, which typically arises when a model memorizes the historical quirks of individual assets.
        
    - This topology also provides **linear scalability**: the system’s training time scales linearly with the number of assets, making it computationally efficient even for large portfolios. This **data-usage efficiency** ensures that experience is accumulated across both **time** and **asset** dimensions simultaneously.
        
2. **Portfolio-Vector Memory (PVM)**:
    
    - The framework integrates a **Portfolio-Vector Memory (PVM)**, which stores the previous portfolio vectors. The PVM enables the agent to account for **transaction costs**, ensuring that the portfolio manager minimizes unnecessary rebalancing actions that could erode capital.
        
    - The **PVM** works by allowing the agent to "read" the previous portfolio state $w_{t-1}$ and "write" a new state $w_t$ after executing an action, based on the market environment. This structure mitigates the **churning effect**, where frequent changes in the portfolio can lead to excessive transaction fees.
        
3. **Online Stochastic Batch Learning (OSBL) Schema**:
    
    - **Online Learning**: The model updates its parameters **incrementally** as new data arrives, making it capable of learning from continuous data streams. This is essential for applications like **stock market prediction**, where market data is always changing.
        
    - **Stochastic Learning**: The framework uses **random mini-batches** of data during training, introducing **variability** and randomness into the learning process. This is achieved through techniques like **Stochastic Gradient Descent (SGD)**, where updates are performed based on mini-batches rather than the entire dataset. This helps prevent overfitting to temporary market anomalies.
        
    - **Batch Learning**: The data in each mini-batch is selected **randomly**, ensuring variability and enabling faster updates. The stochastic nature of batch learning ensures that the model doesn't get stuck in local minima and can adapt more effectively to market changes.
        
4. **Fully Exploiting and Explicit Reward Function**:
    
    - **Fully Exploiting**: The **reward function** is designed to allow the agent to **maximize** its performance by fully exploiting the available information. The agent can identify and capitalize on **all opportunities** to maximize its reward.
        
    - **Explicit**: The **reward function** is clearly defined, directly tied to the agent's actions and outcomes, and unambiguous in its interpretation. The agent knows exactly what actions lead to higher rewards, allowing for more efficient learning and decision-making.
        
    - **Reward Function**: The reward for each step $t$ can be calculated as a **logarithmic return** from the portfolio value $p_t$. This ensures that the framework is not just seeking immediate profits but **optimizing for cumulative returns over time, accounting for the agent’s continuous decision-making in the portfolio management task.**
        

### Three Implementations within the Framework:

1. **CNN Implementation**:
    
    - A **Convolutional Neural Network (CNN)** is employed to process **grid-like data** (e.g., price histories). The CNN automatically learns **spatial hierarchies of features** using convolutional layers.
        
    - **Architecture**: The CNN utilizes a **3-layer structure** with **kernels** of height 1. This design ensures that the assets are evaluated independently based on their own price histories, preventing biases related to asset identity.
        
2. **Basic RNN Implementation**:
    
    - A **Recurrent Neural Network (RNN)** captures **sequential data** and learns **time-dependent patterns** in the price histories of assets. This type of network is particularly useful for recognizing **trends** in time-series data.
        
    - **Key Advantage**: The **Basic RNN** is better suited for capturing **cyclical patterns** in market data, which is essential for trading strategies that depend on recurring market behaviors.
        
3. **LSTM Implementation**:
    
    - The **Long Short-Term Memory (LSTM)** network includes **gating mechanisms** designed to manage long-term dependencies in data and solve issues like the **vanishing gradient problem**.
        
    - **Limitation**: Despite its ability to capture long-term dependencies, the **LSTM underperformed** in this case due to the "history repeats itself" nature of financial markets. The LSTM’s design to forget irrelevant information hindered its ability to exploit **cyclical patterns** that were essential in market prediction.
        
### Backtest Experiments:

- **Trading Period**: The backtest was conducted with **30-minute trading periods** in the **cryptocurrency market**, using data from the **Poloniex exchange**.
    
- **Results**: The framework dominated backtest experiments, monopolizing the top three positions in all cases. With a **commission rate of 0.25%**, the framework achieved **at least 4-fold returns** over **50 days**, demonstrating the effectiveness of the proposed approach.
    
- **Comparison**: Traditional models like **PAMR** and **OLMAR** yielded **negative returns**, showcasing the framework's **superior performance** under real-world commission constraints.
    

### Detailed Observations:

- **EIIE Topology**: The **EIIE topology** outperformed the traditional **iCNN** by avoiding the **asset identity bias** that typically arises when a single integrated model tries to memorize long-term behaviors of specific assets. The **EIIE’s** focus on independent evaluations based purely on local price patterns allowed for better adaptability in dynamic market conditions.
    
- **LSTM Limitations**: The **LSTM’s tendency to forget** long-term data actually hindered its performance in this scenario. Financial markets often exhibit **cyclical behaviors** that the **LSTM’s gating mechanism** struggled to capture, making **Basic RNNs** a more effective choice for this task.
    

### Key Framework Insights:

1. **EIIE Topology**:
    
    - **Scalability**: The **EIIE framework** ensures that the training time scales **linearly** with the number of assets, due to **parameter sharing** across evaluators. This makes it highly scalable for large portfolios without exponential computational overhead.
        
    - **Data-Usage Efficiency**: A single **price segment** yields **m** training samples for the individual evaluators, which helps in accumulating experience across both the **time** and **asset** dimensions simultaneously.
        
    - **Asset Plasticity**: The **EIIE framework** is **asset-agnostic**, allowing portfolio composition to be modified in real-time without the need for retraining the network.
        
2. **PVM**:
    
    - The **Portfolio-Vector Memory (PVM)** enables the system to efficiently store previous portfolio states, thus mitigating the **gradient vanishing problem** and allowing for **parallel training** of mini-batches. This speeds up convergence, making the training process more efficient.


### Conclusion:

The **financial-model-free RL framework** introduced in this paper offers an **innovative and scalable** solution for **portfolio management**. By mapping market states directly to portfolio weights and incorporating **transaction costs** into the learning process, this approach overcomes many limitations of traditional model-based strategies. With the **EIIE topology**, **PVM**, and **OSBL schema**, the framework successfully adapts to the **non-stationary nature** of financial markets, providing an efficient, **autonomous portfolio management** system.

The framework’s ability to handle **high-frequency trading**, **transaction costs**, and **market volatility** positions it as a significant advancement in the field of quantitative finance. Despite some areas for future refinement, such as accounting for **slippage** and **market liquidity constraints**, the **EIIE meta-topology** proves to be a dominant solution for **automated portfolio management** in dynamic and **high-friction** markets.

---
# MATHEMATICS 

In the architecture of automated trading systems, the representation of market movement determines the structural integrity of the learning algorithm. For reinforcement learning agents operating in continuous financial environments, modeling price movements as relative ratios, rather than absolute price changes, is a prerequisite for scale-invariance. This approach aligns the agent's internal state with the stochastic evolution of capital growth, which is tethered to the price-relative tensor. By framing state transitions through price relatives, we establish a framework where wealth dynamics are governed by proportions, allowing the agent to generalize underlying growth patterns regardless of the nominal price or currency units of the underlying assets.

### 1. The Dynamics of Price Relatives and State Transitions

The fundamental operator in this mathematical framework is the Price Relative Vector, $\mathbf{y}_t$, which captures the shifting landscape of asset values between two consecutive trading periods. In a continuous market, we define $\mathbf{v}_t$ as the vector of closing prices at period $t$, which simultaneously serves as the opening price for period $t+1$.

**The Price Relative Vector ($\mathbf{y}_t$):**
$$
\mathbf{y}_t := \mathbf{v}_t \oslash \mathbf{v}_{t-1}
= \left(1, \frac{v_{1,t}}{v_{1,t-1}}, \frac{v_{2,t}}{v_{2,t-1}}, \dots, \frac{v_{m,t}}{v_{m,t-1}}\right)^\top
$$

In this construction, the first element $y_{0,t}$ is pegged to 1, representing the quoted currency (e.g., Bitcoin or Cash), which serves as the risk-free unit for the portfolio. The $i$-th element $y_{i,t}$ represents the quotient of consecutive closing prices for a specific asset.

From an architectural perspective, $\mathbf{y}_t$ is the singular link between the agent's action (the allocation vector $\mathbf{w}$) and the environment's reward. In the context of a Partially Observable Markov Decision Process (POMDP), $\mathbf{y}_t$ acts as the multiplicative transition operator that scales the portfolio's capital across time. The evolution of the total portfolio value, $p_t$, is thus defined by the interaction between these price relatives and the agent's prior allocation:

**Portfolio Value Transition ($p_t$):**
$$
p_t = p_{t-1}\,\mathbf{y}_t \cdot \mathbf{w}_{t-1}
$$

Here, $p_{t-1}$ is the value at the beginning of period $t$, and $\mathbf{w}_{t-1}$ is the allocation vector set at the start of that period. This dot product captures the multiplicative growth of capital across the asset universe, condensing the weighted growth of individual holdings into a scalar representation of total capital. While this provides the mechanics for value transition, optimizing for long-term growth necessitates a shift toward a logarithmic characterization of returns.

### 2. Logarithmic Formulations and Reward Transformations

While wealth accumulation is a multiplicative process, gradient-based optimization algorithms achieve higher stability when operating on additive signals. The strategic utility of log-returns lies in their capacity to convert path-dependent multiplicative growth into a path-independent additive process, a transition that is essential for effective Backpropagation Through Time (BPTT).

**Logarithmic Rate of Return ($r_t$):**
$$
r_t := \ln\frac{p_t}{p_{t-1}} = \ln\!\left(\mathbf{y}_t \cdot \mathbf{w}_{t-1}\right)
$$

The periodic log-return $r_t$ is the natural logarithm of the dot product between the allocation weights and the price relatives. This transformation serves as the standard objective for long-term growth optimization (log-utility). By utilizing the natural logarithm, we stabilize the reward signal and prevent the exponential "blow-up" of values common in compounding sequences. Maximizing the sum of log-returns is mathematically equivalent to maximizing the geometric mean of returns, which is the rigorous path to maximizing long-term wealth while ensuring the signal remains suitable for differentiable optimization. This additive form allows us to aggregate individual period performance into a coherent objective for cumulative compounding.

### 3. The Compounding Objective and Cumulative Wealth

The terminal objective of any portfolio management policy is the maximization of accumulated wealth over a finite horizon $t_f$. This value is the product of a sequence of growth factors, where the output of each period provides the basis for the next.

**Final Portfolio Value ($p_f$):**
$$
p_f = p_0 \prod_{t=1}^{t_f+1} \mathbf{y}_t \cdot \mathbf{w}_{t-1}
$$

In this idealization, $p_0$ represents the initial capital and the product operator represents the compounding of periodic growth. This formula defines the objective function for an agent seeking to navigate the market's physics to reach a theoretical wealth ceiling. However, while this equation defines the ideal growth rate, the introduction of transactional friction necessitates a non-linear correction to the reward signal to account for the erosion of wealth during reallocation.

### 4. Transactional Friction and the Transaction Remainder Factor

To prevent an agent from over-trading and chasing marginal gains that are consumed by fees, the model must incorporate frictional loss. At the end of period $t$, price movements cause the initial weights $\mathbf{w}_{t-1}$ to evolve into a new distribution $\mathbf{w}'_t$:

$$
\mathbf{w}'_t = \frac{\mathbf{y}_t \odot \mathbf{w}_{t-1}}{\mathbf{y}_t \cdot \mathbf{w}_{t-1}}
$$

The agent must then rebalance from these evolved weights $\mathbf{w}'_t$ to the new target weights $\mathbf{w}_t$. This redistribution incurs commission fees ($c_s$ for selling, $c_p$ for purchasing), which reduces the portfolio value by a factor $\mu_t$, termed the Transaction Remainder Factor.

**Transaction Remainder Factor ($\mu_t$):**
$$
\mu_t = \frac{1}{1-c_p w_{t,0}}
\left[
1 - c_p w'_{t,0}
- (c_s + c_p - c_s c_p)\sum_{i=1}^{m}\left(w'_{t,i} - \mu_t w_{t,i}\right)_+
\right]
$$

The variable $\mu_t$ is self-referential; the amount of an asset sold depends on the final portfolio value, which is itself diminished by the commission of the sale. Because of this recursive dependency, $\mu_t$ is not analytically solvable. As established in Theorem 1, it must be solved via an iterative convergence method where $\mu_t^{(k)} = f\!\left(\mu_t^{(k-1)}\right)$. The factor $\mu_t \in (0,1]$ transforms the ideal return into a net return, penalizing allocation volatility through the ReLU function $(x)_+$.

The logarithmic reward function is subsequently updated to reflect this friction:

**Log Return with Friction ($r_t$ with cost):**
$$
r_t = \ln\!\left(\mu_t \mathbf{y}_t \cdot \mathbf{w}_{t-1}\right)
$$

By embedding the remainder factor within the log-function, the agent is forced to learn a policy that balances the pursuit of market growth against the non-linear costs of volatility.

### 5. The Strategic Objective Function: Maximizing Log-Growth Rates

The global objective function $R$ is the metric the agent aims to maximize via deterministic policy gradients. This function aggregates the net returns into a single utility score for the entire trading session.

**Average Logarithmic Accumulated Return ($R$):**
$$
R = \frac{1}{t_f}\sum_{t=1}^{t_f+1} r_t
$$

While final wealth $p_f$ is multiplicative, $R$ is defined as the additive average of log-returns. This specific mathematical form is chosen to facilitate full-exploitation training. Under Hypothesis 2 (Zero Market Impact), the agent assumes its actions $\mathbf{w}_t$ do not influence future price relatives $\mathbf{y}_{t+1}$. In this regime, the environment is stationary with respect to the agent's actions, allowing the use of a deterministic policy gradient rather than the stochastic gradients typical of robotics or games where actions alter the state.

Collectively, these formulations model the fundamental physics of compounding. By transforming chaotic market price movements into a structured, cost-aware mathematical landscape, the agent can employ gradient ascent to navigate toward long-term capital growth within a competitive market.

---
### **1. Problem:**
This paper addresses the **portfolio management problem**, where the goal is to **maximize returns** while managing **risk** in **stochastic** environments (financial markets). It proposes a **financial-model-free reinforcement learning (RL)** approach to solving the problem, bypassing traditional price-prediction models and instead focusing on **directly mapping market states to portfolio weights**.

### **2. Setup:**
* **Multiplicative:** The framework involves **multiplicative dynamics**, where the portfolio value grows multiplicatively based on market states (price relatives and portfolio weights). This is captured in the equation $$ p_{t+1} = p_t \cdot \mathbf{y}_t \cdot \mathbf{w}_t $$,which models the portfolio value's growth over time.
* **Continuous:** The portfolio weights $$ \mathbf{w}_t $$are continuously adjusted based on market dynamics, meaning it's a **continuous action space**. This aligns with your thesis idea of having the agent decide both the action (what to do) and the magnitude (how much to commit).

### **3. Key Idea:**
The key idea is using **deep reinforcement learning** with a **model-free approach** where the agent directly maps market states to portfolio weights. The **reward function** is based on **logarithmic returns** to capture **compounding growth** over time, making the system robust to long-term volatility and risk, while ensuring scalability through an **Ensemble of Identical Independent Evaluators (EIIE)** topology.

### **4. Assumptions:**
* **Stationary:** The framework assumes a **stationary** market environment where the market dynamics remain consistent over time. However, the agent’s policy adapts to changing market conditions in real time.
* **Known Probabilities:** The model uses **price-relative vectors** to model the market, but it doesn’t require known probabilities. It does, however, **assume market prices are observable and transition smoothly** between states, which is an **implicit assumption of stationarity**.

### **5. Limitation:**
The method does not address the **adaptive decision-sizing** under **uncertainty** or **non-stationary environments** where probability estimates may be uncertain or the environment could change. Specifically, the system assumes stationarity and does not handle the **compound effects of errors** in decision-making when faced with **uncertain or changing market conditions**. Additionally, the system assumes no **market impact** from the agent’s actions, which is often unrealistic in real-world markets.

### **6. Relation to my idea:**
This paper’s approach to portfolio management is related to my idea of **reinforcement learning under multiplicative dynamics**, but it lacks a mechanism for learning **adaptive decision magnitude** under **uncertainty or non-stationarity**, which is the focus of my thesis.

---
### Integration:

* **Problem:** The RL framework for portfolio management (like this paper) is a **good fit for understanding multiplicative dynamics**, but it doesn’t focus on **uncertainty and decision sizing**—which is the gap I’m looking to address in my thesis.

* **Limitation:** The primary limitation of this paper is that it assumes **stationary environments** and doesn’t integrate an **adaptive decision-sizing mechanism** for handling uncertainty or dynamic market conditions.


