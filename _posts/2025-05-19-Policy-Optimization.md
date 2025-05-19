---
title: 'Summary of Popolar RFT Methods'
layout: single
---

## Policy Gradient

For a policy $\pi_{\theta}$, we want to maximize the expected return $J(\pi_{\theta}) = E_{\tau~\pi_{\theta}}[R(\tau)]$ where $\tau$ is a trajectory and $R(\tau)$ is the return of the trajectory.

We optimiza the policy by gradient ascent, i.e., $\theta_{k+1} \leftarrow \theta_{k} + \alpha \nabla_{\theta} J(\pi_{\theta})$ where $\alpha$ is the learning rate.

After some [derivation](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#part-3-intro-to-policy-optimization), the simplest policy gradient is given by:
$$
\hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) R(\tau),
$$
where $\mathcal{D}$ is a set of collected trajectories.

The definition of return $R(\tau)$ has several forms, including:
- Reward after the action $a_t$: $\sum_{t^\prime=t}^T r(s_t, a_t)$
- State-action value function $Q^\pi(s_t, a_t)$
- TD residual $r_t + V^\pi(s_{t+1}) - V^\pi(s_t)$
- Advantage function $A^\pi(s_t, a_t)$

In the following, we adopt the advantage function as the return. It describes how much better the agent is doing compared to a 'baseline'. The policy gradient is then estimated as:
$$
g = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \hat{A}_t \right].
$$
Here $\hat{\mathbb{E}}_t$ represents the expectation over the sampled experiences.


## PPO and Its Origin

### Trust Region Policy Optimization (TRPO)

Trust Region Policy Optimization (TRPO) is a policy gradient method that uses a trust region to ensure that the policy update does not deviate too much from the current policy.
Essentially, This is done by introducing importance sampling into the policy gradient update.

We note that importance sampling has the following form:
$$
E_{x~p(x)}[f(x)] =  \int p(x) f(x) dx = \int \frac{p(x)}{q(x)} q(x) f(x) dx =  E_{x~q(x)}[f(x) \frac{p(x)}{q(x)}].
$$

> Why do we need importance sampling?
> Policy gradient requires the on-policy data; however, in practice, we often need to use off-policy data for training as well. 
> Importance sampling allows us to use off-policy data for updating the current policy.

Hence, we can rewrite the policy gradient update as follows (see Section 6.1 in [Zhihu](https://zhuanlan.zhihu.com/p/342150033)):
$$
\begin{aligned}
& \max_\theta \quad \mathbb{\hat{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t \right] \\
& \text{s.t.} \quad \mathbb{\hat{E}}_t \left[ \text{KL} \left[ \pi_{\theta_{\text{old}}}(\cdot | s_t), \pi_\theta(\cdot | s_t) \right] \right] \leq \delta.
\end{aligned}
$$

If we use the KL divergence as a penalty, we can use the following objective:
$$
\max_{\theta} \quad \mathbb{\hat{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t - \beta \cdot \text{KL} \left[ \pi_{\theta_{\text{old}}}(\cdot | s_t), \pi_\theta(\cdot | s_t) \right] \right].
$$

The surrogate objective is the above is defined as conservative policy iteration in the [paper](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf).
$$
L^{CPI}(\theta) = \mathbb{\hat{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t \right] = \mathbb{\hat{E}}_t \left[ r_t(\theta) \hat{A}_t \right].
$$


### Proximal Policy Optimization (PPO)

As introduced by [OpenAI](https://arxiv.org/pdf/1707.06347), PPO improves upon TRPO by clipping the update. Let $r_t(\theta)=\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$, then
$$
L^{\text{CLIP}}(\theta) = \mathbb{\hat{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

The effect of clipping is:

- To avoid a too large update;
- When the advantage is positive, allow $r_t(\theta)$ to increase, but not larger than $1+\epsilon$; when the advantage is negative, allow $r_t(\theta)$ to decrease, but not smaller than $1-\epsilon$.

Another alternative way to clip the advantage is to use adaptive KL penalty. This is omitted as it performs worse than the clipping method.

### Generalized Advantage Estimation (GAE)
The problem of advantage estimation lies in the bias-variance trade-off:
- If we use a large time horizon (e.g., of the whole trajectory), the bias is small, but the variance is large;
- If we use a small time horizon (e.g., one step), the variance is small, but the bias is large.

[Generalized Advantage Estimation (GAE)](https://arxiv.org/pdf/1506.02438) balances this trade-off by using the n-step returns and exponentially weighted averages. We explain this step-by-step.

The return of one step is sum of instant reward and discounted value of the next state, i.e., $G_t^{(1)} = r_t + \gamma V(s_{t+1})$. The TD-error is defined as the difference between the value estimate and the target value, i.e., $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. One-step TD is a biased estimator of the value function, but commonly used in Sarsa and Q-learning.

The n-step TD return is defined as $G_t^{(n)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$.
When $n$ is large, the return relies on the true reward $r_i$, and the accumulated stochasticity leads to a large variance.

GAE is defined as:
$$
G_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l},
$$
where $\lambda$ adjusts the decay of weights assigned to historical TD residuals.
This can be viewed as a weighted average of TD residuals, with weight $(1-\gamma\lambda)(\gamma\lambda)^{k-1}$.

The code of GAE is as follows:
```python
def compute_gae_advantage_return():
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]

    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]

        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)

    returns = advantages + values
    advantages = verl_F.masked_whiten(advantages, eos_mask)
```


## GRPO and Its Variants

### Group Relative Policy Optimization (GRPO)

PPO uses an critic to estimate the value of the state-action pair, which can be computationally expensive. GRPO, proposed in [Deepseek-Math](https://arxiv.org/pdf/2402.03300), proposes to use the average reward of multiple sampled outputs for advantage estimation. The advantage is computed as follows:
$$
\hat{A}_{i,t} = \frac{r_i - mean(r)}{std(r)}, \forall i \in [1, K],
$$
and the objective is given as:
$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{\left[ q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q) \right]}
$$
$$
\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ r_{t,i}(\theta) \hat{A}_{i,t}, \text{clip} \left( r_{t,i}(\theta), 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t} \right] - \beta \mathbb{D}_{\text{KL}} \left[ \pi_\theta \| \pi_{\text{ref}} \right] \right\}
$$,
where $r_{t,i}(\theta)=\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}$.

Apart from the advantage estimation, another dfference from PPO is the KL divergence as an unbiased estimator:
$$
\mathbb{D}_{\text{KL}} \left[ \pi_\theta \| \pi_{\text{ref}} \right] = \frac{\pi_{\text{ref}}(o_{i,t} | q, o_{i,<t})}{\pi_\theta(o_{i,t} | q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} | q, o_{i,<t})}{\pi_\theta(o_{i,t} | q, o_{i,<t})} - 1.
$$

We notice that here we need three models:
- the policy model $\theta$ which is our optimization target;
- the old model $\theta_{\text{old}}$ which generates the rollout experiences and may dynamically change during training; 
- the reference model $\theta_{\text{ref}}$ for computing the KL divergence, which is usually the starting point of the policy model.

#### Outcome Supervision RL or Process Supervision RL
Lets us have a closer look at $r_i$. There are two types of $r_i$:

- Out supervision RL: A reward model (or some rule) is used to score the outputs and gives a reward for each rollout experience.
- Process supervision RL: A reward model provides a reward at the end of
each reasoning step; the advantage of each token as the sum of the normalized rewards from the following steps, i.e., $\hat{A}_{i,t} = \sum_{\text{index}(j) \geq t} \widetilde{r}_i^{\text{index}(j)}$.

Despite its simplicity, GRPO is not very stable. Hence, some works propose more stable variants.

### RLOO

## REINFORCE and Its Improved Variants

### REINFORCE
[REINFORCE](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf) is a classical policy gradient algorithm. 
$$
\begin{aligned}
& \nabla_\theta \mathbb{E}_{a_{1:T} \sim \pi_\theta}[r(x, a_{1:T})] \\
& = \sum_{a_{1:T}} \nabla_\theta \left[ \prod_{t=1}^T \pi_\theta(a_t | x, a_{1:t-1}) \right] r(x, a_{1:T}) \\
& = \mathbb{E}_{a_{1:T} \sim \pi_\theta} \left[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t | x, a_{1:t-1}) r(x, a_{1:T}) \right].
\end{aligned}
$$

We define the score function as $s_{\theta}(x, a_{1:t}) = \nabla_\theta \log \pi_\theta(a_t | x, a_{1:t-1})$. Then for $N$ prompts $x_i, i\in [N]$, the gradient estimator is given as:
$$
\hat{g}(\theta) = \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T s_\theta(x^i, a^i_{1:t}) r(x^i, a^i_{1:T}).
$$
This corresponds to reward-weighted likelihood maximization
with samples generated from the rollouts.
However, REINFORCE suffers from a large variance, as pointed out in [Remax](https://arxiv.org/pdf/2310.10505), due to:
- the external randomness inherent in MDP’s transitions, which is actually not a problem for RLHF since the transitions are deterministic and the reward function is given (the result of transition is deterministic); 
- the internal randomness from the policy decisions of the language model (i.e., token generation).


### ReMax
Inspired by REINFORCE with baseline, ReMax modifies the policy gradient as:
$$
\tilde{g}(\theta) = \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \left[ s_\theta(x^i, a^i_{1:t}) \times (r(x^i, a^i_{1:T}) - b_\theta(x^i)) \right],
$$
and the baseline is given by:
$$
b_\theta(x^i) = r(x^i, \bar{a}^i_{1:T}), \quad \bar{a}_t^i \in \arg\max \pi_\theta(\cdot | x^i, \bar{a}_{1:t-1}^i).
$$
ReMax reduces the variance effectively by using the baseline.


### Reinforce++

[Reinforce++](https://arxiv.org/pdf/2501.03262) introduces several tricks to enhance training stability and efficiency:
- Token-Level KL Penalty: $r(s_t, a_t) = \mathbb{I}(s_t = [\text{EOS}]) r(x, y) - \beta \, \text{KL}(t)$, where $\text{KL}(t) = \log \left( \frac{\pi_{\theta_{\text{old}}}^{\text{RL}}(a_t | s_t)}{\pi_{\text{REF}}(a_t | s_t)} \right)$. (Comment: Why is this helpful?)

- PPO-Clip Integration: $L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$.

- Mini-Batch Updates, instead of full-batch updates.

- Reward Normalization and Clipping.

- Advantage Normalization: $A_t(s_t, a_t) = r(x, y) - \beta \cdot \sum_{i=t}^T \text{KL}(i)$ and $A_{\text{normalized}} = \frac{A - \mu_A}{\sigma_A}$.


## A Variant of OPMD
[Kimi k1.5](https://arxiv.org/pdf/2501.12599) applies a variant of online policy mirror decent (OPMD) as the training algorithm.
Recall the objective function is:
$$
\max_\theta \mathbb{E}_{(x, y^*) \sim \mathcal{D}} \left[ \mathbb{E}_{(y, z) \sim \pi_\theta} [r(x, y, y^*)] - \tau \text{KL}(\pi_\theta(x) \| \pi_{\theta_i}(x)) \right],
$$
where $\pi_{\theta_i}(x)$ is the current policy model as a reference. 

To derive the gradient of the above objective function, we use the Policy Gradient Theorem for the first term:
$$
\nabla_\theta \mathbb{E}_{(x, y^*) \sim \mathcal{D}} \left[ \mathbb{E}_{(y, z) \sim \pi_\theta} [r(x, y, y^*)] \right]
= \mathbb{E}_{(x, y^*) \sim \mathcal{D}} \left[ \mathbb{E}_{(y, z) \sim \pi_\theta} \left[ r(x, y, y^*) \cdot \nabla_\theta \log \pi_\theta(y, z|x) \right] \right].
$$
The second term KL divergence is equal to $\mathbb{E}_{(y, z) \sim \pi_\theta(x)} \left[ \log \frac{\pi_\theta(y, z | x)}{\pi_{\theta_i}(y, z | x)} \right]$, which has the gradient:
$$
\nabla_\theta \mathbb{E}_{(x, y^*) \sim \mathcal{D}} \left[ \text{KL}(\pi_\theta(x) \| \pi_{\theta_i}(x)) \right]
= \mathbb{E}_{(x, y^*) \sim \mathcal{D}} \left[ \mathbb{E}_{(y, z) \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(y, z|x) \right] \right].
$$

Using Lagrange Multiplier, we get that this objective has a closed form solution:
$$
\pi^*(y, z | x) = \pi_{\theta_i}(y, z | x) \exp \left( \frac{r(x, y, y^*)}{\tau} \right) / Z,
$$
where $Z$ is the normalizing factor $Z = \sum_{y^\prime, z^\prime} \pi_{\theta_i}(y^\prime, z^\prime | x) \exp \left( \frac{r(x^\prime, y^\prime, y^*)}{\tau} \right)$.

Taking logarithm of both sides, for any $(y, z)$ the following constraint is satisfied:
$$
r(x, y, y^*) - \tau \log Z = \tau \log \frac{\pi^*(y, z | x)}{\pi_{\theta_i}(y, z | x)}.
$$
This allows us to use the off-policy data during optimization. The surrogate loss is defined as follows:
$$
L(\theta) = \mathbb{E}_{(x, y^*) \sim \mathcal{D}} \left[ \mathbb{E}_{(y, z) \sim \pi_{\theta_i}} \left[ \left( r(x, y, y^*) - \tau \log Z - \tau \log \frac{\pi_\theta(y, z | x)}{\pi_{\theta_i}(y, z | x)} \right)^2 \right] \right].
$$

The authors find that using empirical mean of sampled rewards $\bar{r} = \text{mean}(r(x, y_1, y^*), \dots, r(x, y_k, y^*))$ to approximate $\tau \log Z$ yields effective practical results. This is reasonable since $\tau \log Z$ approaches the expected reward under $\pi_{\theta_i}$ as $\tau \to \infty$. 
Finally, we conclude our learning algorithm by taking the gradient of surrogate loss. For each problem $x$, $k$ responses are sampled using the reference policy $\pi_{\theta_i}$, and the gradient is given by
$$
\frac{1}{k} \sum_{j=1}^k \left( \nabla_\theta \log \pi_\theta(y_j, z_j | x) (r(x, y_j, y^*) - \bar{r}) - \frac{\tau}{2} \nabla_\theta \left( \log \frac{\pi_\theta(y_j, z_j | x)}{\pi_{\theta_i}(y_j, z_j | x)} \right)^2 \right).
$$.

In this method, this gradient resembles the policy gradient of the original loss using the mean of sampled rewards as the baseline.
The main differences are that the responses are sampled from $\pi_{\theta_i}$ rather than on-policy, and an $l_2$-regularization is applied.


## Towards to A Unified Paradigm
In Deepseek-Math, authors summarize a unified paradigm for SFT, RFT, DPO, PPO and GRPO. The key points are the algorithm $\mathcal{A}$, the data source $\mathcal{D}$, and the reward model $\boldsymbol{\pi}_{rf}$.
$$
\nabla_\theta \mathcal{J}_{\mathcal{A}}(\theta) = \mathbb{E}\underbrace{ \left[ (q, o) \sim \mathcal{D} \right]}_{\text{Data Source}} \left( \frac{1}{|o|} \sum_{t=1}^{|o|} \underbrace{\text{GC}_{\mathcal{A}}(q, o, t, \boldsymbol{\pi}_{rf})}_{\text{Gradient Coefficient}} \nabla_\theta \log \pi_\theta(o_t | q, o_{<t}) \right).
$$
The details are omitted here. I note here it is promising to adopt different algorithms and reward functions to accomodate different data samples.