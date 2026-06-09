---
title: "An Algorithmic Guide to Ornstein-Uhlenbeck Processes"
date: 2026-06-10 00:00:00 +0700
categories: [Quantitative Finance, Stochastic Calculus]
tags: [ou-process, simulation, parameter-estimation]
math: true
mermaid: true
---
# Ornstein-Uhlenbeck Processes

_An algorithmic guide from standard Brownian motion to simulation, estimation, and extensions_

> **Reading promise.** The only stochastic-process prerequisite is standard Brownian motion. Each concept is introduced first by intuition, then by notation, then by the algorithm it enables.

## 1. The Problem OU Solves

A standard Brownian motion $B = (B_t)_{t \ge 0}$ is excellent for modelling random local shocks: it has continuous paths, stationary independent increments, and

$$
B_t - B_s \sim \mathcal{N}(0, t-s)
$$

But it has no memory of a preferred level. Once it wanders away, the model does not create any force pulling it back.

The Ornstein-Uhlenbeck process adds exactly that missing force. It keeps the Brownian shock term, but adds a deterministic drift that points toward a long-run level \(m\). In statistical arbitrage language, if a spread is far above its equilibrium, the drift is negative; if it is far below, the drift is positive.

$$
dX_t = \gamma(m - X_t)\,dt + \sigma\,dB_t,\qquad t>0
$$

Here $\gamma$ is the speed of mean reversion, $m$ is the equilibrium level, and $\sigma$ is the diffusion scale. The sign of $m-X_t$ is the whole intuition.

### Notation Used Throughout

| Symbol | Meaning |
|---|---|
| $B_t$ | Standard Brownian motion at time $t$. |
| $X_t$ | The OU state at time $t$, such as a spread, rate, or physical displacement. |
| $m$ | Long-run equilibrium level. |
| $\gamma$ | Mean-reversion speed. The mean-reverting case is $\gamma>0$. |
| $\sigma$ | Diffusion scale multiplying Brownian shocks. |
| $\Delta$ | Sampling interval $t_i - t_{i-1}$. |
| $a_\Delta = e^{-\gamma\Delta}$ | Discrete-time persistence over one interval. |
| $q_\Delta = \sigma^2(1-e^{-2\gamma\Delta})/(2\gamma)$ | One-step conditional innovation variance. |

### A Brownian Motion Comparison

| Feature | Standard Brownian motion | Gaussian OU process |
|---|---|---|
| Local randomness | Driven by $dB_t$. | Also driven by $dB_t$. |
| Preferred level | None. | Mean reverts toward $m$ when $\gamma>0$. |
| Variance over time | $\operatorname{Var}(B_t)=t$ grows without bound. | Long-run variance is $\sigma^2/(2\gamma)$. |
| Stationarity | Not stationary. | Stationary when $\gamma>0$ and $X_0$ has the invariant law. |
| Discrete grid form | $X_i = X_{i-1} + \text{noise}$. | $X_i = (1-a)m + aX_{i-1} + \text{noise}$. |

### Three Fundamental OU Tasks

We can separate the OU process into three practical tasks:

- **Problem 1 (Transition):** given $X_s=x$, compute the distribution of $X_t$.
- **Problem 2 (Simulation):** generate a path $X_0, X_1, \dots, X_n$ on a grid.
- **Problem 3 (Learning):** estimate $\gamma$, $m$, and $\sigma$ from discrete observations.

## 2. Solving the SDE

Start by centering the process around its equilibrium:

$$
Y_t = X_t - m
$$

Since $dY_t=dX_t$, the SDE becomes

$$
dY_t = -\gamma Y_t\,dt + \sigma\,dB_t
$$

Multiply by the integrating factor $e^{\gamma t}$. Itô's product rule gives

$$
d\left(e^{\gamma t}Y_t\right) = \sigma e^{\gamma t}\,dB_t
$$

Integrating from $0$ to $t$ and multiplying back by $e^{-\gamma t}$ yields the explicit solution

$$
X_t
= m + (X_0-m)e^{-\gamma t}
+ \sigma\int_0^t e^{-\gamma(t-s)}\,dB_s
$$

This is the most important formula in the ordinary OU model. The current value is the sum of a decayed initial displacement and a weighted history of Brownian shocks. Recent shocks get weight near one; older shocks are exponentially forgotten.

> **Why the stochastic integral is Gaussian.** For deterministic square-integrable $f$, the Itô integral $\int f(s)\,dB_s$ is normal with mean $0$ and variance $\int f(s)^2\,ds$. Here $f(s)=\sigma e^{-\gamma(t-s)}$.

## 3. Transition Distribution

The explicit solution immediately gives the transition law. For a time gap $\Delta=t-s>0$ and a known value $X_s=x$, define

$$
a_\Delta = e^{-\gamma\Delta},
\qquad
q_\Delta = \frac{\sigma^2(1-e^{-2\gamma\Delta})}{2\gamma}
$$

Then

$$
X_t \mid X_s=x
\sim
\mathcal{N}\left(m+a_\Delta(x-m),\ q_\Delta\right)
$$

This formula is the OU analogue of a one-step HMM transition-emission calculation. In an HMM, a transition matrix tells us how mass moves between hidden states. In an OU process, the Gaussian transition kernel tells us how probability mass moves on the real line.

The transition density is

$$
p_\Delta(x,y)
=
\frac{1}{\sqrt{2\pi q_\Delta}}
\exp\left(
-\frac{(y-m-a_\Delta(x-m))^2}{2q_\Delta}
\right)
$$

### Algorithm 1: Exact OU Transition

```text
function OU-TRANSITION(x, Delta, gamma, m, sigma) returns mean, variance

    a <- exp(-gamma * Delta)
    q <- sigma^2 * (1 - a^2) / (2 * gamma)
    mean <- m + a * (x - m)
    return mean, q
 
```

The algorithm has no approximation error for the Gaussian OU model. The only stochastic part is the later draw from the normal distribution.

## 4. Mean Reversion and Stationarity

Taking conditional expectation in the solution gives

$$
\mathbb{E}[X_t\mid X_0]
=
m + (X_0-m)e^{-\gamma t}
$$

When $\gamma>0$, the factor $e^{-\gamma t}$ decays to zero. The process forgets its starting point in expectation. The conditional variance is

$$
\operatorname{Var}(X_t\mid X_0)
=
\frac{\sigma^2(1-e^{-2\gamma t})}{2\gamma}
$$

Thus the limiting distribution is

$$
X_\infty \sim \mathcal{N}\left(m,\frac{\sigma^2}{2\gamma}\right)
$$

If $X_0$ is drawn from this distribution, the entire process is strictly stationary. In that stationary regime,

$$
\operatorname{Cov}(X_s,X_t)
=
\frac{\sigma^2}{2\gamma}e^{-\gamma|t-s|}
$$.

> **Parameter cases.** If $\gamma=0$, the model reduces to Brownian motion with scale $\sigma$ and $m$ is not identifiable. If $\gamma<0$, the drift pushes the process away from $m$, so the model is explosive rather than mean-reverting.

## 5. Discrete Sampling: The AR(1) Form

In practice, we observe a finite sequence on a grid, not the full continuous path. Let $t_i=i\Delta$ and $X_i=X_{t_i}$. The transition law becomes the exact autoregression

$$
X_i = (1-a)m + aX_{i-1} + \eta_i,
\qquad
a=e^{-\gamma\Delta},
$$

where

$$
\eta_i \sim \mathcal{N}(0,q),
\qquad
q=\frac{\sigma^2(1-a^2)}{2\gamma},
\qquad
\eta_i \text{ independent}
$$

This is the key bridge from continuous-time stochastic calculus to data analysis. The OU process sampled at equal intervals is an AR(1) process with a parameter mapping. This is exactly the point emphasized in the OU survey's discretely sampled process section.

> **Exact versus Euler.** Euler-Maruyama would use $X_i \approx X_{i-1} + \gamma(m-X_{i-1})\Delta + \sigma\sqrt{\Delta}\varepsilon_i$. The exact AR(1) transition should be preferred whenever the model is the Gaussian OU process, because it remains correct even when $\Delta$ is not tiny.

### Algorithm 2: Exact Simulation on a Grid

```text
function SIMULATE-OU(x0, n, Delta, gamma, m, sigma) returns x[0:n]
    x[0] <- x0
    a <- exp(-gamma * Delta)
    q <- sigma^2 * (1 - a^2) / (2 * gamma)

    for i from 1 to n do
        epsilon <- STANDARD-NORMAL()
        x[i] <- m + a * (x[i-1] - m) + sqrt(q) * epsilon

    return x
```

The initialization is the starting value, the recursion is the Gaussian transition, and the termination is the simulated path. This mirrors the initialization-recursion-termination structure used for HMM dynamic programs.

## 6. Learning Parameters from Data

Suppose we observe $x_0,x_1,\dots,x_n$ at equally spaced times with gap $\Delta$. Since the sampled OU is an AR(1), first estimate

$$
x_i = c + ax_{i-1} + u_i,
\qquad
c=(1-a)m
$$

For Gaussian innovations, ordinary least squares gives the same point estimates for $a$ and $c$ as conditional maximum likelihood. Then map the discrete parameters back to continuous time:

$$
\widehat{\gamma}
=
-\frac{\log(\widehat{a})}{\Delta},
\qquad
\widehat{m}
=
\frac{\widehat{c}}{1-\widehat{a}},
\qquad
\widehat{\sigma}^2
=
\widehat{q}\frac{2\widehat{\gamma}}{1-\widehat{a}^2}
$$

The estimate $\widehat{q}$ is the innovation variance estimated from residuals. The mapping is meaningful for mean reversion when $0<\widehat{a}<1$. Values near one imply slow mean reversion; values outside this range signal either sampling noise, model mismatch, or a non-mean-reverting series.

### Algorithm 3: AR(1)-Based OU Estimation

```text
function ESTIMATE-OU(x[0:n], Delta) returns gamma_hat, m_hat, sigma_hat
    y <- x[1:n]
    z <- x[0:n-1]

    a_hat <- sum((z - mean(z)) * (y - mean(y))) / sum((z - mean(z))^2)
    c_hat <- mean(y) - a_hat * mean(z)

    residuals <- y - c_hat - a_hat * z
    q_hat <- mean(residuals^2)

    gamma_hat <- -log(a_hat) / Delta
    m_hat <- c_hat / (1 - a_hat)
    sigma2_hat <- q_hat * 2 * gamma_hat / (1 - a_hat^2)

    return gamma_hat, m_hat, sqrt(sigma2_hat)
```

> **Numerical guardrail.** Before applying the logarithm, check that $0<\widehat{a}<1$. If $\widehat{a}$ is very close to 1, the estimated half-life $\log(2)/\widehat{\gamma}$ becomes very large and uncertainty is usually high.

### Likelihood View

The transition density also gives a direct conditional log-likelihood:

$$
\ell(m,\gamma,\sigma)
=
-\frac{1}{2}
\sum_{i=1}^n
\left[
\log(2\pi q_\Delta)
+
\frac{
\left(x_i-m-a_\Delta(x_{i-1}-m)\right)^2
}{q_\Delta}
\right]
$$

For irregular sampling, replace $\Delta$ by $\Delta_i=t_i-t_{i-1}$ inside $a_i$ and $q_i$, then maximize the same sum numerically.

### Algorithm 4: Irregular-Grid Log-Likelihood

```text
function OU-LOG-LIKELIHOOD(times[0:n], x[0:n], gamma, m, sigma) returns loglik
    loglik <- 0
    for i from 1 to n do
        Delta <- times[i] - times[i-1]
        a <- exp(-gamma * Delta)
        q <- sigma^2 * (1 - a^2) / (2 * gamma)
        mu <- m + a * (x[i-1] - m)
        loglik <- loglik - 0.5 * (log(2*pi*q) + (x[i] - mu)^2 / q)

    return loglik
```

## 7. From OU to Generalised OU

The ordinary OU process can be defined in three equivalent ways: as an SDE, as an explicit stochastic integral, and as a time-changed Brownian motion. The survey explains that once Brownian motion is replaced by more general Lévy drivers, these routes no longer automatically define the same object.

The Generalised OU process keeps the explicit-integral idea. Let $(\xi_t,\eta_t)_{t\ge0}$ be a bivariate Lévy process. A Lévy process is the stationary-independent-increment cousin of Brownian motion; unlike SBM, it may contain jumps. The GOU is defined by

$$
X_t
=
m(1-e^{-\xi_t})
+
e^{-\xi_t}\int_0^t e^{\xi_{s-}}\,d\eta_s
+
X_0e^{-\xi_t},
\qquad t\ge0
$$

This is the same mental picture as before: old information is discounted, new noise is accumulated, and the process is pulled toward a level. The difference is that the discounting path $\xi$ and the innovation path $\eta$ can jump and can be dependent.

| Object | Ordinary Gaussian OU | Generalised OU |
|---|---|---|
| Discount factor | $e^{-\gamma t}$ | $e^{-\xi_t}$ |
| Noise source | $\sigma\,dB_t$ | $d\eta_t$, possibly with jumps |
| Integral | Itô integral against Brownian motion | Semimartingale integral with left limit $\xi_{s-}$ |
| Stationarity driver | $\gamma>0$ and invariant normal law | Convergence of an exponential Lévy integral |


The survey's stationarity message is: for GOU, the question becomes whether an exponential Lévy integral converges. In ordinary OU, this reduces to the simple condition $\gamma>0$.

## 8. Summary

- The OU process is standard Brownian motion plus a linear restoring drift.
- The explicit solution turns stochastic calculus into a Gaussian transition formula.
- Equal-grid samples of a Gaussian OU are exactly AR(1), not merely approximately AR(1).
- Simulation, likelihood evaluation, and parameter learning all follow from the same transition law.
- The Generalised OU process keeps the discounted-integral structure but allows Lévy-driven jumps and richer stationarity behavior.

## References

- Maller, R. A., Müller, G., and Szimayer, A. _Ornstein-Uhlenbeck Processes and Extensions_. Main source for OU, GOU, stationarity, discretisation, and statistical issues.
- Uhlenbeck, G. E. and Ornstein, L. S. (1930). _On the theory of Brownian motion_. Original OU model.
- Vasicek, O. A. (1977). _An equilibrium characterisation of the term structure_. A classical finance application of Gaussian OU dynamics.
