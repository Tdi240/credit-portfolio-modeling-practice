import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, binom
import numpy as np
from numpy.polynomial.hermite import hermgauss
from jax.scipy.optimize import minimize
from functools import partial

"""
Utilities to compute the log-likelihood and perform MLE of asset correlations
in one‑factor Gaussian latent‑variable default model using Gauss–Hermite
quadrature to integrate out the systemic factor.

We observe counts of defaults d_k out of n_k obligors for several clusters
and periods. The marginal likelihood is obtained by integrating the conditional
binomial likelihood over X with density phi(x) (N(0,1)). Gauss–Hermite
quadrature approximates that integral.
"""

# Gauss–Hermite Quadrature (configurable degree)
def get_gh_nodes_weights(deg: int = 50):
    """
    Return Gauss–Hermite quadrature nodes and weights adapted for
    integrals wrt N(0,1) density.

    hermgauss(deg) returns nodes t_j and weights w_j 

    To approximate int f(x) phi(x) dx where phi is the N(0,1) density,
    use change of variables x = sqrt2 t, which gives:
        int f(x) phi(x) dx = (1/sqrt(pi)) int f(sqrt2 t) e^{-t^2} dt

    The transformed nodes are x_j = sqrt2 t_j and weights are w_j' = w_j / sqrt(pi).

    Returns:
        nodes_std: jnp.array of transformed nodes x_j
        weights_std: jnp.array of transformed weights w_j'
    """
    nodes, weights = hermgauss(deg)
    nodes_std = nodes * np.sqrt(2.0)
    weights_std = weights / np.sqrt(np.pi)
    return jnp.array(nodes_std), jnp.array(weights_std)


# Conditional Default Probability (vectorised over clusters)
@partial(jax.jit, static_argnames=[])
def cond_default_prob(rho_vec, pd_vec, x):
    """
    Conditional default probability p_k(x) = P(A_k < c_k | X = x)
    for the one‑factor Gaussian model, vectorised over clusters.

    Parameters
    rho_vec : array-like, shape (K,)
        Asset correlations for each cluster.
    pd_vec : array-like, shape (K,)
        Marginal PDs for each cluster.
    x : scalar
        Systemic factor value.

    Returns
    probs : jnp.array, shape (K,)
        Conditional default probabilities for each cluster.
    """
    rho_vec = jnp.clip(rho_vec, 1e-6, 1 - 1e-6)
    inv = norm.ppf(pd_vec)                     # thresholds (c_k or B, depending on notation)
    arg = (inv - jnp.sqrt(rho_vec) * x) / jnp.sqrt(1 - rho_vec)
    return norm.cdf(arg)


# Log‑Likelihood Given X (vectorised over clusters)
def log_likelihood_given_x(d, n, rho_vec, pd_vec, x):
    """
    Compute the log‑likelihood of observing counts d out of n obligors
    for each cluster, conditional on the systemic factor x.

    For each cluster k, the conditional distribution of d_k given x is
    Binomial(n_k, p_k(x)). Returns the sum of log‑PMFs across clusters.

    Numerical safeguards:
      - Clips probabilities to [1e-6, 1-1e-6] before taking logs.
      - Caps extremely small log‑PMFs at -500 to avoid -inf.
    """
    probs = cond_default_prob(rho_vec, pd_vec, x)
    probs = jnp.clip(probs, 1e-6, 1 - 1e-6)

    # Binomial log‑PMF for each group
    log_pmfs = binom.logpmf(d, n, probs)
    log_pmfs = jnp.maximum(log_pmfs, -500.0)

    return jnp.sum(log_pmfs)



# Log‑Marginal Likelihood (integrated over X via log‑sum‑exp)
def log_marginal_likelihood(d, n, rho_vec, pd_vec, nodes, log_weights):
    """
    Compute log P(d | rho) = log int P(d | x, rho) phi(x) dx
    using pre‑computed Gauss–Hermite quadrature nodes and log‑weights.

    Parameters
    ----------
    d : array-like, shape (K,)
        Observed default counts.
    n : array-like, shape (K,)
        Number of obligors.
    rho_vec : array-like, shape (K,)
        Correlations
    pd_vec : array-like, shape (K,)
        Marginal PDs.
    nodes : array-like, shape (deg,)
        Quadrature nodes.
    log_weights : array-like, shape (deg,)
        Log of quadrature weights.
    """
    def log_integrand(x):
        return log_likelihood_given_x(d, n, rho_vec, pd_vec, x)

    log_vals = jax.vmap(log_integrand)(nodes)   # shape (deg,)
    return jax.scipy.special.logsumexp(log_vals + log_weights)


# Total Log‑Likelihood (summed over periods, JIT)
@partial(jax.jit, static_argnames=('deg',))
def log_likelihood(rho_vec, D, N_obligors, pd_vec, deg=50):
    """
    Total log‑likelihood summed over independent observation periods.

    Params
    rho_vec : array, shape (K,)
        Correlations for each cluster.
    D : array, shape (T, K)
        Observed default counts.
    N_obligors : array, shape (T, K)
        Number of obligors.
    pd_vec : array, shape (K,)
        Marginal PDs for each cluster.
    deg : int, optional
        Number of Gauss–Hermite quadrature nodes (default 50).

    Returns
    total_log_like : scalar
        Sum of log‑marginal likelihoods over all periods.
    """
    nodes, weights = get_gh_nodes_weights(deg)
    log_weights = jnp.log(weights)

    def log_lik_one(d, n):
        # Skip periods where total obligors = 0 (contributs 0 to log‑likelihood)
        valid = jnp.sum(n) > 0
        return jax.lax.cond(
            valid,
            lambda _: log_marginal_likelihood(d, n, rho_vec, pd_vec, nodes, log_weights),
            lambda _: 0.0,
            operand=None
        )

    log_likes = jax.vmap(log_lik_one)(D, N_obligors)
    return jnp.sum(log_likes)


# Sigmoid Reparameterisation (rho in (0,1) equiv to thet in Reals)
def _rho_to_theta(rho):
    """Map rho in (0,1) to unconstrained real theta via logit."""
    rho = jnp.clip(rho, 1e-6, 1 - 1e-6)
    return jnp.log(rho / (1 - rho))

def _theta_to_rho(theta):
    """Map unconstrained theta to rho in (0,1) via sigmoid."""
    return jax.nn.sigmoid(theta)


# Optimiser 1: BFGS with callback support
def fit_rho_bfgs(D, N_obligors, pd_vec, init_rho=None, deg=50,
                 callback=None, options=None):
    """
    Fit asset correlations rho by maximizing the marginal log‑likelihood
    using the BFGS quasi‑Newton optimizer.

    Parameters
    D : array, shape (T, K)
        Default counts.
    N_obligors : array, shape (T, K)
        Obligor counts.
    pd_vec : array, shape (K,)
        Marginal PDs.
    init_rho : array, shape (K,), optional
        Initial guess for rho (default: 0.1 for all clusters).
    deg : int, optional
        Gauss–Hermite quadrature degree.
    callback : callable, optional
        Function called after each iteration, signature callback(xk).
    options : dict, optional
        Additional options passed to jax.scipy.optimize.minimize.

    Returns
    rho_hat : array, shape (K,)
        Estimated asset correlations (in (0,1)).
    """
    K = D.shape[1]
    if init_rho is None:
        init_rho = jnp.full(K, 0.1)

    init_theta = _rho_to_theta(init_rho)

    def loss(theta):
        rho = _theta_to_rho(theta)
        return -log_likelihood(rho, D, N_obligors, pd_vec, deg)

    result = minimize(loss, init_theta, method='BFGS',
                      callback=callback, options=options)
    return _theta_to_rho(result.x)


# Optimiser 2: Gradient Descent  (optional clipping)
def fit_rho_gd(D, N_obligors, pd_vec, init_rho=None, deg=50,
               lr=0.01, steps=1000, grad_clip=None, verbose=True):
    """
    Simple gradient‑descent optimizer for θ = logit(rho).

    Params
    D : array, shape (T, K)
        Default counts.
    N_obligors : array, shape (T, K)
        Obligor counts.
    pd_vec : array, shape (K,)
        Marginal PDs.
    init_rho : array, shape (K,), optional
        Initial guess for rho (default: 0.1 for all clusters).
    deg : int, optional
        Gauss–Hermite quadrature degree.
    lr : float, optional
        Learning rate.
    steps : int, optional
        Number of gradient steps.
    grad_clip : float or None, optional
        If not None, clip gradients to [-grad_clip, grad_clip].
    verbose : bool, optional
        If True, print progress every 100 steps.

    Returns
    rho_hat : array, shape (K,)
        Estimated asset correlations (in (0,1)).
    """
    K = D.shape[1]
    if init_rho is None:
        init_rho = jnp.full(K, 0.1)

    init_theta = _rho_to_theta(init_rho)

    def loss(theta):
        rho = _theta_to_rho(theta)
        return -log_likelihood(rho, D, N_obligors, pd_vec, deg)

    grad_loss = jax.grad(loss)
    theta = init_theta.copy()

    for i in range(steps):
        g = grad_loss(theta)
        if grad_clip is not None:
            g = jnp.clip(g, -grad_clip, grad_clip)
        theta = theta - lr * g

        if verbose and (i % 100 == 0 or i == steps - 1):
            rho_current = _theta_to_rho(theta)
            loss_val = loss(theta)
            print(f"Step {i:4d}, loss = {loss_val:.4f}, rho = {rho_current}")

    return _theta_to_rho(theta)