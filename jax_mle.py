import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, binom
import numpy as np
from numpy.polynomial.hermite import hermgauss
from jax.scipy.optimize import minimize

#  Gauss-Hermite Quadrature 
def get_gh_nodes_weights(deg=50):
    nodes, weights = hermgauss(deg)
    nodes_std = nodes * np.sqrt(2.0)
    weights_std = weights / np.sqrt(np.pi)
    return jnp.array(nodes_std), jnp.array(weights_std)

nodes, weights = get_gh_nodes_weights(50)

#  Conditional Default Probability 
def cond_default_prob(rho, pd, x):
    """
    rho: asset correlation in [0,1]
    pd: marginal default probability
    x: systemic factor
    """
    rho = jnp.clip(rho, 1e-6, 1 - 1e-6)
    inv = norm.ppf(pd)
    arg = (inv - jnp.sqrt(rho) * x) / jnp.sqrt(1 - rho)
    return norm.cdf(arg)

#  Log-Likelihood Given X (log-space for numerical stability)
def log_likelihood_given_x(d, n, rho_vec, pd_vec, x):
    """
    Log-likelihood of observing default counts d out of n obligors,
    given systemic factor x and asset correlations rho_vec.
    Uses binom.logpmf to avoid underflow with large n.
    """
    probs = jnp.array([cond_default_prob(rho_vec[k], pd_vec[k], x) for k in range(len(d))])
    probs = jnp.clip(probs, 1e-6, 1 - 1e-6)
    log_pmfs = binom.logpmf(d, n, probs)
    # Cap at -500 to prevent -inf which causes NaN gradients
    return jnp.sum(jnp.maximum(log_pmfs, -500.0))

#  Log-Marginal Likelihood (Integrated over X via log-sum-exp)
def log_marginal_likelihood(d, n, rho_vec, pd_vec):
    """
    log P(d | rho) = log ∫ P(d|x,rho) phi(x) dx
    Evaluated via Gauss-Hermite quadrature with log-sum-exp trick.
    """
    def log_integrand(x):
        return log_likelihood_given_x(d, n, rho_vec, pd_vec, x)
    log_vals = jax.vmap(log_integrand)(nodes)  # (deg,)
    # log-sum-exp: log( sum w_i * exp(log_val_i) )
    log_weights = jnp.log(weights)
    return jax.scipy.special.logsumexp(log_vals + log_weights)

#  Log-Likelihood 
def log_likelihood(rho_vec, D, N_obligors, pd_vec):
    """
    Total log-likelihood over all observations.
    D: (N, K) array of default counts
    N_obligors: (N, K) array of obligor counts
    rho_vec: (K,) asset correlations
    pd_vec: (K,) marginal PDs
    """
    def log_lik_one(d, n):
        return log_marginal_likelihood(d, n, rho_vec, pd_vec)
    log_likes = jax.vmap(log_lik_one)(D, N_obligors)
    return jnp.sum(log_likes)

# Sigmoid reparameterization: ρ = sigmoid(θ) to enforce ρ ∈ (0, 1) during unconstrained optimization
def _rho_to_theta(rho):
    """Map ρ ∈ (0,1) → θ ∈ ℝ  (logit)"""
    rho = jnp.clip(rho, 1e-6, 1 - 1e-6)
    return jnp.log(rho / (1 - rho))

def _theta_to_rho(theta):
    """Map θ ∈ ℝ → ρ ∈ (0,1)  (sigmoid)"""
    return jax.nn.sigmoid(theta)

#  BFGS (with sigmoid reparameterization)
def fit_rho_bfgs(D, N_obligors, pd_vec, init_rho=None):
    K = D.shape[1]
    if init_rho is None:
        init_rho = jnp.full(K, 0.1)

    init_theta = _rho_to_theta(init_rho)

    def loss(theta):
        rho = _theta_to_rho(theta)
        return -log_likelihood(rho, D, N_obligors, pd_vec)

    result = minimize(loss, init_theta, method='BFGS')
    return _theta_to_rho(result.x)

#  Gradient Descent
def fit_rho_gd(D, N_obligors, pd_vec, init_rho=None, lr=0.01, steps=1000):
    K = D.shape[1]
    if init_rho is None:
        init_rho = jnp.full(K, 0.1)

    init_theta = _rho_to_theta(init_rho)

    def loss(theta):
        rho = _theta_to_rho(theta)
        return -log_likelihood(rho, D, N_obligors, pd_vec)

    grad_loss = jax.grad(loss)
    theta = init_theta.copy()

    for i in range(steps):
        g = grad_loss(theta)
        theta = theta - lr * g
        if i % 100 == 0:
            print(f"Step {i}, loss = {loss(theta):.4f}, rho = {_theta_to_rho(theta)}")

    return _theta_to_rho(theta)