import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.stats import norm  # norm.cdf, .pdf
import numpy as np
from scipy.optimize import minimize  # alt to jax
from numpy.polynomial.hermite import hermgauss  # Gauss-Hermite quadrature


def cond_default_prob_gc(p, m_pd, x):
    """
    One factor Gaussian copula model
    Returns P(D=d | X=x)
    """
    inv_norm_m_pd = norm.ppf(m_pd)  # quantile fc - PD marginals
    denom = jnp.sqrt(1 - p**2)  
    arg = (inv_norm_m_pd - p * x) / denom
    return norm.cdf(arg)



def likelihood_given_x(d, p_vec, m_pd_vec, x):   
    """
    Likelihood given x
    Returns P(D=d | X=x)
    """
    cond_probs = jnp.array([cond_default_prob_gc(p_vec[k], m_pd_vec[k], x) for k in range(len(d))])
    # for each cluster: if d[k]==1, mult by cond_probs[k]; otherwise by (1-cond_probs[k])
    likelihood = jnp.prod(jnp.where(d==1, cond_probs, 1-cond_probs))
    return likelihood


def gauss_hermite_integral(f, deg=50):
    """
    Approximate integral f(x) phi(x) dx using Gauss-Hermite quadrature.
    """
    nodes, weights = hermgauss(deg)
    # nodes are roots of Hermite polynomial, weights are for integral f(x) e^{-x^2} dx

    # But phi(x) = (1/sqrt(2pi)) e^{-x^2/2}. So we need to adjust.
    # Actually, Gauss-Hermite quadrature gives integral f(x) e^{-x^2} dx approx sum w_i f(x_i).
    # For phi(x) = e^{-x^2/2} / sqrt(2pi), we set f(x) = g(x) * e^{-x^2/2}? 

    # Easier:
    # integral g(x) phi(x) dx = (1/sqrt(2pi)) integral g(x) e^{-x^2/2} dx.

    # notation must be updated, can be a bit confusing - maybe I will put in md in the notebook

    # Change variable: let y = x/sqrt(2), then e^{-x^2/2} = e^{-y^2} and dx = sqrt(2) dy.
    # Then integral g(x) phi(x) dx = (1/sqrt(pi)) integral g(sqrt(2) y) e^{-y^2} dy.
    # Gauss-Hermite quadrature directly approximates integral h(y) e^{-y^2} dy approx sum w_i h(y_i).
    # So we can use nodes_y = nodes / sqrt(2), and weights = weights / sqrt(pi).

    nodes_y = nodes * jnp.sqrt(2)
    weights_adj = weights / jnp.sqrt(jnp.pi)
    return jnp.sum(weights_adj * f(nodes_y))

# Precompute Gauss-Hermite nodes and weights for standard normal phi(x)
def gauss_hermite_std_normal(deg):
    nodes, weights = hermgauss(deg)
    # Transform to N(0,1): nodes_std = nodes * sqrt(2)
    # Because original nodes are for integral f(x) e^{-x^2} dx.
    # We want integral f(x) phi(x) dx with phi(x)=e^{-x^2/2}/sqrt(2pi).
    # Using substitution z = x/sqrt(2), we get integral f(sqrt(2) z) e^{-z^2} dz / sqrt(pi).
    # Then Gauss-Hermite gives sum w_i f(sqrt(2) z_i) / sqrt(pi).
    nodes_std = nodes * np.sqrt(2.0)
    weights_std = weights / np.sqrt(np.pi)
    return nodes_std, weights_std

nodes, weights = gauss_hermite_std_normal(50) 
nodes = jnp.array(nodes)
weights = jnp.array(weights)

def marginal_likelihood_one(d, p_vec, m_pd_vec): #marginal likelihood for one cluster
    """
    Returns P(D=d) = integral P(D=d|x) phi(x) dx
    """
    def integrand(g):
        return likelihood_given_x(d, p_vec, m_pd_vec, g)
    # Quadrature sum
    # vmap to evaluate integrand at all nodes simultaneously
    vals = jax.vmap(integrand)(nodes)   # shape (deg,)
    return jnp.sum(weights * vals)    


def log_likelihood(p_vec, D, m_pd_vec): # log-likelihood for the whole dataset
    """
    p_vec: vector of factor loadings (K,)
    D: array of defaults (N, K)
    m_pd_vec: marginal default probs (K,)
    Returns total log-likelihood (scalar)
    """
    N = D.shape[0]
    # marginal likelihood for each observation
    def log_lik_one(d):
        return jnp.log(marginal_likelihood_one(d, p_vec, m_pd_vec) + 1e-12) # to avoid log(0)
    log_like_vals = jax.vmap(log_lik_one)(D) # shape (N,)
    return jnp.sum(log_like_vals)


def fit_factor_loadings_gd(D, m_pd_vec, init_p=None, lr=0.01, steps=10000):
    """
    Fit factor loadings via gradient descent on negative log-likelihood.
    """
    K = D.shape[1]
    if init_p is None:
        init_p = jnp.zeros(K) + 0.3

    loss_fn = lambda p: -log_likelihood(p, D, m_pd_vec)
    grad_loss = jax.grad(loss_fn)

    p_vec = init_p.copy()
    for i in range(steps):
        gl = grad_loss(p_vec)
        p_vec = p_vec - lr * gl
        p_vec = jnp.clip(p_vec, -0.99, 0.99)
        if i % 100 == 0:
            print(f'Step {i}, loss = {loss_fn(p_vec):.4f}')
    return p_vec


def fit_factor_loadings_bfgs(D, m_pd_vec, init_p=None):
    """
    Fit factor loadings via L-BFGS from JAX.
    """
    from jax.scipy.optimize import minimize as jax_minimize

    K = D.shape[1]
    if init_p is None:
        init_p = jnp.zeros(K) + 0.3

    loss_fn = lambda p: -log_likelihood(p, D, m_pd_vec)
    result = jax_minimize(loss_fn, init_p, method='BFGS')
    return result.x