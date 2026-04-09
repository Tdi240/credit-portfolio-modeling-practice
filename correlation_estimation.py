"""
Methods to estimate asset correlation

Given observed default correlations and marginal PDs, recover the latent asset correlation
This is done using the bivariate normal CDF

three solvers: Newton-Raphson, Gauss-Hermite quadrature, and moment-matching.
"""

import numpy as np
from dataclasses import dataclass

from scipy import optimize, stats
from scipy.special import ndtri
from scipy.stats import multivariate_normal



def _bvn_cdf(c, rho):
    """Bivariate normal CDF in scipy"""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return multivariate_normal.cdf([c, c], mean=[0.0, 0.0], cov=cov)


def _bvn_pdf(c, rho): # pdf of biv N, rho is correlation
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return multivariate_normal.pdf([c, c], mean=[0.0, 0.0], cov=cov)


def default_corr_to_target(pd_val, rho_default):
    """Convert (PD, rho_default) to the target value"""
    return rho_default * pd_val * (1.0 - pd_val) + pd_val ** 2



def newton_asset_correlation(pd_val, rho_default, rho0=0.10, tol=1e-10, max_iter=50):
    """
    Solve for asset correlation using Newton-Raphson.
    """
    c = ndtri(pd_val)
    target = default_corr_to_target(pd_val, rho_default)

    rho = np.clip(rho0, 1e-4, 0.999)

    for _ in range(max_iter):
        f_val = _bvn_cdf(c, rho) - target
        f_prime = _bvn_pdf(c, rho)

        if abs(f_prime) < 1e-30:
            return np.nan  # degenerate — density ≈ 0

        step = f_val / f_prime
        rho_new = rho - step

        # Project back into (0, 1)
        rho_new = np.clip(rho_new, 1e-6, 1.0 - 1e-6)

        if abs(f_val) < tol:
            return float(rho_new)

        rho = rho_new

    # Did not converge — return last value if close, else NaN
    return float(rho) if abs(_bvn_cdf(c, rho) - target) < 1e-6 else np.nan


# Gauss–Hermite quadrature solver

def _bvn_cdf_quadrature(c, rho, n_nodes=32):
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    # Hermite nodes are for weight exp(-u²); transform to N(0,1):
    #   t = √2 · u,  φ(t)dt = (1/√π) · exp(-u²) du
    t = np.sqrt(2.0) * nodes
    w = weights / np.sqrt(np.pi)

    sqrt_one_minus_rho_sq = np.sqrt(1.0 - rho ** 2)

    # Conditional: Φ((c - ρ·t) / √(1 - ρ²))  ×  𝟙{t ≤ c}
    conditional = np.where(
        t <= c,
        stats.norm.cdf((c - rho * t) / sqrt_one_minus_rho_sq),
        0.0,
    )
    return float(np.dot(w, conditional))


def quadrature_asset_correlation(pd_val, rho_default, n_nodes=32, tol=1e-10):
   
    c = ndtri(pd_val)
    target = default_corr_to_target(pd_val, rho_default)

    def objective(rho):
        return _bvn_cdf_quadrature(c, rho, n_nodes) - target

    try:
        rho_asset = optimize.brentq(objective, 1e-6, 0.999, xtol=tol)
    except ValueError:
        rho_asset = np.nan
    return float(rho_asset)

    try:
        rho_asset = optimize.brentq(objective, 1e-6, 0.999, xtol=tol)
    except ValueError:
        rho_asset = np.nan
    return float(rho_asset)



def brent_asset_correlation(pd_val, rho_default, tol=1e-10):

    c = ndtri(pd_val)
    target = default_corr_to_target(pd_val, rho_default)

    def objective(rho):
        if rho <= 0:
            # At ρ=0 , always < target → negative residual
            return ndtri(pd_val) ** 2 - target
        return _bvn_cdf(c, rho) - target

    try:
        rho_asset = optimize.brentq(objective, 1e-6, 0.999, xtol=tol)
    except ValueError:
        rho_asset = np.nan
    return float(rho_asset)

    try:
        rho_asset = optimize.brentq(objective, 1e-6, 0.999, xtol=tol)
    except ValueError:
        rho_asset = np.nan
    return float(rho_asset)


# estimateion of all clusters at once

@dataclass
class CorrelationEstimate:
    grade: str
    pd: float
    var_pd: float
    rho_default: float
    rho_asset: float
    method: str


def estimate_asset_correlations(grades, pds, ts_var_by_grade, method="newton", **kwargs):

    solvers = {
        "newton": newton_asset_correlation,
        "quadrature": quadrature_asset_correlation,
        "brent": brent_asset_correlation,
    }
    if method not in solvers:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {list(solvers.keys())}."
        )
    solver = solvers[method]
    results = []

    for g, pd_val in zip(grades, pds):
        var_pd = ts_var_by_grade[g]
        rho_def = var_pd / (pd_val * (1.0 - pd_val))
        rho_def = float(np.clip(rho_def, 0.001, 0.999))
        rho_asset = solver(pd_val, rho_def, **kwargs)
        results.append(
            CorrelationEstimate(
                grade=g,
                pd=float(pd_val),
                var_pd=float(var_pd),
                rho_default=rho_def,
                rho_asset=float(rho_asset),
                method=method,
            )
        )
    return results


def compare_methods(grades, pds, ts_var_by_grade):

    return {
        m: estimate_asset_correlations(grades, pds, ts_var_by_grade, method=m)
        for m in ("bisection", "newton", "quadrature")
    }
