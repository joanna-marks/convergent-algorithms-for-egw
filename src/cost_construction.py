
from __future__ import annotations

from typing import Any

import numpy as np


def sqdist(z: np.ndarray) -> np.ndarray:
    """ Compute pairwise squared distance """
    return ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)


def entropy_term(P: np.ndarray) -> float:
    """ Compute entropy in a numerically stable way """
    P_safe = np.maximum(P, 1e-300)
    return float(np.sum(P_safe * np.log(P_safe)))


def egw_objective(P: np.ndarray, Cx: np.ndarray, Cy: np.ndarray, wtx: np.ndarray, wty: np.ndarray, reg: float) -> float:
    """ Compute the interaction term value, plus marginal and interaction entropy """
    quadratic = -2.0 * np.sum((Cx @ P @ Cy.T) * P)
    entropy = reg * entropy_term(P)
    marginal_entropy = -reg * (entropy_term(wtx) + entropy_term(wty))
    return float(quadratic + entropy + marginal_entropy)


def gw_marginal_constant(Cx: np.ndarray, Cy: np.ndarray, wtx: np.ndarray, wty: np.ndarray) -> float:
    """ Compute the constant gw terms """
    return float(wtx @ (Cx**2) @ wtx + wty @ (Cy**2) @ wty)


def amd_theoretical_step(Cx: np.ndarray, Cy: np.ndarray, reg: float) -> float:
    """ Theoretical AMD step size from Theorem 4"""
    m_max_norm = 4.0 * float(np.max(Cx)) * float(np.max(Cy))
    return float(min(reg / (2.0 * (m_max_norm + reg)), 1.0))


def random_weights(rng: np.random.Generator, n_points: int, kind: str) -> np.ndarray:
    """ Compute random or uniform weights"""
    if kind == "random":
        weights = rng.random(n_points)
        return weights / weights.sum()
    if kind == "uniform":
        return np.ones(n_points) / n_points
    raise ValueError("data.weights must be 'random' or 'uniform'.")

def lag2_frobenius(couplings: np.ndarray) -> np.ndarray:
    """ Compute coupling lag-2 frobenius norm of """
    couplings = np.asarray(couplings)
    return np.linalg.norm(couplings[2:] - couplings[:-2], axis=(1, 2))


def build_cost_matrices(x: np.ndarray, y: np.ndarray, data: dict[str, Any]):
    """Build the intra domain cost matrices Cx, Cy for the selected cost type"""
    power = data.get("cost_power", 2)
    if power not in (2, 6):
        raise ValueError("data.cost_power must be 2 or 6.")
    
    if data.get("distance", "squared_euclidean") == "spherical_geodesic":
        # geodesic dsitance
        if power != 2:
            raise ValueError("data.cost_power=6 applies to Euclidean costs, not spherical_geodesic.")
        Cx = np.arccos(np.clip(x @ x.T, -1.0, 1.0))
        Cy = np.arccos(np.clip(y @ y.T, -1.0, 1.0))
    else:
        #euclidean distance
        Cx = sqdist(x) ** (power // 2)
        Cy = sqdist(y) ** (power // 2)

    # scale cost values
    cost_scale = data.get("cost_scale")
    if cost_scale is not None:
        scale_by = str(data.get("cost_scale_by", "median"))
        if scale_by == "median":
            denominator = max(float(np.median(Cx)), float(np.median(Cy)))
        elif scale_by == "max":
            denominator = max(float(Cx.max()), float(Cy.max()))
        else:
            raise ValueError("data.cost_scale_by must be 'median' or 'max'.")
        scale = float(cost_scale) / denominator
        Cx *= scale
        Cy *= scale

    return Cx, Cy
