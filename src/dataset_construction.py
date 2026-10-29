from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from .cost_construction import random_weights, sqdist, gw_marginal_constant

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

""" EXAMPLE RECONSTRUCTION
This code below creates functions needed to create a reconstruction of Figure 7 in Houry et al 2026. 
Since the exact reconstruction code was not avalilible, this has been created using ChatGPT's codex and was based on the orignal image.
This is hence an approximation of Figure 7 rather then it's exact replication.
"""


X_PIXELS = [[105,131], [120,138], [98,197], [144,202], [126,225],
            [114,238], [133,242], [145,230], [187,219], [180,250],
            [203,234], [233,266], [211,150], [214,139], [122,238]]
Y_PIXELS = [[244,48], [260,43], [279,50], [282,60], [267,74],
            [230,88], [290,87], [284,103], [304,109], [291,119],
            [255,119], [245,140], [344,115], [344,130], [313,25]]


def example_point_clouds():
    if len(X_PIXELS) != 15 or len(Y_PIXELS) != 15:
        raise ValueError("The Figure 7 reconstruction must contain 15 points in each cloud.")
    return (np.array(X_PIXELS, dtype=float) * [1, -1] / 150,
            np.array(Y_PIXELS, dtype=float) * [1, -1] / 150)



""" SPHERE GENERATION
The functions below are used to sample sample on points on the sphere
"""

def sample_von_mises_fisher(
    rng: np.random.Generator, mean_direction: np.ndarray, kappa: float, n_points: int
) -> np.ndarray:
    """Sample the three-dimensional vMF distribution used in the notebook."""
    mean_direction = np.asarray(mean_direction, dtype=float)
    mean_direction /= np.linalg.norm(mean_direction)
    if kappa < 1e-12:
        points = rng.normal(size=(n_points, 3))
        return points / np.linalg.norm(points, axis=1, keepdims=True)

    u = rng.random(n_points)
    w = np.logaddexp(
        np.log1p(-u) - kappa,
        np.log(u) + kappa,
    ) / kappa
    phi = 2.0 * np.pi * rng.random(n_points)
    radial = np.sqrt(np.maximum(0.0, 1.0 - w**2))
    points = np.column_stack((radial * np.cos(phi), radial * np.sin(phi), w))

    north = np.array([0.0, 0.0, 1.0])
    cross = np.cross(north, mean_direction)
    cosine = float(np.dot(north, mean_direction))
    sine = np.linalg.norm(cross)
    if sine < 1e-12:
        if cosine > 0:
            return points
        return points * np.array([1.0, -1.0, -1.0])
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    rotation = np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
    return points @ rotation.T


def random_rotation_matrix(rng: np.random.Generator, dim: int = 3) -> np.ndarray:
    """A Haar-uniform random rotation matrix (proper, det=+1) via the QR trick."""
    a = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    d = np.sign(np.diag(r))
    d[d == 0] = 1.0
    q = q * d
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q



def make_point_clouds(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """ Create datasets and the associated weights"""
    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    data = config["data"]
    dataset = data.get("type", "ring_gaussians")
    arrays: dict[str, np.ndarray]

    # Sample points for the Gaussian--ring example
    if dataset == "ring_gaussians":
        n_points = int(data["n_points"])
        dimension = int(data["dimension"])
        if dimension != 2:
            raise ValueError("The ring and Gaussians experiment requires dimension = 2.")
        x = rng.normal(float(data["gaussian_mean"]), float(data["gaussian_sigma"]), size=(n_points, dimension))
        angles = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
        radius = float(data["ring_radius"])
        y = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        arrays = {"x": x, "y": y}

    #Sample points for the Gaussian mixture example
    elif dataset == "gaussian_mixture":
        n_components = int(data["n_components"])
        nx, ny = int(data["n_points_x"]), int(data["n_points_y"])
        dx, dy = int(data["dimension_x"]), int(data["dimension_y"])
        concentration = float(data.get("dirichlet_concentration", 2.0))
        probs_x = rng.dirichlet(concentration * np.ones(n_components))
        probs_y = rng.dirichlet(concentration * np.ones(n_components))
        centers_x = float(data["center_scale_x"]) * rng.normal(size=(n_components, dx))
        centers_y = float(data["center_scale_y"]) * rng.normal(size=(n_components, dy))
        labels_x = rng.choice(n_components, size=nx, p=probs_x)
        x = centers_x[labels_x] + float(data["sigma_x"]) * rng.normal(size=(nx, dx))
        labels_y = rng.choice(n_components, size=ny, p=probs_y)
        y = centers_y[labels_y] + float(data["sigma_y"]) * rng.normal(size=(ny, dy))
        arrays = {"x": x, "y": y, "labels_x": labels_x, "labels_y": labels_y,
                  "centers_x": centers_x, "centers_y": centers_y,
                  "component_probs_x": probs_x, "component_probs_y": probs_y}
        
    # Sample points for the sphere example
    elif dataset == "sphere":
        n_points = int(data["n_points"])
        direction = np.asarray(data["mean_direction"], dtype=float)
        kappa = float(data["kappa"])
        x = sample_von_mises_fisher(rng, direction, kappa, n_points)
        if data.get("rotate_y", False):
            rotation = random_rotation_matrix(rng)
            y = x @ rotation.T
        else:
            direction_y = np.asarray(data.get("mean_direction_y", data["mean_direction"]), dtype=float)
            kappa_y = float(data.get("kappa_y", data["kappa"]))
            y = sample_von_mises_fisher(rng, direction_y, kappa_y, n_points)
        arrays = {"x": x, "y": y}

    # Generate the point clouds for the Figure 7 reconstruction example
    elif dataset == "example_reconstruction":
        x, y = example_point_clouds()
        arrays = {"x": x, "y": y}
    else:
        raise ValueError(f"Unknown data.type: {dataset!r}")

    ### generate weights
    default_weights = "uniform" if dataset == "example_reconstruction" else "random"
    kind = str(data.get("weights", default_weights))
    arrays["wtx"] = random_weights(rng, len(x), kind)
    if dataset == "sphere" and data.get("rotate_y", False):
        arrays["wty"] = arrays["wtx"].copy()
    else:
        arrays["wty"] = random_weights(rng, len(y), kind)
    return arrays


