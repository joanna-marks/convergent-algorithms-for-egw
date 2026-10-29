from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "averaged_md_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

from .cost_construction import build_cost_matrices


def plot_transport(ax, x, y, plan):
    """ Plot a transport plan between two point clouds, with line thickness proportional to the plan's weights."""
    segments = np.array([[a, b] for a in x for b in y])
    weights = np.clip(plan.ravel() * max(len(x), len(y)), 0, 1)
    colors = np.zeros((len(weights), 4))
    colors[:, :3] = 0.25
    colors[:, 3] = weights * 0.8
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=2 * weights))
    ax.scatter(*x.T, c="#38b274", s=28, zorder=3)
    ax.scatter(*y.T, c="#ffda00", s=28, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")


def save_transport_snapshots(metrics: list[dict], arrays: dict, x: np.ndarray, y: np.ndarray,
                             output_path: Path) -> None:
    """One row per method, coupling snapshots at selected iterations."""
    rows = []
    for result in metrics:
        key = result["method"].replace(".", "p")
        couplings = arrays.get(f"couplings_{key}")
        if couplings is not None and len(couplings) >= 1:
            rows.append((result.get("label", result["method"]), np.asarray(couplings)))
    if not rows:
        return
    n_cols = 6
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.2 * n_cols, 2.6 * len(rows)), squeeze=False)
    for row, (label, couplings) in enumerate(rows):
        last = len(couplings) - 1
        selected = sorted({s for s in (0, 1, 2, last - 2, last - 1, last) if 0 <= s <= last})
        for col, step in enumerate(selected):
            plot_transport(axes[row, col], x, y, couplings[step])
            axes[row, col].set_title(f"t={step}")
        for col in range(len(selected), n_cols):
            axes[row, col].axis("off")
        axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                          ha="right", va="center", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def project_for_plotting(points: np.ndarray) -> np.ndarray:
    """Project points to 2D for plotting using svd"""
    if points.shape[1] == 2:
        return points
    if points.shape[1] == 1:
        return np.column_stack([points[:, 0], np.zeros(len(points))])
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def save_point_cloud_plot(arrays: dict[str, np.ndarray], output_path: Path, data=None) -> None:
    x = arrays["x"]
    y = arrays["y"]

    if "Cx" in arrays and "Cy" in arrays:
        Cx, Cy = arrays["Cx"], arrays["Cy"]
    else:
        Cx, Cy = build_cost_matrices(x, y, data or {})

    fig = plt.figure(figsize=(16, 4.5))
    if x.shape[1] == 3 and y.shape[1] == 3:
    # plotting for points on the sphere
        point_cloud_axis = fig.add_subplot(1, 3, 1, projection="3d")
        polar = np.linspace(0.0, np.pi, 15)
        azimuth = np.linspace(0.0, 2.0 * np.pi, 30)
        sphere_x = np.outer(np.cos(azimuth), np.sin(polar))
        sphere_y = np.outer(np.sin(azimuth), np.sin(polar))
        sphere_z = np.outer(np.ones_like(azimuth), np.cos(polar))
        point_cloud_axis.plot_wireframe(
            sphere_x, sphere_y, sphere_z, color="lightgray", linewidth=0.4, alpha=0.5
        )
        point_cloud_axis.scatter(x[:, 0], x[:, 1], x[:, 2], s=10, alpha=0.7, label="X")
        point_cloud_axis.scatter(y[:, 0], y[:, 1], y[:, 2], s=10, alpha=0.7, label="Y")
        point_cloud_axis.set_title("Point Clouds (3D)")
        point_cloud_axis.view_init(elev=20, azim=45)
        if hasattr(point_cloud_axis, "set_box_aspect"):
            point_cloud_axis.set_box_aspect((1, 1, 1))
        point_cloud_axis.legend()
    else:
        #plotting in 2D
        point_cloud_axis = fig.add_subplot(1, 3, 1)
        x_plot, y_plot = project_for_plotting(x), project_for_plotting(y)
        point_cloud_axis.scatter(x_plot[:, 0], x_plot[:, 1], s=12, alpha=0.65, label="X")
        point_cloud_axis.scatter(y_plot[:, 0], y_plot[:, 1], s=12, alpha=0.65, label="Y")
        point_cloud_axis.set_title("Point Clouds")
        point_cloud_axis.set_aspect("equal", adjustable="box")
        point_cloud_axis.legend()

    cx_axis = fig.add_subplot(1, 3, 2)
    cx_im = cx_axis.imshow(Cx, cmap="magma")
    cx_axis.set_title("Cx costs")
    cx_axis.set_xlabel("Index")
    cx_axis.set_ylabel("Index")
    fig.colorbar(cx_im, ax=cx_axis, fraction=0.046, pad=0.04)

    cy_axis = fig.add_subplot(1, 3, 3)
    cy_im = cy_axis.imshow(Cy, cmap="magma")
    cy_axis.set_title("Cy costs")
    cy_axis.set_xlabel("Index")
    cy_axis.set_ylabel("Index")
    fig.colorbar(cy_im, ax=cy_axis, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def method_label(result):
    """Return a label for a method, preferring the user-specified label if present."""
    return result.get("label") or f"alpha={result['alpha']}"


def majority_cutoff(counts: np.ndarray, n_seeds: int) -> int:
    """Return the largest iteration index j such that at least half of the seeds have recorded values for all iterations up to j."""
    valid = counts >= n_seeds / 2
    if valid.all():
        return len(valid)
    return int(np.argmax(~valid))


def stationarity_band(result, running_min=False):
    """Geometric mean and log-std of stationarity values across seeds, up to the majority cutoff."""
    prefix = "stationarity_running_min" if running_min else "stationarity"
    log_mean = np.asarray(result[f"{prefix}_log_mean"], dtype=float)
    log_std = np.asarray(result[f"{prefix}_log_std"], dtype=float)
    cutoff = majority_cutoff(np.asarray(result[f"{prefix}_count"]), len(result["iterations"]))
    center = np.asarray(result.get(f"{prefix}_geometric_mean", 10.**log_mean), dtype=float)
    return (np.arange(1, cutoff+1), center[:cutoff],
            10.**(log_mean[:cutoff]-log_std[:cutoff]), 10.**(log_mean[:cutoff]+log_std[:cutoff]))


def objective_band(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean and std of objective values across seeds, up to the majority cutoff."""
    mean = np.asarray(result["objective_mean"])
    std = np.asarray(result["objective_std"])
    cutoff = majority_cutoff(np.asarray(result["objective_count"]), len(result["iterations"]))
    mean, std = mean[:cutoff], std[:cutoff]
    iterations = np.arange(0, len(mean))
    return iterations, mean, mean - std, mean + std


def stationarity_rate_reference(
    result: dict[str, Any], span: int | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """ Add the O(1/J) reference line"""
    log_mean = result.get("stationarity_log_mean")
    if not log_mean:
        return None
    max_iter = span if span is not None else len(log_mean)
    if max_iter == 0:
        return None
    initial_value = 10.0 ** log_mean[0]
    if not np.isfinite(initial_value) or initial_value <= 0:
        return None
    scale = initial_value * 100.0
    ref_iterations = np.arange(1, max_iter + 1)
    return ref_iterations, scale / ref_iterations


def style_stationarity_axis(axis, kind):
    axis.set_xscale("log")
    values = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in axis.lines]) if axis.lines else np.array([])
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if np.any(finite == 0):
        axis.set_yscale("symlog", linthresh=float(positive.min()/10) if positive.size else 1e-12)
        axis.set_ylim(bottom=0)
    else:
        axis.set_yscale("log")
    axis.set_xlabel("Iterations")
    axis.set_ylabel(r"$\|\tilde{\nabla}\ell_{\varepsilon}(u_j)\|^2$" if kind == "dual"
                    else r"$\|x_j-\tilde{x}_j\|_1^2$")
    axis.set_title(f"{kind.capitalize()} stationarity")
    axis.grid(alpha=.15)


def draw_stationarity(axis, results, kind):
    plotted = False
    span = 0
    for result in results:
        iterations, center, lower, upper = stationarity_band(result)
        if not np.isfinite(center).any():
            continue
        plotted = True
        span = max(span, len(iterations))
        label = method_label(result)
        line = axis.plot(iterations, center, label=label, color=result.get("color"))[0]
        axis.fill_between(iterations, lower, upper, color=line.get_color(), alpha=.2)
        if "stationarity_running_min_log_mean" in result:
            it, running_min, _, _ = stationarity_band(result, running_min=True)
            axis.plot(it, running_min, color=line.get_color(), linestyle=":", linewidth=1.5, alpha=.8)
    if not plotted:
        axis.set_title(f"{kind.capitalize()} stationarity")
        axis.text(.5, .5, "No recorded stationarity\n" + ", ".join(method_label(r) for r in results),
                  transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
        return
    reference = next((stationarity_rate_reference(r, span=span) for r in results
                      if stationarity_rate_reference(r, span=span) is not None), None)
    if reference is not None:
        axis.plot(*reference, label="O(1/j)", linestyle="--", color="black")
    tolerances = sorted({r["outer_tol"] for r in results if r.get("outer_tol", 0) and r["outer_tol"] > 0})
    for tol in tolerances:
        axis.axhline(tol, color="gray", linestyle="-.", label=f"Stopping tolerance")
    axis.plot([], [], color="gray", linestyle=":", label="Running min")
    style_stationarity_axis(axis, kind)


def style_objective_axis(axis):
    axis.set_xscale("log")
    values = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in axis.lines]) if axis.lines else np.array([])
    finite = values[np.isfinite(values)]
    nonzero = np.abs(finite[finite != 0])
    wide_range = nonzero.size > 0 and nonzero.max() >= 10 * nonzero.min()
    if wide_range:
        positive = finite[finite > 0]
        linthresh = float(positive.min()/10) if positive.size else float(nonzero.min()/10)
        if np.any(finite <= 0):
            axis.set_yscale("symlog", linthresh=linthresh)
        else:
            axis.set_yscale("log")
    axis.set_xlabel("Iterations")
    axis.set_ylabel("EGW")
    axis.set_title("Objective")
    axis.grid(alpha=.15)


def draw_objective(axis, aggregate):
    plotted = False
    for result in aggregate:
        iterations, mean, lower, upper = objective_band(result)
        iterations, mean, lower, upper = iterations[1:], mean[1:], lower[1:], upper[1:]
        if not np.isfinite(mean).any():
            continue
        plotted = True
        line = axis.plot(iterations, mean, label=method_label(result), color=result.get("color"))[0]
        axis.fill_between(iterations, lower, upper, color=line.get_color(), alpha=.2)
    if not plotted:
        axis.set_title("Objective")
        axis.text(.5, .5, "No recorded objective updates\n" + ", ".join(method_label(r) for r in aggregate),
                  transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
        return
    style_objective_axis(axis)


def interaction_objective_band(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean and std of the objective's interaction term alone (the marginal constant
    added under objective_mode="full" stripped out), across seeds, up to the majority
    cutoff."""
    mean = np.asarray(result["objective_interaction_mean"])
    std = np.asarray(result["objective_interaction_std"])
    cutoff = majority_cutoff(np.asarray(result["objective_interaction_count"]), len(result["iterations"]))
    mean, std = mean[:cutoff], std[:cutoff]
    iterations = np.arange(0, len(mean))
    return iterations, mean, mean - std, mean + std


def style_interaction_objective_axis(axis):
    axis.set_xscale("log")
    values = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in axis.lines]) if axis.lines else np.array([])
    finite = values[np.isfinite(values)]
    nonzero = np.abs(finite[finite != 0])
    wide_range = nonzero.size > 0 and nonzero.max() >= 10 * nonzero.min()
    if wide_range:
        positive = finite[finite > 0]
        linthresh = float(positive.min()/10) if positive.size else float(nonzero.min()/10)
        if np.any(finite <= 0):
            axis.set_yscale("symlog", linthresh=linthresh)
        else:
            axis.set_yscale("log")
    axis.set_xlabel("Iterations")
    axis.set_ylabel("EGW (interaction term)")
    axis.set_title("Objective (interaction term only)")
    axis.grid(alpha=.15)


def draw_interaction_objective(axis, aggregate):
    plotted = False
    for result in aggregate:
        if "objective_interaction_mean" not in result:
            continue
        iterations, mean, lower, upper = interaction_objective_band(result)
        iterations, mean, lower, upper = iterations[1:], mean[1:], lower[1:], upper[1:]
        if not np.isfinite(mean).any():
            continue
        plotted = True
        line = axis.plot(iterations, mean, label=method_label(result), color=result.get("color"))[0]
        axis.fill_between(iterations, lower, upper, color=line.get_color(), alpha=.2)
    if not plotted:
        axis.set_title("Objective (interaction term only)")
        axis.text(.5, .5, "No recorded interaction-only objective updates",
                  transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
        return
    style_interaction_objective_axis(axis)


def save_interaction_objective_plot(aggregate, output_path) -> None:
    fig, axis = plt.subplots(figsize=(7, 5))
    draw_interaction_objective(axis, aggregate)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def lag2_band(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean and std of the lag-2 coupling difference ||P_k - P_{k-2}||_F across seeds,
    up to the majority cutoff. Only present when the run kept full coupling history."""
    mean = np.asarray(result["lag2_mean"])
    std = np.asarray(result["lag2_std"])
    cutoff = majority_cutoff(np.asarray(result["lag2_count"]), len(result["iterations"]))
    mean, std = mean[:cutoff], std[:cutoff]
    iterations = np.arange(2, 2 + len(mean))
    return iterations, mean, mean - std, mean + std


def style_lag2_axis(axis):
    axis.set_yscale("log")
    axis.set_xlabel("Iterations")
    axis.set_ylabel(r"$\|x_k - x_{k-2}\|_F$")
    axis.set_title("Lag-2 coupling difference")
    axis.grid(alpha=.15)


def draw_lag2(axis, aggregate):
    plotted = False
    for result in aggregate:
        if "lag2_mean" not in result:
            continue
        iterations, mean, lower, upper = lag2_band(result)
        if not np.isfinite(mean).any():
            continue
        plotted = True
        line = axis.plot(iterations, mean, label=method_label(result), color=result.get("color"))[0]
        axis.fill_between(iterations, np.clip(lower, 1e-300, None), upper, color=line.get_color(), alpha=.2)
    if not plotted:
        axis.set_title("Lag-2 coupling difference")
        axis.text(.5, .5, "No recorded lag-2 differences", transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()
        return
    style_lag2_axis(axis)


def save_overlay_plot(aggregate, output_path):
    groups = [(kind, [r for r in aggregate if r.get("stationarity_kind", "primal") == kind])
              for kind in ("primal", "dual")]
    groups = [(kind, results) for kind, results in groups if results]
    has_lag2 = any("lag2_mean" in r for r in aggregate)
    n_panels = len(groups) + 1 + (1 if has_lag2 else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(7*n_panels, 5), squeeze=False)
    for axis, (kind, results) in zip(axes[0], groups):
        draw_stationarity(axis, results, kind)
    draw_objective(axes[0][len(groups)], aggregate)
    if has_lag2:
        draw_lag2(axes[0][len(groups) + 1], aggregate)

    seen = {}
    for axis in axes[0]:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            seen.setdefault(label, handle)

    def annotation_rank(label):
        if label == "O(1/j)":
            return 1
        if label.startswith("Stopping tolerance"):
            return 2
        if label == "Running min":
            return 3
        return 0

    labels = sorted(seen, key=annotation_rank)
    handles = [seen[label] for label in labels]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_average_plot(aggregate, output_dir: Path) -> None:
    save_overlay_plot(aggregate, output_dir / "averaged_metrics.pdf")

    def objective_differs(result):
        full, interaction = result.get("objective_mean"), result.get("objective_interaction_mean")
        if full is None or interaction is None or len(full) != len(interaction):
            return False
        return not np.allclose(full, interaction, equal_nan=True)

    if any(objective_differs(r) for r in aggregate):
        save_interaction_objective_plot(aggregate, output_dir / "averaged_metrics_interaction_objective.pdf")
