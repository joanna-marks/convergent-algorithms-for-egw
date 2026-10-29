from __future__ import annotations

import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np


from .dataset_construction import make_point_clouds
from .cost_construction import build_cost_matrices, gw_marginal_constant, lag2_frobenius

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a dictionary."""
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return config

def alpha_key(alpha: float) -> str:
    """Return a stable, filesystem/NPZ-friendly label for an alpha value."""
    return format(float(alpha), ".15g").replace("-", "m").replace(".", "p")


# Reuses alpha_key's formatting; reg values need the same filesystem/NPZ-safe treatment.
reg_key = alpha_key


def output_directory(config: dict[str, Any], experiment_name: str, suffix: str = "multi_seed") -> Path:
    output_root = Path(config["experiment"].get("output_root", "results"))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_root / experiment_name / f"{timestamp}_{suffix}"


def json_dump(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        def clean(item):
            if isinstance(item, dict):
                return {key: clean(value) for key, value in item.items()}
            if isinstance(item, (list, tuple)):
                return [clean(value) for value in item]
            if isinstance(item, float) and not np.isfinite(item):
                return None
            return item
        json.dump(clean(value), output_file, indent=2, allow_nan=False)


def prepare_experiment(config: dict[str, Any]):
    """Shared data, costs, initialization and objective offset for all solvers."""
    clouds = make_point_clouds(config)
    x = clouds["x"]
    y = clouds["y"]
    wtx = clouds["wtx"]
    wty = clouds["wty"]

    Cx, Cy = build_cost_matrices(x, y, config["data"])
    algorithm = config["algorithm"]
    init_kind = str(algorithm.get("init_coupling", "product"))
    if init_kind == "product":
        P_init = wtx[:, None] * wty[None, :]
    else:
        raise ValueError("algorithm.init_coupling must be 'product'.")
    objective_mode = str(algorithm.get("objective_mode", "interaction"))

    if objective_mode == "full":
        gw_offset = gw_marginal_constant(Cx, Cy, wtx, wty)
    elif objective_mode == "interaction":
        gw_offset = 0.0
    else:
        raise ValueError("algorithm.objective_mode must be 'interaction' or 'full'.")
    return clouds, Cx, Cy, P_init, gw_offset


def augment_lag2_metrics(seed_dir: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    metrics = json.loads((seed_dir / "metrics.json").read_text())
    arrays = dict(np.load(seed_dir / "arrays.npz", allow_pickle=True))
    for result in metrics:
        key = result["method"].replace(".", "p")
        couplings = arrays.get(f"couplings_{key}")
        result["lag2_frobenius"] = (
            lag2_frobenius(couplings).tolist() if couplings is not None and len(couplings) >= 3 else []
        )
    (seed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics, arrays


def pad_trajectories(trajectories: list[list[float]]) -> np.ndarray:
    """Pad variable-length runs with NaN so stopped runs are not extrapolated."""
    width = max(map(len, trajectories), default=0)
    padded = np.full((len(trajectories), width), np.nan)
    for row, values in enumerate(trajectories):
        padded[row, : len(values)] = values
    return padded


def finite_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.where(np.isfinite(values), values, np.nan)
    counts = np.sum(np.isfinite(values), axis=0)
    means = np.full(values.shape[1], np.nan)
    stds = np.full(values.shape[1], np.nan)
    valid = counts > 0
    means[valid] = np.nansum(values[:, valid], axis=0) / counts[valid]
    centered = values[:, valid] - means[valid]
    stds[valid] = np.sqrt(np.nansum(centered**2, axis=0) / counts[valid])
    return means, stds, counts


def log_finite_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        log_values = np.where(values > 0, np.log10(values), np.nan)
    return finite_stats(log_values)


def aggregate_runs(
    runs: list[dict[str, Any]], seeds: list[int]
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Aggregate by method, retaining every seed and its actual stopping length."""
    if not runs or len(runs) != len(seeds):
        raise ValueError("Provide one run per seed.")
    def method_key(result):
        return result.get("method", f"amd_alpha_{result.get('alpha')}")
    keys = [method_key(result) for result in runs[0]["results"]]
    if any([method_key(r) for r in run["results"]] != keys for run in runs):
        raise ValueError("The methods/order changed between seed runs.")
    aggregate = []
    arrays = {"seeds": np.asarray(seeds, dtype=int)}
    for index, key in enumerate(keys):
        results = [run["results"][index] for run in runs]
        first = results[0]
        criterion = (first.get("stationarity_kind", "primal"), first.get("stopping_criterion"))
        if any((r.get("stationarity_kind", "primal"), r.get("stopping_criterion")) != criterion for r in results):
            raise ValueError(f"The stationarity criterion changed between seeds for {key}.")
        alpha = first.get("alpha")
        suffix = f"alpha_{alpha_key(alpha)}" if alpha is not None else key
        entry = {name: first.get(name) for name in
                 ("method", "label", "family", "alpha", "outer_tol", "implementation", "stationarity_evaluation", "stopping_criterion")}
        entry["method"] = key
        entry["color"] = f"C{index % 10}"
        entry["stationarity_kind"] = first.get("stationarity_kind", "primal")
        entry["iterations"] = [int(r["iterations"]) for r in results]
        entry["statuses"] = [r.get("status", "completed") for r in results]
        for metric, source in (("stationarity", "stationarity"),
                               ("stationarity_running_min", "stationarity"),
                               ("objective", "objectives"), ("objective_interaction", "objectives_interaction"),
                               ("marginal", "marginal_l1"),
                               ("sinkhorn_calls", "cumulative_sinkhorn_calls"),
                               ("lag2", "lag2_frobenius")):
            if metric in ("marginal", "sinkhorn_calls", "lag2", "objective_interaction") and not any(
                r.get(source) is not None for r in results
            ):
                continue
            values = [r.get(source) or [] for r in results]
            if metric == "stationarity_running_min":
                values = [np.minimum.accumulate(v).tolist() for v in values]
            raw = pad_trajectories(values)
            mean, std, count = finite_stats(raw)
            stats = dict(raw=raw, mean=mean, std=std, count=count)
            if metric.startswith("stationarity"):
                log_mean, log_std, log_count = log_finite_stats(raw)
                zeros = np.sum(raw == 0, axis=0)
                geometric_mean = 10.**log_mean
                geometric_mean[zeros > 0] = 0.
                # A finite log-standard deviation does not exist when a seed is exactly zero.
                log_std[zeros > 0] = np.nan
                stats.update(log_mean=log_mean, log_std=log_std, log_count=log_count,
                             zero_count=zeros, geometric_mean=geometric_mean)
            for name, value in stats.items():
                arrays[f"{metric}_{name}_{suffix}"] = value
                if name != "raw":
                    entry[f"{metric}_{name}"] = value.tolist()
        attempted = [r for r in results if not r.get("status", "").startswith("not applicable")]
        completed = [r for r in attempted if not r.get("status", "").startswith("failed")]
        seconds = [float(r["seconds"]) for r in completed if r.get("seconds") is not None]
        entry.update(runtime_seconds=[r.get("seconds") for r in results],
                     runtime_mean_seconds=float(np.mean(seconds)) if seconds else None,
                     runtime_std_seconds=float(np.std(seconds)) if seconds else None,
                     runtime_count=len(seconds), failed_count=len(attempted)-len(completed),
                     skipped_count=len(results)-len(attempted),
                     outer_tol_reached_count=sum(bool(r.get("outer_tol_reached")) for r in results),
                     marginal_tolerance_missed_count=sum(r.get("inner_marginals_ok") is False for r in results))
        arrays[f"runtime_seconds_{suffix}"] = np.asarray([r.get("seconds", np.nan) for r in results], dtype=float)
        totals = [r["cumulative_sinkhorn_calls"][-1] for r in completed if r.get("cumulative_sinkhorn_calls")]
        entry.update(total_sinkhorn_calls=[
                         (r["cumulative_sinkhorn_calls"][-1] if r.get("cumulative_sinkhorn_calls") else None)
                         for r in results],
                     total_sinkhorn_calls_mean=float(np.mean(totals)) if totals else None,
                     total_sinkhorn_calls_std=float(np.std(totals)) if totals else None,
                     total_sinkhorn_calls_count=len(totals))
        aggregate.append(entry)
    return aggregate, arrays
