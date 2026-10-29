#!/usr/bin/env python3
"""Run one experiment for multiple seeds and aggregate its trajectories."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.multi_seed_setup import run_method_sweep
from src.plots import save_average_plot
from src.utilis import aggregate_runs, json_dump, load_config, output_directory, reg_key


def run_seeds_for_config(
    config: dict[str, Any], experiment_name: str, seeds: list[int], output_dir: Path,
    *, resume: bool = False,
) -> None:
    """Run every seed for one fully-resolved config and write its aggregate outputs.

    Shared by the plain multi-seed run and each reg value of a reg sweep — the only
    difference between those two callers is what output_dir they pass in and whether
    `config["algorithm"]["reg"]` was overridden before calling.
    """
    if resume:
        for seed in seeds:
            seed_dir = output_dir / "raw" / f"seed_{seed}"
            if all((seed_dir / name).exists() for name in ("metrics.json", "arrays.npz", "effective_config.json")):
                previous = json.loads((seed_dir / "effective_config.json").read_text())
                if previous["experiment"]["seed"] != seed or any(
                    previous.get(key) != config.get(key) for key in ("data", "algorithm", "saving")
                ):
                    raise ValueError(f"Cannot reuse seed {seed}: saved experiment settings differ.")
    json_dump(output_dir / "effective_config.json", config)

    runs: list[dict[str, Any]] = []
    resumed_seeds: list[int] = []
    for seed in seeds:
        seed_config = copy.deepcopy(config)
        seed_config.setdefault("experiment", {})["name"] = experiment_name
        seed_config["experiment"]["seed"] = seed
        seed_dir = output_dir / "raw" / f"seed_{seed}"
        files = [seed_dir / name for name in ("metrics.json", "arrays.npz", "effective_config.json")]
        if resume and all(path.exists() for path in files):
            results = json.loads(files[0].read_text())
            resumed_seeds.append(seed)
            print(f"Reusing completed seed {seed}", flush=True)
        else:
            print(f"Running seed {seed}", flush=True)
            results, arrays = run_method_sweep(seed_config)
            seed_dir.mkdir(parents=True, exist_ok=resume)
            json_dump(seed_dir / "effective_config.json", seed_config)
            json_dump(seed_dir / "metrics.json", results)
            np.savez_compressed(seed_dir / "arrays.npz", **arrays)
        runs.append({"seed": seed, "results": results})
        print(f"Completed seed {seed}")

    aggregate, aggregate_arrays = aggregate_runs(runs, seeds)
    json_dump(output_dir / "raw_metrics.json", runs)
    json_dump(output_dir / "averaged_metrics.json", aggregate)
    runtime_fields = ["method", "label", "runtime_mean_seconds", "runtime_std_seconds"]
    with (output_dir / "runtime_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=runtime_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in runtime_fields} for row in aggregate)
    sinkhorn_fields = ["method", "label", "total_sinkhorn_calls_mean", "total_sinkhorn_calls_std"]
    with (output_dir / "sinkhorn_calls_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sinkhorn_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in sinkhorn_fields} for row in aggregate)
    np.savez_compressed(output_dir / "averaged_arrays.npz", **aggregate_arrays)
    json_dump(
        output_dir / "run_info.json",
        {
            "experiment_name": experiment_name,
            "seeds": seeds,
            "number_of_seeds": len(seeds),
            "resumed_seeds": resumed_seeds,
            "reg": config.get("algorithm", {}).get("reg"),
            "distance": config["data"].get("distance", "squared_euclidean"),
            "cost_power": config["data"].get("cost_power", 2),
            "methods": [r["method"] for r in aggregate],
            "runtime_definition": "Solver calls including history and marginal monitoring; excludes shared preparation, objective diagnostics, plots and file I/O. Means exclude failed/skipped solves; their counts and raw elapsed times are retained.",
            "stationarity_definition": "Primal AMD: squared L1 proposal residual. Dual/variational/accelerated: recorded approximate gradient squared L2 (accelerated evaluated at z_j).",
            "converged_definition": "Outer tolerance reached and monitored marginal errors <= feasibility_tol; not a certificate of exact inner optimality or global optimality.",
        },
    )
    if config.get("saving", {}).get("save_plots", True):
        save_average_plot(aggregate, output_dir)


def run_experiment(
    config_path: Path,
    experiment_name: str,
    seeds: list[int],
    reg_values: list[float] | None = None,
    methods: list[str] | None = None,
    factorization: str | None = None,
    sinkhorn_method: str | None = None,
    cost_power: int | None = None,
) -> Path:
    base_config = load_config(config_path)
    if cost_power is not None:
        if cost_power not in (2, 6):
            raise ValueError("cost_power must be 2 or 6.")
        base_config["data"]["cost_power"] = cost_power
    algorithm = base_config.setdefault("algorithm", {})
    if methods is not None:
        algorithm["methods"] = methods
    if factorization is not None:
        algorithm["factorization"] = factorization
    if sinkhorn_method is not None:
        algorithm["sinkhorn_method"] = sinkhorn_method
        algorithm["sinkhorn_methods"] = {}  # Explicit CLI override applies to every method.
    algorithm.setdefault("methods", ["amd"])
    algorithm.setdefault("factorization", "memory_efficient")
    algorithm.setdefault("var_inner_steps", 50)
    algorithm.setdefault("var_inner_step", .5)
    algorithm.setdefault("variational_L", 2.)
    algorithm.setdefault("accelerated_L", 2.)
    algorithm.setdefault("sinkhorn_method", "sinkhorn")
    algorithm.setdefault("feasibility_tol", 1e-8)
    algorithm.setdefault("threads", 1)

    if not reg_values:
        output_dir = output_directory(base_config, experiment_name)
        output_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, output_dir / "config.yaml")
        run_seeds_for_config(base_config, experiment_name, seeds, output_dir)
        return output_dir

    sweep_dir = output_directory(base_config, experiment_name, suffix="reg_sweep")
    sweep_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, sweep_dir / "config.yaml")
    for reg in reg_values:
        reg_config = copy.deepcopy(base_config)
        reg_config.setdefault("algorithm", {})["reg"] = reg
        reg_dir = sweep_dir / f"reg_{reg_key(reg)}"
        reg_dir.mkdir(parents=True)
        run_seeds_for_config(reg_config, experiment_name, seeds, reg_dir)
        print(f"Completed reg={reg}")
    return sweep_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a configured experiment over seeds and save raw and mean/std trajectories."
    )
    parser.add_argument("experiment_name", help="Experiment label and default config filename.")
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config (default: configs/<experiment_name>.yaml).",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", nargs="+", type=int, help="Explicit seeds, e.g. --seeds 0 1 2.")
    seed_group.add_argument("--num-seeds", type=int, default=5, help="Use seeds 0,...,N-1 (default: 5).")
    parser.add_argument(
        "--reg-values",
        nargs="+",
        type=float,
        help="Sweep algorithm.reg over these values instead of using the config's single "
        "reg, e.g. --reg-values 0.1 0.01 0.001. Each reg gets its own seeds, aggregate "
        "stats, and averaged plot, saved side by side under one *_reg_sweep directory.",
    )
    parser.add_argument("--methods", nargs="+",
                        choices=["amd", "md", "dual_variational", "variational", "accelerated", "all"],
                        help="Methods to run (default: algorithm.methods, or amd). amd uses configured alpha_values; "
                        "md runs alpha=1 only.")
    parser.add_argument("--factorization", choices=["memory_efficient", "dense"],
                        help="Representation for the variational solvers (default: memory_efficient).")
    parser.add_argument("--sinkhorn-method", choices=["sinkhorn", "sinkhorn_epsilon_scaling"],
                        help="Override the Sinkhorn backend for all selected methods.")
    parser.add_argument("--cost-power", type=int, choices=[2, 6],
                        help="Euclidean cost power (overrides data.cost_power; default: 2).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if Path(args.experiment_name).name != args.experiment_name:
        raise ValueError("experiment_name must be a single directory-safe name.")
    seeds = args.seeds if args.seeds is not None else list(range(args.num_seeds))
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique.")
    config_path = (args.config or REPO_ROOT / "configs" / f"{args.experiment_name}.yaml").resolve()
    output_dir = run_experiment(
        config_path, args.experiment_name, seeds, reg_values=args.reg_values,
        methods=args.methods, factorization=args.factorization, sinkhorn_method=args.sinkhorn_method,
        cost_power=args.cost_power,
    )
    print(f"Saved multi-seed experiment to: {output_dir}")


if __name__ == "__main__":
    main()
