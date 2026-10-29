#!/usr/bin/env python3
""" Run the instability example"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.plots import save_average_plot, save_transport_snapshots
from src.utilis import aggregate_runs, augment_lag2_metrics, json_dump, load_config, reg_key
from run_multi_seed import run_seeds_for_config
_UNSET = object()


def add_instability_example_extras(power_dir: Path) -> None:
    seed_dir = power_dir / "raw" / "seed_0"
    metrics, arrays = augment_lag2_metrics(seed_dir)
    aggregate, _aggregate_arrays = aggregate_runs([{"seed": 0, "results": metrics}], [0])
    json_dump(power_dir / "averaged_metrics.json", aggregate)
    save_average_plot(aggregate, power_dir)
    save_transport_snapshots(metrics, arrays, arrays["x"], arrays["y"],
                             power_dir / "transport_plans.pdf")


def _step_or_auto(value):
    """'Set up for L parsing"""
    if value.lower() in ("auto", "theoretical", "none", "null"):
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/instability.yaml")
    parser.add_argument("--powers", nargs="+", type=int, default=[6, 2], choices=[2, 6])
    parser.add_argument("--methods", nargs="+",
                        choices=["amd", "md", "dual_variational", "variational", "accelerated", "all"],
                        help="Override algorithm.methods from the config (default: whatever the config specifies, or 'all').")
    parser.add_argument("--iterations", type=int, help="Override outer iteration budget (all methods).")
    parser.add_argument("--eps", type=float, help="Entropic regularization epsilon (single value; see --reg for a sweep).")
    parser.add_argument("--reg", nargs="+", type=float,
                        help="Sweep entropic regularization epsilon over these values instead of a single one "
                        "(overrides --eps and the config's reg), e.g. --reg 0.1 0.01. Each value gets its own "
                        "subdirectory under each power's directory.")
    parser.add_argument("--tol", type=float, help="Outer stopping tolerance (method-dependent residual).")
    parser.add_argument("--sinkhorn-tol", type=float)
    parser.add_argument("--sinkhorn-iterations", type=int)
    parser.add_argument("--factorization", choices=["memory_efficient", "dense"], default="memory_efficient")
    parser.add_argument("--sinkhorn-method",
                        choices=["sinkhorn", "sinkhorn_epsilon_scaling"])
    parser.add_argument("--var-inner-steps", type=int, help="Inner ascent iteration budget J (variational family).")
    parser.add_argument("--var-inner-tol", type=float, help="Inner ascent convergence tolerance.")
    parser.add_argument("--var-inner-step", type=_step_or_auto, default=_UNSET,
                        help="Inner ascent step (gamma), a float or 'auto' for 1/(1+lambda_max(B1^TB1)/reg).")
    parser.add_argument("--variational-L", type=_step_or_auto, default=_UNSET,
                        help="L' for Variational's outer step (eta=1/L'), a float or 'auto' for compute_L.")
    parser.add_argument("--accelerated-L", type=_step_or_auto, default=_UNSET,
                        help="L for Accelerated, a float or 'auto' for compute_accelerated_L = max(1, L'-1).")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    data = config.setdefault("data", {})
    data.setdefault("type", "example_reconstruction")
    algorithm = config.setdefault("algorithm", {})
    if args.methods is not None:
        algorithm["methods"] = args.methods
    algorithm.setdefault("methods", ["all"])
    if args.iterations is not None:
        algorithm["max_iterations"] = args.iterations
    if args.eps is not None:
        algorithm["reg"] = args.eps
    if args.tol is not None:
        algorithm["outer_tol"] = args.tol
    if args.sinkhorn_tol is not None:
        algorithm["sinkhorn_tol"] = args.sinkhorn_tol
    if args.sinkhorn_iterations is not None:
        algorithm["sinkhorn_max_iterations"] = args.sinkhorn_iterations
    if args.sinkhorn_method is not None:
        algorithm["sinkhorn_method"] = args.sinkhorn_method
        algorithm["sinkhorn_methods"] = {}
    if args.var_inner_steps is not None:
        algorithm["var_inner_steps"] = args.var_inner_steps
    if args.var_inner_tol is not None:
        algorithm["var_inner_tol"] = args.var_inner_tol
    if args.var_inner_step is not _UNSET:
        algorithm["var_inner_step"] = args.var_inner_step
    if args.variational_L is not _UNSET:
        algorithm["variational_L"] = args.variational_L
    if args.accelerated_L is not _UNSET:
        algorithm["accelerated_L"] = args.accelerated_L
    algorithm.setdefault("outer_tol", algorithm.get("tol", 0.0))
    algorithm.setdefault("factorization", args.factorization)
    algorithm.setdefault("var_inner_steps", 50)
    algorithm.setdefault("var_inner_tol", 1e-6)
    algorithm.setdefault("var_inner_step", 0.5)
    algorithm.setdefault("variational_L", 2.0)
    algorithm.setdefault("accelerated_L", 2.0)
    algorithm.setdefault("sinkhorn_method", "sinkhorn")
    algorithm.setdefault("feasibility_tol", 1e-8)
    algorithm.setdefault("threads", 1)
    algorithm.setdefault("objective_mode", "full")
    K = int(algorithm["max_iterations"])
    reg_values = args.reg if args.reg is not None else [algorithm["reg"]]
    reg_values = [float(reg) for reg in reg_values]
    if K < 1 or any(reg <= 0 or not np.isfinite(reg) for reg in reg_values):
        raise SystemExit("Require max_iterations >= 1 and finite reg > 0.")

    saving = config.setdefault("saving", {})
    saving["save_full_history"] = True

    output = args.output
    if output is None:
        output = ROOT / "results" / "instability_example" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)

    for power in args.powers:
        for reg in reg_values:
            print(f"=== power={power}, reg={reg} ===", flush=True)
            run_config = copy.deepcopy(config)
            run_config["data"]["cost_power"] = power
            run_config["algorithm"]["reg"] = reg
            run_dir = output / f"power_{power}"
            if len(reg_values) > 1:
                run_dir = run_dir / f"reg_{reg_key(reg)}"
            run_dir.mkdir(parents=True)
            run_seeds_for_config(run_config, f"instability_example_power_{power}_reg_{reg_key(reg)}", [0], run_dir)
            add_instability_example_extras(run_dir)
            metrics = json.loads((run_dir / "raw" / "seed_0" / "metrics.json").read_text())
            for result in metrics:
                print(f"  {result.get('label', result['method'])}: {result['status']}", flush=True)

    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
