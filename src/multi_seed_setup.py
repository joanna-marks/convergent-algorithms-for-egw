"""Adapters for comparing amd/dual_variational/variational/accelerated on one shared
point-cloud problem."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from . import amd, variational_solvers
from .matrix_construction import (
    construct_B0_B1_efficient, construct_B0_B1_memory_efficient, compute_L, compute_accelerated_L,
)
from .cost_construction import amd_theoretical_step, egw_objective
from .utilis import prepare_experiment

METHODS = ("amd", "md", "dual_variational", "variational", "accelerated")
DEFAULT_ALL = ("amd", "dual_variational", "variational", "accelerated")


def selected_jobs(algorithm):
    """Expand algorithm.methods/alpha_values into a flat list of (key, label, family, alpha) jobs."""
    methods = algorithm.get("methods", ["amd"])
    if isinstance(methods, str):
        methods = [methods]
    if methods == ["all"]:
        methods = DEFAULT_ALL
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError(f"Choose methods from {METHODS}, or 'all' alone.")
    jobs = []
    for method in methods:
        alphas = algorithm.get("alpha_values", [.1, .5, .9, 1.]) if method == "amd" else [None]
        if method == "md":
            alphas = [1.]
        for alpha in alphas:
            family = "amd" if method == "md" else method
            theoretical = isinstance(alpha, str) and alpha.strip().lower() == "theoretical"
            if theoretical:
                alpha = "theoretical"
            else:
                alpha = None if alpha is None else float(alpha)
                if alpha is not None and (not np.isfinite(alpha) or not 0 < alpha <= 1):
                    raise ValueError("AMD alpha values must be in (0, 1].")
            if family == "amd":
                key = "amd_alpha_theoretical" if theoretical else f"amd_alpha_{alpha:g}"
            else:
                key = family
            # Placeholder labels for the three variational families: run_methods_on_problem
            # overwrites these once dispatch is resolved.
            if family == "amd":
                label = "AMD (theoretical)" if theoretical else (
                    "MD (alpha=1)" if alpha == 1 else f"AMD (alpha={alpha:g})")
            else:
                label = {
                    "dual_variational": "Variational (1)", "variational": "Variational (L)",
                    "accelerated": "Accelerated variational (L)",
                }[family]
            if not any(job[0] == key for job in jobs):
                jobs.append((key, label, family, alpha))
    if not jobs:
        raise ValueError("The method selection produced no runs.")
    return jobs


def marginal_error(P, a, b):
    """Max L1 violation of coupling P's row/column sums against target marginals a, b."""
    return float(max(np.abs(P.sum(1)-a).sum(), np.abs(P.sum(0)-b).sum()))


@contextmanager
def selected_sinkhorn(method, tol, maxiter, errors):
    original_var, original_amd = variational_solvers.Sinkhorn, amd.Sinkhorn
    def solve(C, reg, a, b, **kwargs):
        P = original_var(C, reg, a, b, delta=tol, num_iter=maxiter, method=method)
        if not np.isfinite(P).all():
            raise FloatingPointError("Sinkhorn returned a non-finite coupling.")
        errors.append(marginal_error(P, a, b))
        return P
    variational_solvers.Sinkhorn = amd.Sinkhorn = solve
    try:
        yield
    finally:
        variational_solvers.Sinkhorn, amd.Sinkhorn = original_var, original_amd


def run_method_sweep(config):
    """Validate a config's algorithm settings and run every selected method on the dataset it describes."""
    algorithm = config["algorithm"]
    jobs = selected_jobs(algorithm)
    K, reg = int(algorithm["max_iterations"]), float(algorithm["reg"])
    outer_tol = float(algorithm.get("outer_tol", algorithm.get("tol", 1e-6)))
    if K < 1 or reg <= 0 or not np.isfinite(reg) or outer_tol < 0 or not np.isfinite(outer_tol):
        raise ValueError("Require max_iterations >= 1, finite reg > 0 and finite outer_tol >= 0.")
    representation = algorithm.get("factorization", "memory_efficient")
    if representation not in ("memory_efficient", "dense"):
        raise ValueError("algorithm.factorization must be 'memory_efficient' or 'dense'.")
    threads = int(algorithm.get("threads", 1))
    if threads < 1:
        raise ValueError("algorithm.threads must be positive.")
    with threadpool_limits(limits=threads):
        return _run_methods(config, jobs, K, reg, outer_tol, representation)


def _run_methods(config, jobs, K, reg, outer_tol, representation):
    """Build the problem from `config` via prepare_experiment, then dispatch to run_methods_on_problem."""
    algorithm, saving = config["algorithm"], config.get("saving", {})
    start = perf_counter()
    clouds, Cx, Cy, P_init, offset = prepare_experiment(config)
    a, b = clouds["wtx"], clouds["wty"]
    preparation_seconds = perf_counter()-start
    extra_arrays = dict(clouds)
    if saving.get("save_distance_matrices", False):
        extra_arrays.update(Cx=Cx, Cy=Cy)
    return run_methods_on_problem(
        Cx, Cy, a, b, P_init, offset, jobs, K, reg, outer_tol, representation, algorithm, saving,
        extra_arrays=extra_arrays, preparation_seconds=preparation_seconds,
    )


def run_methods_on_problem(
    Cx, Cy, a, b, P_init, offset, jobs, K, reg, outer_tol, representation, algorithm, saving,
    extra_arrays=None, preparation_seconds=0.0,
):
    """Run every selected method on an already-built (Cx, Cy, P_init) problem; shared by _run_methods and any other caller with its own data pipeline (e.g. Figure 7)."""
    sinkhorn_tol = float(algorithm.get("sinkhorn_tol", 1e-9))
    sinkhorn_maxiter = int(algorithm.get("sinkhorn_max_iterations", 200))
    feasibility_tol = float(algorithm.get("feasibility_tol", 1e-8))
    J = int(algorithm.get("var_inner_steps", 50))
    gamma = algorithm.get("var_inner_step", .5)
    var_L = algorithm.get("variational_L", 2.)
    acc_L = algorithm.get("accelerated_L", 2.)
    gamma, var_L, acc_L = [None if v is None else float(v) for v in (gamma, var_L, acc_L)]
    var_inner_tol = float(algorithm.get("var_inner_tol", 1e-6))
    if min(J, sinkhorn_maxiter) < 1 or not all(
        np.isfinite(v) and v > 0 for v in (gamma, var_L, acc_L, var_inner_tol, sinkhorn_tol, feasibility_tol)
        if v is not None
    ):
        raise ValueError("Inner iteration limits, steps, L values and accuracy tolerances must be positive.")
    backends = algorithm.get("sinkhorn_methods", {})
    for _, _, family, _ in jobs:
        backend = backends.get(family, algorithm.get("sinkhorn_method", "sinkhorn"))
        if backend not in variational_solvers.SINKHORN_SOLVERS:
            raise ValueError(f"Unknown Sinkhorn method for {family}: {backend}")

    arrays = {**(extra_arrays or {}), "P_init": P_init}
    if any(j[2] == "amd" for j in jobs):
        A1, A2 = amd.factorize_for_mirror_descent(Cx)
        B1_md, B2_md = amd.factorize_for_mirror_descent(Cy)
    needs_dual = any(j[2] in ("dual_variational", "variational", "accelerated") for j in jobs)
    if needs_dual:
        if representation == "memory_efficient":
            *factors, eig0, eig1 = construct_B0_B1_memory_efficient(Cx, Cy)
            dual_args = (*factors, a, b)
            accelerated_args = (*factors, eig0, a, b)
        else:
            B0, B1, eig0, eig1 = construct_B0_B1_efficient(Cx, Cy)
            # Match M=-4*kron(Cx,Cy), as in the notebook and AMD objective.
            B0, B1 = np.sqrt(2)*B0, np.sqrt(2)*B1
            eig0, eig1 = 2*eig0, 2*eig1
            dual_args = accelerated_args = (B0, B1, a, b)
        if gamma is None:
            gamma = 1 / (1 + (eig1.max() if eig1.size else 0.) / reg)
        if var_L is None:
            var_L = compute_L(eig0, eig1, reg)
        if acc_L is None:
            acc_L = compute_accelerated_L(eig0, eig1, reg)
        if not all(np.isfinite(v) and v > 0 for v in (gamma, var_L, acc_L)):
            raise ValueError("Spectral steps require positive finite L values and inner step.")
    arrays["preparation_seconds"] = np.asarray(preparation_seconds)
    store_history = bool(saving.get("save_full_history", False))
    results = []
    for key, label, family, alpha in jobs:
        if family == "amd" and alpha == "theoretical":
            alpha = amd_theoretical_step(Cx, Cy, reg)
        if family == "dual_variational":
            label = "Variational (1)"
        elif family == "variational":
            label = "Variational (L)"
        elif family == "accelerated":
            label = "Accelerated variational (L)"
        backend = backends.get(family, algorithm.get("sinkhorn_method", "sinkhorn"))
        dual_family = family in ("dual_variational", "variational", "accelerated")
        result = dict(method=key, label=label, family=family, alpha=alpha,
                      stationarity_kind="dual" if dual_family else "primal",
                      stopping_criterion="dual_gradient_l2_squared" if dual_family else "primal_proposal_l1_squared",
                      stationarity_evaluation=("z_j" if family == "accelerated" else
                                               "u_j" if dual_family else "P_j"),
                      implementation=representation if dual_family else "standard",
                      outer_tol=outer_tol, stationarity=[], objectives=[], objectives_interaction=[], iterations=0,
                      outer_tol_reached=False, converged=False, status="failed", seconds=0.,
                      sinkhorn_method=backend, sinkhorn_calls=0, max_sinkhorn_marginal_l1=None,
                      final_stationarity=None, final_objective=None, final_marginal_l1=None,
                      marginal_l1=[], inner_marginals_ok=None, preparation_seconds=preparation_seconds)
        if family == "dual_variational":
            result.update(var_inner_steps=J, var_inner_tol=var_inner_tol, var_inner_step=float(gamma), outer_step=1.)
        elif family in ("variational", "accelerated"):
            result.update(var_inner_steps=J, var_inner_step=float(gamma), var_inner_tol=var_inner_tol,
                          variational_L=float(var_L), accelerated_L=float(acc_L),
                          outer_step=1/var_L if family == "variational" else None)
        if family == "accelerated":
            result["accelerated_beta"] = float(1 / (2 * acc_L))
        print(f"  Running {label}", flush=True)
        errors = []
        start = perf_counter()
        objectives = []
        objectives_interaction = []
        margins = []
        seen_first = []
        def record_metrics(P):
            if not np.isfinite(P).all():
                raise FloatingPointError("Non-finite coupling history.")
            if not seen_first:
                seen_first.append(True)
                np.testing.assert_allclose(P, P_init, rtol=0, atol=1e-15)
            # offset (gw_marginal_constant, or 0. under objective_mode="interaction") does
            # not depend on P, so tracking both here is free -- no need to re-solve to see
            # the interaction-only curve separately from the full one.
            interaction_value = float(egw_objective(P, Cx, Cy, a, b, reg))
            objectives.append(interaction_value + offset)
            objectives_interaction.append(interaction_value)
            margins.append(marginal_error(P, a, b))
        try:
            with selected_sinkhorn(backend, sinkhorn_tol, sinkhorn_maxiter, errors):
                common = dict(K=K, reg=reg, store_history=store_history, store_scalars=True, metrics_fn=record_metrics)
                if family == "amd":
                    residual = []
                    couplings, proposals = amd.MirrorDescent(
                        reg, a, b, A1, A2, B1_md, B2_md, P_init.copy(), alpha, K, tol=outer_tol,
                        store_history=store_history, metrics_fn=record_metrics, stationarity_fn=residual.append)
                    history = dict(Couplings=couplings, stopping_residual=residual)
                elif family == "dual_variational":
                    solve = variational_solvers.DualSolverVariationalMemoryEfficient if representation == "memory_efficient" else variational_solvers.DualSolverVariational
                    history = solve(*dual_args, J=J, gamma=gamma, inner_tol=var_inner_tol, outer_tol=outer_tol,
                                    if_track=True, **common)
                elif family == "variational":
                    solve = variational_solvers.VariationalSolverMemoryEfficient if representation == "memory_efficient" else variational_solvers.VariationalSolver
                    history = solve(*dual_args, J=J, gamma=gamma, eta=1/var_L, inner_tol=var_inner_tol, outer_tol=outer_tol,
                                    if_track=True, **common)
                else:
                    solve = variational_solvers.AcceleratedVariationalSolverMemoryEfficient if representation == "memory_efficient" else variational_solvers.AcceleratedVariationalSolver
                    history = solve(*accelerated_args, J=J, gamma_v=gamma, L=acc_L, inner_tol=var_inner_tol,
                                    outer_tol=outer_tol, if_track=True, **common)
            result["seconds"] = perf_counter()-start
            couplings = history["Couplings"]
            stationarity = list(map(float, history["stopping_residual"]))
            if not np.isfinite(stationarity+objectives).all():
                raise FloatingPointError("Non-finite objective or stationarity.")
            reached = bool(stationarity and stationarity[-1] < result["outer_tol"])
            accurate_marginals = max(errors, default=0.) <= feasibility_tol and margins[-1] <= feasibility_tol
            status = "outer tolerance reached" if reached else "iteration limit reached"
            result.update(stationarity=stationarity, objectives=objectives,
                          objectives_interaction=objectives_interaction, iterations=len(stationarity),
                          outer_tol_reached=reached, converged=reached and accurate_marginals, status=status,
                          final_stationarity=stationarity[-1], final_objective=objectives[-1],
                          marginal_l1=margins, final_marginal_l1=margins[-1], inner_marginals_ok=accurate_marginals)
            if "inner_converged" in history:
                result["inner_converged"] = history["inner_converged"]
            if family == "amd":
                per_step = [1] * len(stationarity)
            elif "sinkhorn_calls" in history:
                per_step = list(history["sinkhorn_calls"])
            else:
                per_step = None
            if per_step is not None and len(per_step) == len(stationarity):
                result["cumulative_sinkhorn_calls"] = list(map(int, np.concatenate([[0], np.cumsum(per_step)])))
            array_key = key.replace(".", "p")
            if saving.get("save_final_couplings", True):
                arrays[f"final_coupling_{array_key}"] = couplings[-1]
            if saving.get("save_full_history", False):
                arrays[f"couplings_{array_key}"] = np.asarray(couplings)
                if family == "amd":
                    arrays[f"proposals_{array_key}"] = np.asarray(proposals)
                else:
                    for name in ("u", "v", "z"):
                        if name in history:
                            arrays[f"{name}_{array_key}"] = np.asarray(history[name])
        except Exception as error:
            result.update(seconds=perf_counter()-start, status=f"failed: {type(error).__name__}: {error}")
        result.update(sinkhorn_calls=len(errors), max_sinkhorn_marginal_l1=max(errors, default=None))
        results.append(result)
        print(f"  {label}: {result['status']} ({result['seconds']:.2f}s)", flush=True)
    return results, arrays
