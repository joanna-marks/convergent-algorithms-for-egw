# Convergent methods for entropic Gromov–Wasserstein

This repository implements and compares iterative solvers for the entropic Gromov–Wasserstein (EGW) problem. It includes:

- **Averaged mirror descent (AMD)**, with a tunable averaging parameter $\alpha$ (for $\alpha = 1$ it is standard mirror descent).
- **Variational solvers** following Rioux et al. (2026): a primal variational method, an accelerated variational method and a dual variational method.

The repository also provides benchmark datasets, experiment configs and plotting utilities, so the solvers can be compared on convergence, runtime and number of Sinkhorn calls over several random seeds.

<!-- ## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
``` -->

## Demo notebook

`demo_notebook.ipynb` runs selected methods on each of the four included datasets:

- Gaussian mixture
- sphere
- Gaussian ring
- the instability example from Houry et al. (2026)

It uses small problem sizes so that it runs quickly.

## Running experiments

### Multi-seed comparisons

```bash
python scripts/run_multi_seed.py gaussian_mixture --methods all --num-seeds 5
python scripts/run_multi_seed.py sphere --methods all --num-seeds 5
python scripts/run_multi_seed.py ring_gaussian --methods amd dual_variational --num-seeds 5
```

`--methods all` runs AMD, dual variational, variational and accelerated variational. AMD runs every value in `alpha_values` from the config; the supplied configs include $\alpha = 1$ (standard mirror descent).

To run one method or a subset, pass any of `amd`, `md`, `dual_variational`, `variational` and `accelerated` to `--methods`. For example:

```bash
python scripts/run_multi_seed.py sphere --methods amd dual_variational --num-seeds 5
```

More seeds give smoother averages but take longer to run.

### Instability example

```bash
python scripts/run_example.py --reg 0.1 0.01
```

This runs both cost powers (6 and 2) for each regularization value $\epsilon \in \{0.1, 0.01\}$. Results are saved under `results/instability_example/`.

### Configuration and outputs

Each dataset has a YAML file in `configs/` that sets dataset sizes, regularization, iteration limits and solver settings. Each run creates a timestamped folder under `results/<dataset>/`. The folder holds per-seed results, averaged metrics, runtime summaries and plots.

<!-- ## Repository structure

| Path | Contents |
| --- | --- |
| `src/amd.py` | Averaged mirror descent |
| `src/variational_solvers.py` | Variational, accelerated variational and dual variational solvers (Rioux et al., 2026) |
| `src/cost_construction.py` | Construction of cost matrices |
| `src/matrix_construction.py` | Cost-matrix decompositions for the dual methods, using the memory-efficient implementation of Rioux et al. (2026) |
| `src/dataset_construction.py` | Dataset generation |
| `src/multi_seed_setup.py` | Multi-seed experiment runner |
| `src/plots.py` | Plotting utilities |
| `scripts/` | Command-line entry points |
| `configs/` | Experiment configurations | -->

## Citation

If you use this code, please cite:

```bibtex
@article{marks2026averageddual,
  title   = {Averaged Mirror Descent and Dual Gradient Methods: Convergent Algorithms for Entropic Gromov-Wasserstein Problems},
  author  = {Joanna Marks and Gabriel Rioux and Riccardo Passeggeri},
  journal = {arXiv preprint arXiv:2609.31848},
  year    = {2026}
}
```
