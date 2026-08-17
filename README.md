# 3D Root Architecture Simulator

A reproducible scientific-computing framework for stochastic 3D root growth, quantitative architecture analysis, and scalable Monte Carlo experiments. The project combines a deterministic simulation engine, environmental and developmental parameterization, spatial collision constraints, automated scientific-invariant testing, resumable storage, and local or Slurm/HPC execution.

## Engineering Scope

- **Reproducible simulation:** task-indexed random streams make experiments independent of scheduling order.
- **Parameterized experiments:** rainfall, resource concentrations, developmental duration, branching, elongation, thickness, and safety limits are configurable.
- **Spatial algorithms:** continuous 3D axes use curvature limits, geometric collision checks, cylindrical branch-collar clearance, and self-avoidance.
- **Quantitative outputs:** the engine reports architecture, topology, branching, direction, resource, radius, and performance metrics.
- **Regression validation:** 72 simulator/HPC tests cover determinism, scientific invariants, exact schema-v26 hashes, checkpoints, and storage.
- **Scalable execution:** the same engine supports one-off runs, multiprocessing shards, resumable checkpoints, and five-replicate Slurm arrays.

## Model Overview

Development proceeds in discrete biological steps. Every active root tip receives one bounded extension attempt per step. Branch sites arise along continuous material arc through a seeded Poisson process, and accepted branches must satisfy emergence and collision constraints.

Water, phosphorus, nitrogen, and potassium fields influence post-initiation direction and elongation. Cross-sectional transport area determines spatially varying root radius, while deterministic seed and task mapping support controlled Monte Carlo comparisons.

For scientific assumptions, equations, event ordering, and numerical details, see [docs/model_design.md](docs/model_design.md).

## Installation

The validated release environment uses Python 3.14.5. Exact package versions from that environment are recorded in `requirements.txt`; test-only dependencies are recorded in `requirements-dev.txt`.

```bash
git clone https://github.com/luisferangulob/3d-root-architecture-simulator.git
cd 3d-root-architecture-simulator

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

## Run a Simulation

The following command runs 50 developmental steps and prints one JSON result record:

```bash
python single_root_sim.py \
  --mode single \
  --steps 50 \
  --rain-probability 0.50 \
  --branch-probability 0.10 \
  --thickness-increment 0.10 \
  --seed 12345 \
  --max-seconds-per-simulation 0
```

Run the short built-in smoke cases:

```bash
python single_root_sim.py --mode smoke --steps 25 --max-seconds-per-simulation 0
```

Inspect the complete command-line interface:

```bash
python single_root_sim.py --help
```

## Outputs and Metrics

Single mode writes a structured JSON record to standard output. Reported fields include:

- root length, depth, width, and depth-to-width ratio;
- axis, branch, sampled-point, and extension counts;
- topology and Horton-Strahler summaries;
- branching-generation and lateral-age diagnostics;
- direction, curvature, emergence, and collision statistics;
- water and mineral capture metrics;
- per-axis scientific radius profiles;
- stopping status and execution profiling.

Batch mode writes CSV results, a completion bitmap, and run metadata. HPC mode adds atomic checkpoints, progress records, logs, and lossless NumPy result bundles. Generated outputs are excluded from version control.

## Testing

Run the complete simulator and HPC suite from the repository root:

```bash
python -m pip install -r requirements-dev.txt
pytest
```

The suite validates deterministic seeds, branching semantics, resource independence of initiation, 3D geometry, collision behavior, radius propagation, metrics, checkpoint resume, lazy result storage, and Slurm manifest behavior.

A compact regression fixture in `tests/fixtures/schema_v26_regression.json` replaces a duplicated frozen engine. It cryptographically validates the full deterministic scientific result record, exact geometry, topology, radii, selected axis metadata, and branch-site state for a representative schema-v26 case.

## Parameter Sweeps and HPC

The current Schema-v26 fixed parameter grid contains 3,430,350 tasks:

```text
70 thickness increments × 99 rain probabilities
× 99 branch probabilities × 5 replicates
```

Batch mode can restrict work with `--task-start`, `--task-stop`, `--shard-id`, and `--num-shards`. For long runs, `root_hpc_manager.py` creates immutable five-replicate Slurm manifests, and `root_hpc_worker.py` executes checkpointed array tasks.

The provided partition names and X-disk convention are site-oriented defaults, not universal cluster assumptions. Set `ROOT_HPC_RUNS_DIR` to redirect run storage, and adapt allowed partition names for another Slurm environment.

The HPC manifest's `app_path` field is provenance metadata for the application-integrated workflow. The scientific simulation engine does not import or depend on the Streamlit application.

## Repository Structure

```text
3d-root-architecture-simulator/
├── docs/
│   └── model_design.md
├── tests/
│   ├── fixtures/
│   │   └── schema_v26_regression.json
│   └── test_root_architecture_hpc.py
├── .gitignore
├── README.md
├── pytest.ini
├── requirements-dev.txt
├── requirements.txt
├── root_hpc_manager.py
├── root_hpc_storage.py
├── root_hpc_worker.py
└── single_root_sim.py
```

- `single_root_sim.py` contains the simulation, metrics, CLI, and sharded batch runner.
- `root_hpc_manager.py` creates, submits, resumes, cancels, and polls Slurm runs.
- `root_hpc_worker.py` executes one deterministic Slurm-array replicate.
- `root_hpc_storage.py` provides atomic checkpoints and memory-mapped result bundles.
- `tests/` contains simulation, scientific-invariant, storage, and HPC regression tests.

The Streamlit/Plotly application and rendering tests are maintained separately in the companion `root-architecture-visualizer` repository.

## Technical Stack

- Python
- NumPy
- SciPy spatial algorithms
- pytest
- psutil for optional process-memory reporting
- Slurm for optional HPC execution

## Limitations

- The model is a computational abstraction rather than a complete representation of root biology or soil physics.
- Runtime and memory use can grow rapidly with developmental duration, branching, and sampled-point limits.
- Exact numerical reproducibility requires compatible Python, NumPy, and SciPy environments.
- Cluster partitions, storage paths, and resource requests require site-specific configuration.

## Research Context

This software was developed as part of computational research into three-dimensional root system architecture at the University of Arizona. This statement describes the research context and does not imply institutional endorsement.

## Author

**Luis Angulo**
