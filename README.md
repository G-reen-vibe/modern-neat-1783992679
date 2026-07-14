# Modern NEAT: Criticality-Regulated Information-Theoretic NeuroEvolution

A research project reimagining NEAT (NeuroEvolution of Augmenting Topologies)
for modern RL benchmarks. The goal is a **new fundamental algorithm** — not
a hybrid — that preserves NEAT's principles (minimal starting structure,
topological search, diversity preservation) while being measurably faster
and more sample-efficient than canonical NEAT.

## Repository layout
- `src/` — core algorithm implementations
  - `genome.py` — genome representation shared by all algorithms
  - `innovation.py` — innovation registry
  - `evaluator.py` — RL evaluation harness (gymnasium)
  - `baseline_neat.py` — canonical NEAT (Stanley & Miikkulainen 2002)
- `scripts/` — experiment drivers
- `experiments/` — ablation / comparison configs
- `results/` — JSON logs, plots

## Methodology
- Multiple baselines: canonical NEAT, fixed-topology GA, random search
- Multiple benchmarks: CartPole-v1, MountainCar-v0, Acrobot-v1
- Multiple seeds with confidence intervals

## Status
- Phase 0: setup in progress
