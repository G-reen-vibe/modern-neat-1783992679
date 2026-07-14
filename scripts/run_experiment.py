"""
Experiment driver: run an algorithm on a benchmark across multiple seeds,
log per-generation stats, and persist results to JSON.

Usage:
    python scripts/run_experiment.py --algo neat --env CartPole-v1 \
        --gens 50 --seeds 5 --out results/neat_cartpole.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Make src importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline_neat import NEAT, NEATConfig
from src.evaluator import get_env_info


def build_algo(algo: str, env_name: str, cfg_kwargs: dict, seed: int):
    """Instantiate algorithm with given config."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    # Set per-seed innovation start (deterministic)
    info = get_env_info(env_name)
    # Build env to get dims
    import gymnasium as gym
    env = gym.make(env_name)
    n_in = int(np.prod(env.observation_space.shape))
    if hasattr(env.action_space, 'n'):
        n_out = int(env.action_space.n)
    else:
        n_out = int(np.prod(env.action_space.shape))
    env.close()

    if algo == 'neat':
        cfg = NEATConfig(**cfg_kwargs)
        return NEAT(n_in, n_out, cfg, birth_gen=0)
    raise ValueError(f"Unknown algo: {algo}")


def run_experiment(algo: str, env_name: str, gens: int, n_seeds: int,
                   eval_seeds_per_gen: int, cfg_kwargs: dict,
                   out_path: str, eval_seeds_final: int = 20,
                   verbose: bool = True):
    """Run n_seeds independent runs of `algo` on `env_name`.
    Returns dict with full history.
    """
    info = get_env_info(env_name)
    all_runs = []
    for seed in range(n_seeds):
        algo_inst = build_algo(algo, env_name, cfg_kwargs, seed=seed)
        history = []
        t0 = time.time()
        for gen in range(gens):
            # Use 2 eval seeds per gen to keep cost down; final eval uses more
            eval_seeds = [seed * 100 + gen * 7 + k for k in range(eval_seeds_per_gen)]
            stats = algo_inst.step(env_name, eval_seeds)
            history.append(stats)
            if verbose and (gen % 10 == 0 or gen == gens - 1):
                print(f"  [seed {seed}] gen {gen}: best={stats['best']:.1f} "
                      f"mean={stats['mean']:.1f} species={stats['num_species']} "
                      f"complex={stats['avg_complexity']:.1f}")
        elapsed = time.time() - t0
        # Final evaluation of best genome on more seeds
        from src.evaluator import eval_genome
        best_g = max(algo_inst.pop, key=lambda g: g.fitness)
        final_evals = [eval_genome(best_g, env_name, s)
                       for s in range(1000, 1000 + eval_seeds_final)]
        all_runs.append({
            'seed': seed,
            'history': history,
            'elapsed_s': elapsed,
            'final_best_fitness': float(np.mean(final_evals)),
            'final_best_std': float(np.std(final_evals)),
            'final_best_min': float(np.min(final_evals)),
            'final_best_max': float(np.max(final_evals)),
            'final_best_evals': [float(x) for x in final_evals],
            'final_best_complexity': best_g.complexity(),
            'final_best_hidden': best_g.num_hidden(),
        })
        if verbose:
            print(f"  [seed {seed}] DONE in {elapsed:.1f}s, "
                  f"final best mean = {np.mean(final_evals):.1f} "
                  f"± {np.std(final_evals):.1f}")

    result = {
        'algo': algo,
        'env': env_name,
        'gens': gens,
        'n_seeds': n_seeds,
        'cfg': cfg_kwargs,
        'runs': all_runs,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', required=True, choices=['neat'])
    p.add_argument('--env', required=True)
    p.add_argument('--gens', type=int, default=50)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--eval-seeds-per-gen', type=int, default=2)
    p.add_argument('--pop', type=int, default=80)
    p.add_argument('--out', required=True)
    p.add_argument('--quiet', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = {'pop_size': args.pop}
    run_experiment(args.algo, args.env, args.gens, args.seeds,
                   args.eval_seeds_per_gen, cfg, args.out,
                   verbose=not args.quiet)
