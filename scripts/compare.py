"""
Unified comparison runner for NEAT and CRIT-NEAT.

Runs both algorithms on a given env with multiple seeds, returns a unified
results JSON. Used to track progress across research rounds.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline_neat import NEAT, NEATConfig
from src.crit_neat import CRITNEAT, CRITConfig
from src.evaluator import eval_genome, get_env_info
import gymnasium as gym


# Per-env solve thresholds
_SOLVE_THRESHOLDS = {
    'CartPole-v1': 475.0,    # >=475 over 500 max
    'CartPole-v0': 195.0,
    'MountainCar-v0': -110.0,  # >=-110 (max is 0)
    'Acrobot-v1': -100.0,      # >=-100 (max is 0)
    'LunarLander-v2': 200.0,
}


def _compute_first_solve(history, env_name):
    thresh = _SOLVE_THRESHOLDS.get(env_name)
    if thresh is None:
        return None
    for h in history:
        if h['best'] >= thresh:
            return h['gen']
    return None


def build_algo(algo: str, env_name: str, cfg_kwargs: dict, seed: int):
    np.random.seed(seed)
    random.seed(seed)
    env = gym.make(env_name)
    n_in = int(np.prod(env.observation_space.shape))
    discrete = hasattr(env.action_space, 'n')
    n_out = int(env.action_space.n) if discrete else int(np.prod(env.action_space.shape))
    env.close()
    if algo == 'neat':
        cfg = NEATConfig(**cfg_kwargs)
        return NEAT(n_in, n_out, cfg, birth_gen=0)
    elif algo == 'crit':
        cfg = CRITConfig(**cfg_kwargs)
        return CRITNEAT(n_in, n_out, cfg, discrete_actions=discrete, birth_gen=0)
    raise ValueError(f"Unknown algo: {algo}")


def run_one(algo: str, env_name: str, gens: int, eval_seeds_per_gen: int,
            cfg_kwargs: dict, seed: int, eval_seeds_final: int = 20,
            verbose: bool = False) -> dict:
    algo_inst = build_algo(algo, env_name, cfg_kwargs, seed=seed)
    history = []
    t0 = time.time()
    for gen in range(gens):
        eval_seeds = [seed * 1000 + gen * 7 + k for k in range(eval_seeds_per_gen)]
        stats = algo_inst.step(env_name, eval_seeds)
        history.append(stats)
        if verbose and (gen % 10 == 0 or gen == gens - 1):
            print(f"    [seed {seed}] gen {gen}: best={stats['best']:.1f} "
                  f"mean={stats['mean']:.1f} sp={stats['num_species']} "
                  f"cx={stats['avg_complexity']:.1f}")
    elapsed = time.time() - t0
    best_g = max(algo_inst.pop, key=lambda g: g.fitness)
    final_evals = [eval_genome(best_g, env_name, s)
                   for s in range(10000, 10000 + eval_seeds_final)]
    print(f"    [seed {seed}] final eval done")
    return {
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
        'first_solve_gen': _compute_first_solve(history, env_name),
    }


def run_comparison(env_name: str, gens: int, n_seeds: int, out_dir: str,
                   pop_size: int = 60, eval_seeds_per_gen: int = 2,
                   algos: list = None, tag: str = None) -> dict:
    """Run multiple algos on env and save results."""
    if algos is None:
        algos = ['neat', 'crit']
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = tag or f"compare_{env_name.replace('-', '_')}"
    results = {'env': env_name, 'gens': gens, 'n_seeds': n_seeds,
               'pop_size': pop_size, 'tag': tag, 'algos': {}}
    for algo in algos:
        print(f"\n=== {algo.upper()} on {env_name} ({n_seeds} seeds, {gens} gens) ===")
        runs = []
        cfg = {'pop_size': pop_size}
        for s in range(n_seeds):
            print(f"  seed {s}...")
            r = run_one(algo, env_name, gens, eval_seeds_per_gen, cfg, s, verbose=True)
            runs.append(r)
            print(f"  seed {s}: final_best={r['final_best_fitness']:.1f} "
                  f"± {r['final_best_std']:.1f}, time={r['elapsed_s']:.1f}s, "
                  f"complexity={r['final_best_complexity']}, "
                  f"first_solve={r['first_solve_gen']}")
        results['algos'][algo] = {
            'runs': runs,
            'mean_final': float(np.mean([r['final_best_fitness'] for r in runs])),
            'std_final': float(np.std([r['final_best_fitness'] for r in runs])),
            'mean_time': float(np.mean([r['elapsed_s'] for r in runs])),
            'mean_complexity': float(np.mean([r['final_best_complexity'] for r in runs])),
            'solve_rate': float(np.mean([1.0 if r['first_solve_gen'] is not None else 0.0
                                          for r in runs])),
        }
        print(f"  Summary: mean={results['algos'][algo]['mean_final']:.1f} "
              f"± {results['algos'][algo]['std_final']:.1f}, "
              f"solve_rate={results['algos'][algo]['solve_rate']*100:.0f}%, "
              f"avg_time={results['algos'][algo]['mean_time']:.1f}s")
    out_path = os.path.join(out_dir, f"{tag}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='CartPole-v1')
    p.add_argument('--gens', type=int, default=30)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--pop', type=int, default=60)
    p.add_argument('--out-dir', default='results')
    p.add_argument('--tag', default=None)
    p.add_argument('--algos', nargs='+', default=['neat', 'crit'])
    p.add_argument('--eval-seeds-per-gen', type=int, default=2)
    args = p.parse_args()
    run_comparison(args.env, args.gens, args.seeds, args.out_dir,
                   pop_size=args.pop, eval_seeds_per_gen=args.eval_seeds_per_gen,
                   algos=args.algos, tag=args.tag)
