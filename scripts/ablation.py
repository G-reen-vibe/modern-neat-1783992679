"""Ablation study: turn off each CRIT component and measure effect."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import random
from src.crit_neat import CRITNEAT, CRITConfig
from src.evaluator import eval_genome, get_env_info
import gymnasium as gym


def run_one(env_name, gens, cfg, seed, eval_seeds_per_gen=2):
    np.random.seed(seed); random.seed(seed)
    env = gym.make(env_name)
    n_in = int(np.prod(env.observation_space.shape))
    discrete = hasattr(env.action_space, 'n')
    n_out = int(env.action_space.n) if discrete else int(np.prod(env.action_space.shape))
    env.close()
    algo = CRITNEAT(n_in, n_out, cfg, discrete_actions=discrete)
    history = []
    for g in range(gens):
        seeds = [seed * 1000 + g * 7 + k for k in range(eval_seeds_per_gen)]
        stats = algo.step(env_name, seeds)
        history.append(stats['best'])
    best_g = max(algo.pop, key=lambda g: g.fitness)
    final = [eval_genome(best_g, env_name, s) for s in range(10000, 10000+15)]
    return float(np.mean(final)), float(np.std(final)), int(best_g.complexity())


def main():
    env_name = 'MountainCar-v0'
    gens = 25
    seeds = [0, 1]
    pop = 30

    ablations = [
        ('full', {}),
        ('no_criticality_growth', {'use_criticality_growth': False}),
        ('no_functional_pruning', {'use_functional_pruning': False}),
        ('no_adaptive_rates', {'use_adaptive_rates': False}),
        ('no_novelty_bonus', {'use_novelty_bonus': False}),
        ('no_soft_sharing', {'use_soft_sharing': False}),
        ('no_structural_novelty_bias', {'use_structural_novelty_bias': False}),
    ]

    results = {}
    for name, overrides in ablations:
        print(f"\n=== {name} ===")
        cfg_kwargs = {'pop_size': pop}
        cfg_kwargs.update(overrides)
        run_results = []
        for s in seeds:
            t0 = time.time()
            cfg = CRITConfig(**cfg_kwargs)
            mean_f, std_f, cx = run_one(env_name, gens, cfg, s)
            print(f"  seed {s}: mean={mean_f:.1f} ± {std_f:.1f}, cx={cx}, t={time.time()-t0:.1f}s")
            run_results.append((mean_f, std_f, cx))
        means = [r[0] for r in run_results]
        cxs = [r[2] for r in run_results]
        results[name] = {
            'runs': run_results,
            'mean': float(np.mean(means)),
            'std': float(np.std(means)),
            'mean_cx': float(np.mean(cxs)),
        }
        print(f"  Summary: mean={np.mean(means):.1f} ± {np.std(means):.1f}, cx={np.mean(cxs):.1f}")

    print("\n\n=== Ablation Summary ===")
    print(f"{'name':<30} {'mean':<10} {'std':<10} {'cx':<10}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['mean']):
        print(f"{name:<30} {r['mean']:<10.1f} {r['std']:<10.1f} {r['mean_cx']:<10.1f}")

    with open('results/ablation_mountaincar.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
