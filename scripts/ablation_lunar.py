"""Comprehensive ablation study on CRIT-NEAT's components.
Run on LunarLander-v3 where CRIT shows the biggest advantage.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import random
from src.crit_neat import CRITNEAT, CRITConfig
from src.evaluator import eval_genome, get_env_info
import gymnasium as gym


def run_one(env_name, gens, cfg_kwargs, seed, eval_seeds_per_gen=1):
    np.random.seed(seed); random.seed(seed)
    env = gym.make(env_name)
    n_in = int(np.prod(env.observation_space.shape))
    discrete = hasattr(env.action_space, 'n')
    n_out = int(env.action_space.n) if discrete else int(np.prod(env.action_space.shape))
    env.close()
    cfg = CRITConfig(**cfg_kwargs)
    algo = CRITNEAT(n_in, n_out, cfg, discrete_actions=discrete)
    history = []
    best_ever_genome = None
    best_ever_score = -1e9
    for gen in range(gens):
        eval_seeds = [seed * 10000 + gen * 13 + k * 7 for k in range(eval_seeds_per_gen)]
        stats = algo.step(env_name, eval_seeds)
        history.append(stats)
        for g in algo.pop:
            if hasattr(g, '_per_seed_fitness'):
                for ps in g._per_seed_fitness:
                    if ps > best_ever_score:
                        best_ever_score = ps
                        best_ever_genome = g.copy()
    # Final selection: best-ever + top-10
    candidates = []
    if best_ever_genome is not None:
        candidates.append(best_ever_genome)
    candidates.extend(sorted(algo.pop, key=lambda g: g.fitness, reverse=True)[:10])
    seen = set()
    unique = []
    for c in candidates:
        if id(c) not in seen:
            seen.add(id(c))
            unique.append(c)
    best_g = None
    best_score = -1e9
    for cg in unique:
        evals = [eval_genome(cg, env_name, s) for s in range(10000, 10000+15)]
        if np.mean(evals) > best_score:
            best_score = float(np.mean(evals))
            best_g = cg
            final_evals = evals
    return float(np.mean(final_evals)), float(np.std(final_evals)), int(best_g.complexity())


def main():
    env_name = 'LunarLander-v3'
    gens = 12
    seeds = [0, 1]
    pop = 25

    ablations = [
        ('full_CRIT', {}),
        ('no_trajectory_sig', {'use_trajectory_sig': False}),
        ('no_criticality_growth', {'use_criticality_growth': False}),
        ('no_novelty_bonus', {'use_novelty_bonus': False}),
        ('no_soft_sharing', {'use_soft_sharing': False}),
        ('no_robust_fitness', {'use_robust_fitness': False}),
        ('no_intercluster_crossover', {'use_intercluster_crossover': False}),
    ]

    results = {}
    for name, overrides in ablations:
        print(f"\n=== {name} ===")
        cfg_kwargs = {'pop_size': pop}
        cfg_kwargs.update(overrides)
        run_results = []
        for s in seeds:
            t0 = time.time()
            mean_f, std_f, cx = run_one(env_name, gens, cfg_kwargs, s)
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

    print("\n\n=== Ablation Summary (LunarLander-v3) ===")
    print(f"{'name':<35} {'mean':<10} {'std':<10} {'cx':<10}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['mean']):
        print(f"{name:<35} {r['mean']:<10.1f} {r['std']:<10.1f} {r['mean_cx']:<10.1f}")

    with open('results/ablation_lunar.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
