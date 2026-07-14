"""Comprehensive multi-seed benchmark across all envs.
Designed to be run in chunks (one env at a time) to fit in tool timeout.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import random
from scripts.compare import run_one


def run_env_benchmark(env_name, gens, n_seeds, pop, eps, algos, tag):
    print(f"\n{'='*70}")
    print(f"  ENV: {env_name} | gens={gens} | seeds={n_seeds} | pop={pop} | eps={eps}")
    print(f"{'='*70}")
    results = {}
    for algo in algos:
        print(f"\n--- {algo.upper()} ---")
        runs = []
        if algo == 'tide':
            if pop <= 50:
                cfg = {'grid_n_bins': 5, 'genomes_per_cell': 2}
            elif pop <= 80:
                cfg = {'grid_n_bins': 6, 'genomes_per_cell': 2}
            else:
                cfg = {'grid_n_bins': 7, 'genomes_per_cell': 2}
        else:
            cfg = {'pop_size': pop}
        for s in range(n_seeds):
            t0 = time.time()
            r = run_one(algo, env_name, gens, eps, cfg, s,
                        eval_seeds_final=15, verbose=True)
            runs.append(r)
            print(f"  seed {s}: final={r['final_best_fitness']:.1f} ± {r['final_best_std']:.1f}, "
                  f"time={r['elapsed_s']:.1f}s, cx={r['final_best_complexity']}, "
                  f"first_solve={r['first_solve_gen']}")
        means = [r['final_best_fitness'] for r in runs]
        cxs = [r['final_best_complexity'] for r in runs]
        solve_rate = np.mean([1.0 if r['first_solve_gen'] is not None else 0.0 for r in runs])
        results[algo] = {
            'runs': runs,
            'mean': float(np.mean(means)),
            'std': float(np.std(means)),
            'solve_rate': float(solve_rate),
            'avg_cx': float(np.mean(cxs)),
            'avg_time': float(np.mean([r['elapsed_s'] for r in runs])),
        }
        print(f"  SUMMARY: mean={np.mean(means):.1f} ± {np.std(means):.1f}, "
              f"solve_rate={solve_rate*100:.0f}%, "
              f"avg_cx={np.mean(cxs):.1f}, "
              f"avg_time={np.mean([r['elapsed_s'] for r in runs]):.1f}s")
    # Save
    out_path = f'results/benchmark_{tag}.json'
    compact = {}
    for algo, r in results.items():
        compact[algo] = {
            'mean': r['mean'], 'std': r['std'],
            'solve_rate': r['solve_rate'], 'avg_cx': r['avg_cx'],
            'avg_time': r['avg_time'],
            'per_seed': [{'final': run['final_best_fitness'],
                          'std': run['final_best_std'],
                          'cx': run['final_best_complexity'],
                          'first_solve': run['first_solve_gen'],
                          'time': run['elapsed_s']}
                         for run in r['runs']],
        }
    with open(out_path, 'w') as f:
        json.dump(compact, f, indent=2)
    print(f"\nSaved to {out_path}")
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--env', required=True)
    p.add_argument('--gens', type=int, default=30)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--pop', type=int, default=50)
    p.add_argument('--eps', type=int, default=2)
    p.add_argument('--algos', nargs='+', default=['neat', 'crit', 'tide'])
    p.add_argument('--tag', default=None)
    args = p.parse_args()
    tag = args.tag or args.env.replace('-', '_').lower()
    run_env_benchmark(args.env, args.gens, args.seeds, args.pop, args.eps,
                      args.algos, tag)
