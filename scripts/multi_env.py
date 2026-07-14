"""Multi-env benchmark: run NEAT vs CRIT on multiple envs and summarize."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import random
from scripts.compare import run_one, _SOLVE_THRESHOLDS


def main():
    configs = [
        ('CartPole-v1', 15, 2, 30, 2),
        ('MountainCar-v0', 20, 2, 30, 2),
        ('Acrobot-v1', 15, 2, 25, 1),
    ]
    all_results = {}
    for env_name, gens, n_seeds, pop, eps in configs:
        print(f"\n{'='*60}")
        print(f"ENV: {env_name} (gens={gens}, seeds={n_seeds}, pop={pop}, eps={eps})")
        print(f"{'='*60}")
        env_results = {'neat': [], 'crit': []}
        for algo in ['neat', 'crit']:
            print(f"\n--- {algo.upper()} ---")
            for s in range(n_seeds):
                t0 = time.time()
                r = run_one(algo, env_name, gens, eps, {'pop_size': pop}, s,
                            eval_seeds_final=15, verbose=True)
                env_results[algo].append(r)
                print(f"  seed {s}: final={r['final_best_fitness']:.1f} ± {r['final_best_std']:.1f}, "
                      f"time={r['elapsed_s']:.1f}s, cx={r['final_best_complexity']}, "
                      f"first_solve={r['first_solve_gen']}, "
                      f"best_ever_train={r.get('best_ever_training_score', 'n/a')}")
        # Summarize
        print(f"\n=== Summary for {env_name} ===")
        for algo in ['neat', 'crit']:
            runs = env_results[algo]
            means = [r['final_best_fitness'] for r in runs]
            cxs = [r['final_best_complexity'] for r in runs]
            solve_rate = np.mean([1.0 if r['first_solve_gen'] is not None else 0.0 for r in runs])
            print(f"  {algo.upper()}: mean={np.mean(means):.1f} ± {np.std(means):.1f}, "
                  f"solve_rate={solve_rate*100:.0f}%, "
                  f"avg_cx={np.mean(cxs):.1f}, "
                  f"avg_time={np.mean([r['elapsed_s'] for r in runs]):.1f}s")
        all_results[env_name] = env_results
    with open('results/multi_env_benchmark.json', 'w') as f:
        # Strip history for compactness
        compact = {}
        for env, env_res in all_results.items():
            compact[env] = {}
            for algo, runs in env_res.items():
                compact[env][algo] = [{
                    'final_best_fitness': r['final_best_fitness'],
                    'final_best_std': r['final_best_std'],
                    'final_best_complexity': r['final_best_complexity'],
                    'first_solve_gen': r['first_solve_gen'],
                    'elapsed_s': r['elapsed_s'],
                    'best_ever_training_score': r.get('best_ever_training_score', None),
                } for r in runs]
        json.dump(compact, f, indent=2)
    print("\nSaved to results/multi_env_benchmark.json")


if __name__ == '__main__':
    main()
