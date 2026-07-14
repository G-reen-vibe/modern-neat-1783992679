"""Sanity test: CRIT-NEAT on CartPole-v1 for 5 generations."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import random
np.random.seed(0); random.seed(0)
from src.crit_neat import CRITNEAT, CRITConfig
from src.evaluator import eval_genome

cfg = CRITConfig(pop_size=30)
env_name = 'CartPole-v1'
algo = CRITNEAT(num_inputs=4, num_outputs=2, cfg=cfg, discrete_actions=True)
print(f"Initial pop: {len(algo.pop)} genomes, best fitness = ", end='')
best0 = max(eval_genome(g, env_name, seed=0) for g in algo.pop)
print(f"{best0:.1f}")

t0 = time.time()
for gen in range(5):
    seeds = [0, 1]
    stats = algo.step(env_name, seeds)
    print(f"  gen {stats['gen']}: best={stats['best']:.1f} mean={stats['mean']:.1f} "
          f"species={stats['num_species']} complexity={stats['avg_complexity']:.1f} "
          f"archive={stats['archive_size']} mut_rate={stats['avg_mut_rate']:.2f}")
print(f"5 gens took {time.time()-t0:.1f}s")
