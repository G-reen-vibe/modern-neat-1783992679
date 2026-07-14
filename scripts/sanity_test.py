"""Quick sanity test: baseline NEAT on CartPole-v1 for 5 generations."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import time
from src.baseline_neat import NEAT, NEATConfig
from src.evaluator import eval_genome

cfg = NEATConfig(pop_size=30, p_add_node=0.03, p_add_conn=0.05,
                 compat_threshold=3.0)
env_name = 'CartPole-v1'
neat = NEAT(num_inputs=4, num_outputs=2, cfg=cfg, birth_gen=0)
print(f"Initial pop: {len(neat.pop)} genomes, best fitness = ", end='')
best0 = max(eval_genome(g, env_name, seed=0) for g in neat.pop)
print(f"{best0:.1f}")

t0 = time.time()
for gen in range(5):
    seeds = [0, 1]
    stats = neat.step(env_name, seeds)
    print(f"  gen {stats['gen']}: best={stats['best']:.1f} mean={stats['mean']:.1f} "
          f"species={stats['num_species']} complexity={stats['avg_complexity']:.1f}")
print(f"5 gens took {time.time()-t0:.1f}s")
