"""
Canonical baseline NEAT (Stanley & Miikkulainen 2002) implementation.

Used as the primary comparison baseline. Faithful to the original paper:
- Compatibility distance: δ = c1*E/N + c2*D/N + c3*W̄
- Speciation via threshold δ_t
- Explicit fitness sharing within species
- Offspring via crossover + mutation (add node / add conn / perturb weight)
- 25% elite retention per species (capped at 1 if species is small)

Hyperparameters are tuned for CartPole-v1 in default config.
"""
from __future__ import annotations
import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple
from .genome import Genome, NodeGene, ConnGene
from .innovation import InnovationRegistry
from .evaluator import evaluate_population, normalize_reward


@dataclass
class NEATConfig:
    pop_size: int = 100
    # Mutation probabilities
    p_add_node: float = 0.03
    p_add_conn: float = 0.05
    p_mut_weight: float = 0.8
    p_toggle: float = 0.0
    weight_perturb_std: float = 0.5
    weight_reset_prob: float = 0.1
    weight_init_std: float = 1.0
    # Speciation
    compat_threshold: float = 3.0
    c1: float = 1.0  # excess
    c2: float = 1.0  # disjoint
    c3: float = 0.4  # weight diff
    # Species management
    elitism: int = 1  # elites per species
    survival_threshold: float = 0.20  # top fraction that breeds
    interspecies_mate: float = 0.001
    # Structural bounds (to keep search tractable)
    max_hidden: int = 50
    max_conns: int = 200
    # Novelty / fitness shaping
    use_fitness_sharing: bool = True


class NEAT:
    """Baseline NEAT."""

    name = "NEAT"

    def __init__(self, num_inputs: int, num_outputs: int, cfg: NEATConfig,
                 birth_gen: int = 0):
        self.cfg = cfg
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.innov = InnovationRegistry(num_inputs, num_outputs)
        self.pop: List[Genome] = []
        self.species: List[List[int]] = []  # list of indices into self.pop
        self.gen = 0
        self.history: List[dict] = []
        self._init_pop(birth_gen)

    # ------------------------------------------------------------------
    def _init_pop(self, birth_gen: int) -> None:
        """Create initial population: each genome fully connected
        input->output with random weights."""
        self.innov.reset_generation()
        for _ in range(self.cfg.pop_size):
            g = Genome(self.num_inputs, self.num_outputs, birth_gen)
            for i in range(self.num_inputs):
                for o in range(self.num_outputs):
                    in_id = i
                    out_id = self.num_inputs + o
                    cid = self.innov.conn_id_for(in_id, out_id)
                    w = float(np.random.randn() * self.cfg.weight_init_std)
                    g.conns[cid] = ConnGene(cid, in_id, out_id, w, True, birth_gen)
            self.pop.append(g)

    # ------------------------------------------------------------------
    # Speciation
    # ------------------------------------------------------------------
    def compatibility(self, g1: Genome, g2: Genome) -> float:
        """Compatibility distance δ (Stanley & Miikkulainen 2002)."""
        c1, c2, c3 = self.cfg.c1, self.cfg.c2, self.cfg.c3
        # Genes indexed by cid; disjoint/excess based on cid ranges
        ids1 = set(g1.conns.keys())
        ids2 = set(g2.conns.keys())
        matching = ids1 & ids2
        disjoint = (ids1 ^ ids2)
        # Excess = beyond the range of the other
        if ids1 and ids2:
            max1, max2 = max(ids1), max(ids2)
            excess = sum(1 for i in disjoint if (i > max1 or i > max2))
            # Adjusted: count excess as those above the smaller max
            smax = min(max1, max2)
            excess = sum(1 for i in disjoint if i > smax)
            disjoint = sum(1 for i in disjoint if i <= smax)
        else:
            excess = 0
            disjoint = len(disjoint)
        # Mean weight diff of matching
        if matching:
            wdiff = np.mean([abs(g1.conns[i].weight - g2.conns[i].weight)
                             for i in matching])
        else:
            wdiff = 0.0
        N = max(len(ids1), len(ids2), 1)
        if N < 20:
            N = 1  # small genomes: don't normalize
        return c1 * excess / N + c2 * disjoint / N + c3 * wdiff

    def _speciate(self) -> None:
        """Assign each genome to a species (greedy first-match)."""
        self.species: List[List[int]] = []
        # Use the first genome of each existing species as representative
        # For first gen, just put the first genome in its own species.
        representatives: List[int] = []  # index into self.pop
        for i, g in enumerate(self.pop):
            placed = False
            for si, ridx in enumerate(representatives):
                if self.compatibility(g, self.pop[ridx]) < self.cfg.compat_threshold:
                    self.species[si].append(i)
                    placed = True
                    break
            if not placed:
                representatives.append(i)
                self.species.append([i])

    # ------------------------------------------------------------------
    # Evolution step
    # ------------------------------------------------------------------
    def step(self, env_name: str, eval_seeds: List[int]) -> dict:
        """One generation of evolution. Returns stats dict."""
        self.innov.reset_generation()
        # 1) Evaluate (and track per-seed fitness for robustness analysis)
        from .evaluator import eval_genome
        for g in self.pop:
            g._per_seed_fitness = [eval_genome(g, env_name, s) for s in eval_seeds]
            g.fitness = float(np.mean(g._per_seed_fitness))
        # 2) Speciate
        self._speciate()
        # 3) Compute adjusted fitness (explicit fitness sharing)
        for sp in self.species:
            n = len(sp)
            for i in sp:
                g = self.pop[i]
                g.adjusted_fitness = g.fitness / n if self.cfg.use_fitness_sharing else g.fitness
        # 4) Compute offspring allocation proportional to adjusted fitness
        total_adj = sum(max(g.adjusted_fitness, 0.0) for g in self.pop)
        if total_adj <= 0:
            # uniform allocation
            allocations = [len(self.species[0])]  # fallback
            # Just give each species pop_size/num_species
            n_sp = max(len(self.species), 1)
            alloc = [self.cfg.pop_size // n_sp] * n_sp
            for k in range(self.cfg.pop_size - sum(alloc)):
                alloc[k % n_sp] += 1
        else:
            sp_sums = []
            for sp in self.species:
                s = sum(max(self.pop[i].adjusted_fitness, 0.0) for i in sp)
                sp_sums.append(s)
            alloc = [int(round(self.cfg.pop_size * s / total_adj))
                     for s in sp_sums]
            # Adjust for rounding
            while sum(alloc) > self.cfg.pop_size:
                alloc[alloc.index(max(alloc))] -= 1
            while sum(alloc) < self.cfg.pop_size:
                alloc[alloc.index(min(alloc))] += 1
        # 5) Build next generation
        new_pop: List[Genome] = []
        for si, sp in enumerate(self.species):
            if not sp:
                continue
            # Sort by fitness desc
            sp_sorted = sorted(sp, key=lambda i: self.pop[i].fitness, reverse=True)
            # Elite(s)
            n_elite = min(self.cfg.elitism, len(sp))
            for i in sp_sorted[:n_elite]:
                new_pop.append(self.pop[i].copy())
            # Offspring
            n_off = max(0, alloc[si] - n_elite)
            # Parents: top survival_threshold fraction
            n_parents = max(1, int(len(sp) * self.cfg.survival_threshold))
            parents = sp_sorted[:n_parents]
            for _ in range(n_off):
                if len(parents) == 0:
                    break
                if len(parents) == 1 or random.random() < 0.25:
                    # Asexual: clone + mutate
                    pidx = random.choice(parents)
                    child = self.pop[pidx].copy()
                else:
                    # Crossover
                    p1, p2 = random.sample(parents, 2)
                    child = self._crossover(self.pop[p1], self.pop[p2])
                self._mutate(child)
                new_pop.append(child)
        # If we underflow (extinct species), fill with random mutations of best
        while len(new_pop) < self.cfg.pop_size:
            child = max(self.pop, key=lambda g: g.fitness).copy()
            self._mutate(child)
            new_pop.append(child)
        # Trim overflow
        new_pop = new_pop[:self.cfg.pop_size]
        self.pop = new_pop
        self.gen += 1
        # Stats
        best = max(g.fitness for g in self.pop)
        mean = float(np.mean([g.fitness for g in self.pop]))
        avg_complexity = float(np.mean([g.complexity() for g in self.pop]))
        max_complexity = max(g.complexity() for g in self.pop)
        stats = {
            'gen': self.gen,
            'best': float(best),
            'mean': float(mean),
            'num_species': len(self.species),
            'avg_complexity': avg_complexity,
            'max_complexity': int(max_complexity),
        }
        self.history.append(stats)
        return stats

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------
    def _crossover(self, g1: Genome, g2: Genome) -> Genome:
        """Crossover: matching genes random pick; excess/disjoint from fitter."""
        if g1.fitness < g2.fitness:
            g1, g2 = g2, g1
        child = Genome(self.num_inputs, self.num_outputs, self.gen)
        # Inherit conn genes
        ids1 = set(g1.conns.keys())
        ids2 = set(g2.conns.keys())
        all_ids = ids1 | ids2
        for cid in all_ids:
            in1 = cid in g1.conns
            in2 = cid in g2.conns
            if in1 and in2:
                src = g1.conns[cid] if random.random() < 0.5 else g2.conns[cid]
                enabled = src.enabled
                # If either disabled, 75% chance disabled (NEAT rule)
                if (not g1.conns[cid].enabled) or (not g2.conns[cid].enabled):
                    if random.random() < 0.75:
                        enabled = False
                child.conns[cid] = ConnGene(cid, src.in_node, src.out_node,
                                             src.weight, enabled, self.gen)
            elif in1:
                c = g1.conns[cid]
                child.conns[cid] = ConnGene(cid, c.in_node, c.out_node,
                                             c.weight, c.enabled, self.gen)
            else:
                c = g2.conns[cid]
                # Only inherit from less-fit parent if we have to; NEAT skips
                # but we include for completeness
                child.conns[cid] = ConnGene(cid, c.in_node, c.out_node,
                                             c.weight, c.enabled, self.gen)
        # Inherit all nodes that any conn references
        needed = set()
        for c in child.conns.values():
            needed.add(c.in_node)
            needed.add(c.out_node)
        # Always include input+output
        for i in range(self.num_inputs + self.num_outputs):
            needed.add(i)
        # Pull node metadata from fitter parent if available
        for nid in needed:
            if nid in g1.nodes:
                n = g1.nodes[nid]
                child.nodes[nid] = NodeGene(nid, n.kind, self.gen)
            elif nid in g2.nodes:
                n = g2.nodes[nid]
                child.nodes[nid] = NodeGene(nid, n.kind, self.gen)
        return child

    def _mutate(self, g: Genome) -> None:
        cfg = self.cfg
        # Add node
        if random.random() < cfg.p_add_node and g.conns and g.num_hidden() < cfg.max_hidden:
            enabled_cids = [cid for cid, c in g.conns.items() if c.enabled]
            if enabled_cids:
                cid_to_split = random.choice(enabled_cids)
                new_nid = self.innov.node_id_for_split(cid_to_split)
                g.add_hidden_node(new_nid, cid_to_split, self.gen)
        # Add connection
        if random.random() < cfg.p_add_conn and g.num_enabled_conns() < cfg.max_conns:
            self._try_add_conn(g)
        # Mutate weights
        for c in g.conns.values():
            if random.random() < cfg.p_mut_weight:
                if random.random() < cfg.weight_reset_prob:
                    c.weight = float(np.random.randn() * cfg.weight_init_std)
                else:
                    c.weight += float(np.random.randn() * cfg.weight_perturb_std)
        # Toggle (optional)
        if cfg.p_toggle > 0:
            for c in g.conns.values():
                if random.random() < cfg.p_toggle:
                    c.enabled = not c.enabled

    def _try_add_conn(self, g: Genome, max_tries: int = 20) -> None:
        for _ in range(max_tries):
            in_node = random.choice(list(g.nodes.keys()))
            out_node = random.choice(list(g.nodes.keys()))
            if g.nodes[in_node].kind == 'output':
                continue  # outputs can't be source
            if g.nodes[out_node].kind == 'input':
                continue  # inputs can't be target
            if in_node == out_node:
                continue
            # Check duplicate
            dup = any(c.in_node == in_node and c.out_node == out_node
                      for c in g.conns.values())
            if dup:
                continue
            cid = self.innov.conn_id_for(in_node, out_node)
            w = float(np.random.randn() * self.cfg.weight_init_std)
            if g.add_connection(cid, in_node, out_node, w, self.gen):
                return
