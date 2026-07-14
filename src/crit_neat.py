"""
CRIT-NEAT: Criticality-Regulated Information-Theoretic NEAT (v0, Round 1).

A fundamental reimagining of NEAT. Four core departures from canonical NEAT:

1. BEHAVIORAL SPECIATION (replaces genetic distance)
   Cluster genomes by their action distributions on a shared probe state set,
   not by structural compatibility. Two genomes with very different topologies
   that behave the same are in the same niche. This decouples diversity
   protection from structure, removing the brittle compat_threshold parameter.

2. CRITICALITY-GUIDED GROWTH (replaces random structural mutation)
   When adding a hidden node, split the connection whose removal would most
   change behavior on the probe set. This ensures new capacity goes where it
   matters most, rather than being random.

3. FUNCTIONAL PRUNING (replaces fixed elitism-only retention)
   Each generation, disable hidden nodes whose activation variance on the
   probe set is below a threshold. They are functionally silent. This gives
   automatic complexity control without an explicit parsimony pressure.

4. ADAPTIVE MUTATION RATES (replaces global fixed mutation probabilities)
   Each genome carries its own mutation rate, adapted via the 1/5th success
   rule from evolution strategies. Mutations that improved fitness raise the
   rate; those that hurt it lower the rate. This is per-individual self-tuning.

Additionally, NOVELTY BONUS uses behavioral distance to an archive of past
behaviors to encourage exploration of behavioral space (not just structural
space).

Every component is toggleable via CRITConfig for ablation studies.
"""
from __future__ import annotations
import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from .genome import Genome, NodeGene, ConnGene
from .innovation import InnovationRegistry
from .evaluator import evaluate_population, get_env_info
import gymnasium as gym


@dataclass
class CRITConfig:
    # Population
    pop_size: int = 80
    # Mutation rate adaptation (1/5 rule)
    init_mut_rate: float = 0.5
    mut_rate_min: float = 0.05
    mut_rate_max: float = 1.5
    mut_rate_step: float = 1.2  # multiply/divide by this
    # Structural mutation probabilities (base; per-genome rate multiplies these)
    p_add_node: float = 0.05
    p_add_conn: float = 0.10
    p_mut_weight: float = 0.8
    weight_perturb_std: float = 0.4
    weight_reset_prob: float = 0.1
    weight_init_std: float = 1.0
    # Toggleable components (for ablation)
    use_behavioral_spec: bool = True
    use_criticality_growth: bool = True
    use_functional_pruning: bool = True
    use_adaptive_rates: bool = True
    use_novelty_bonus: bool = True
    # Behavioral speciation
    n_probe_states: int = 50
    behavioral_threshold: float = 0.5  # max avg KL/L2 for "same niche"
    # Functional pruning
    silent_node_var: float = 0.01
    # Novelty
    novelty_k: int = 5
    novelty_coef: float = 0.05
    archive_size: int = 200
    # Selection
    elitism: int = 2
    survival_threshold: float = 0.30
    # Structural bounds
    max_hidden: int = 30
    max_conns: int = 150


class CRITNEAT:
    """CRIT-NEAT algorithm. API-compatible with NEAT for fair comparison."""

    name = "CRIT-NEAT"

    def __init__(self, num_inputs: int, num_outputs: int, cfg: CRITConfig,
                 discrete_actions: bool = True, birth_gen: int = 0):
        self.cfg = cfg
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.discrete = discrete_actions
        self.innov = InnovationRegistry(num_inputs, num_outputs)
        self.pop: List[Genome] = []
        self.gen = 0
        self.history: List[dict] = []
        # Per-genome mutation rates and recent success counts
        self.mut_rates: List[float] = []
        self.mut_success_window: List[List[bool]] = []  # last K mutation outcomes
        # Behavioral archive of action signatures (list of np arrays)
        self.archive: List[np.ndarray] = []
        # Probe state set (resampled each generation)
        self.probe_states: Optional[np.ndarray] = None
        self._init_pop(birth_gen)

    # ------------------------------------------------------------------
    def _init_pop(self, birth_gen: int) -> None:
        self.innov.reset_generation()
        for _ in range(self.cfg.pop_size):
            g = Genome(self.num_inputs, self.num_outputs, birth_gen)
            # Full input->output connectivity (matches NEAT's "minimal but working")
            for i in range(self.num_inputs):
                for o in range(self.num_outputs):
                    in_id = i
                    out_id = self.num_inputs + o
                    cid = self.innov.conn_id_for(in_id, out_id)
                    w = float(np.random.randn() * self.cfg.weight_init_std)
                    g.conns[cid] = ConnGene(cid, in_id, out_id, w, True, birth_gen)
            self.pop.append(g)
            self.mut_rates.append(self.cfg.init_mut_rate)
            self.mut_success_window.append([])
        self._parent_fits = [0.0] * self.cfg.pop_size

    # ------------------------------------------------------------------
    # Behavioral signatures
    # ------------------------------------------------------------------
    def _sample_probe_states(self, env_name: str) -> None:
        """Sample probe states from random rollouts in the env."""
        info = get_env_info(env_name)
        env = gym.make(env_name)
        try:
            states = []
            obs, _ = env.reset(seed=self.gen * 31 + 7)
            for _ in range(self.cfg.n_probe_states * 3):
                states.append(np.asarray(obs, dtype=np.float32))
                if info['discrete']:
                    action = env.action_space.sample()
                else:
                    action = env.action_space.sample()
                obs, r, term, trunc, _ = env.step(action)
                if term or trunc:
                    obs, _ = env.reset(seed=self.gen * 31 + 7 + len(states))
            # Random subsample
            if len(states) > self.cfg.n_probe_states:
                idx = np.random.choice(len(states), self.cfg.n_probe_states, replace=False)
                states = [states[i] for i in idx]
            self.probe_states = np.array(states[:self.cfg.n_probe_states])
        finally:
            env.close()

    def _behavioral_signature(self, g: Genome) -> np.ndarray:
        """Compute action signature on probe states.
        For discrete env: softmax(output) for each probe state, flattened.
        For continuous env: raw output for each probe state, flattened.
        """
        sigs = []
        for obs in self.probe_states:
            out = g.forward(list(obs))
            if self.discrete:
                # Softmax with temperature
                o = np.array(out)
                o = o - np.max(o)
                e = np.exp(o)
                p = e / (e.sum() + 1e-12)
                sigs.append(p)
            else:
                sigs.append(np.array(out, dtype=np.float32))
        return np.concatenate(sigs)

    def _behavioral_distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Distance between two behavioral signatures.
        For discrete: avg KL divergence across probe states.
        For continuous: normalized L2.
        """
        if self.discrete:
            # Reshape to (n_probe, n_actions)
            n_probe = self.cfg.n_probe_states
            p = s1.reshape(n_probe, self.num_outputs) + 1e-9
            q = s2.reshape(n_probe, self.num_outputs) + 1e-9
            p = p / p.sum(axis=1, keepdims=True)
            q = q / q.sum(axis=1, keepdims=True)
            kl = np.sum(p * np.log(p / q), axis=1)
            return float(np.mean(kl))
        else:
            return float(np.linalg.norm(s1 - s2) / max(1.0, np.linalg.norm(s1) + np.linalg.norm(s2)))

    # ------------------------------------------------------------------
    # Criticality analysis (for guided growth)
    # ------------------------------------------------------------------
    def _conn_criticality(self, g: Genome) -> Dict[int, float]:
        """For each enabled connection, compute the change in output (L2)
        on the probe set when the connection is ablated.
        Returns dict: cid -> criticality score.
        """
        if not self.cfg.use_criticality_growth or not self.probe_states.size:
            return {}
        # Baseline outputs
        base_outs = []
        for obs in self.probe_states:
            base_outs.append(np.array(g.forward(list(obs))))
        base_outs = np.array(base_outs)  # (n_probe, n_out)
        crits = {}
        # Sample up to 20 connections to test (for efficiency)
        enabled_cids = [cid for cid, c in g.conns.items() if c.enabled]
        if len(enabled_cids) > 20:
            enabled_cids = random.sample(enabled_cids, 20)
        for cid in enabled_cids:
            c = g.conns[cid]
            w_save = c.weight
            c.weight = 0.0
            new_outs = []
            for obs in self.probe_states:
                new_outs.append(np.array(g.forward(list(obs))))
            new_outs = np.array(new_outs)
            diff = float(np.mean(np.linalg.norm(base_outs - new_outs, axis=1)))
            crits[cid] = diff
            c.weight = w_save
        return crits

    def _node_activation_variance(self, g: Genome) -> Dict[int, float]:
        """Compute activation variance of each hidden node across probe states."""
        if not g.num_hidden() or not self.probe_states.size:
            return {}
        # Run forward, collect hidden node activations
        from collections import deque
        adj: Dict[int, List[int]] = {nid: [] for nid in g.nodes}
        in_degree: Dict[int, int] = {nid: 0 for nid in g.nodes}
        for c in g.conns.values():
            if c.enabled:
                adj[c.in_node].append(c.out_node)
                in_degree[c.out_node] += 1
        # Topological order
        queue = deque([nid for nid in g.nodes if in_degree[nid] == 0])
        order = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for m in adj[n]:
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    queue.append(m)
        hidden_acts = {nid: [] for nid in g.nodes if g.nodes[nid].kind == 'hidden'}
        for obs in self.probe_states:
            act: Dict[int, float] = {}
            for i in range(self.num_inputs):
                act[i] = float(obs[i])
            for n in order:
                if g.nodes[n].kind == 'input':
                    continue
                total = 0.0
                for c in g.conns.values():
                    if c.enabled and c.out_node == n:
                        total += c.weight * act.get(c.in_node, 0.0)
                act[n] = math.tanh(total)
                if n in hidden_acts:
                    hidden_acts[n].append(act[n])
        return {nid: float(np.var(vs)) for nid, vs in hidden_acts.items() if vs}

    # ------------------------------------------------------------------
    # Selection / speciation
    # ------------------------------------------------------------------
    def _behavioral_speciate(self, sigs: List[np.ndarray]) -> List[List[int]]:
        """Cluster genomes by behavioral similarity.
        Greedy: each genome joins the first cluster whose representative
        is within behavioral_threshold.
        """
        clusters: List[List[int]] = []
        reps: List[int] = []
        for i, sig in enumerate(sigs):
            placed = False
            for ci, ridx in enumerate(reps):
                d = self._behavioral_distance(sig, sigs[ridx])
                if d < self.cfg.behavioral_threshold:
                    clusters[ci].append(i)
                    placed = True
                    break
            if not placed:
                reps.append(i)
                clusters.append([i])
        return clusters

    # ------------------------------------------------------------------
    # Novelty
    # ------------------------------------------------------------------
    def _novelty(self, sig: np.ndarray) -> float:
        if not self.archive or not self.cfg.use_novelty_bonus:
            return 0.0
        dists = [self._behavioral_distance(sig, a) for a in self.archive]
        dists.sort()
        k = min(self.cfg.novelty_k, len(dists))
        return float(np.mean(dists[:k])) if k > 0 else 0.0

    def _update_archive(self, sigs: List[np.ndarray]) -> None:
        if not self.cfg.use_novelty_bonus:
            return
        for sig in sigs:
            # Add if sufficiently different from archive
            if not self.archive:
                self.archive.append(sig.copy())
                continue
            dists = [self._behavioral_distance(sig, a) for a in self.archive]
            if min(dists) > self.cfg.behavioral_threshold * 0.5:
                self.archive.append(sig.copy())
        # Cap archive size: drop oldest
        if len(self.archive) > self.cfg.archive_size:
            self.archive = self.archive[-self.cfg.archive_size:]

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, env_name: str, eval_seeds: List[int]) -> dict:
        self.innov.reset_generation()
        # 1) Sample probe states
        self._sample_probe_states(env_name)
        # 2) Evaluate fitness
        fits = evaluate_population(self.pop, env_name, eval_seeds)
        for g, f in zip(self.pop, fits):
            g.fitness = f
        # 2b) Update mutation success windows: a mutation was "successful"
        # if the genome's fitness exceeded its parent's fitness.
        if self.cfg.use_adaptive_rates and self.gen > 0:
            for i, g in enumerate(self.pop):
                if i < len(self._parent_fits):
                    success = g.fitness > self._parent_fits[i] + 1e-6
                    self.mut_success_window[i].append(success)
                    # Keep only last 5
                    if len(self.mut_success_window[i]) > 5:
                        self.mut_success_window[i] = self.mut_success_window[i][-5:]
        # 3) Compute behavioral signatures
        sigs = [self._behavioral_signature(g) for g in self.pop]
        # 4) Compute novelty bonus
        novelties = [self._novelty(s) for s in sigs]
        info = get_env_info(env_name)
        reward_range = info['r_max'] - info['r_min']
        # Combined fitness
        for i, g in enumerate(self.pop):
            nf = self.cfg.novelty_coef * reward_range * novelties[i] if self.cfg.use_novelty_bonus else 0.0
            g.adjusted_fitness = g.fitness + nf
        # 5) Update archive
        self._update_archive(sigs)
        # 6) Speciate / cluster
        if self.cfg.use_behavioral_spec:
            clusters = self._behavioral_speciate(sigs)
        else:
            clusters = [list(range(len(self.pop)))]  # single cluster
        # 7) Functional pruning
        if self.cfg.use_functional_pruning:
            for g in self.pop:
                var = self._node_activation_variance(g)
                for nid, v in var.items():
                    if v < self.cfg.silent_node_var:
                        for c in g.conns.values():
                            if c.in_node == nid and c.enabled:
                                c.enabled = False
        # 8) Build next generation
        new_pop: List[Genome] = []
        new_rates: List[float] = []
        new_windows: List[List[bool]] = []
        new_parent_fits: List[float] = []
        # Sort population by combined fitness
        order = sorted(range(len(self.pop)),
                       key=lambda i: self.pop[i].adjusted_fitness, reverse=True)
        # Global elites (carry over unchanged)
        for i in order[:self.cfg.elitism]:
            new_pop.append(self.pop[i].copy())
            # Elites: adapt their mut_rate based on their own success window
            r = self.mut_rates[i]
            w = self.mut_success_window[i]
            if self.cfg.use_adaptive_rates and len(w) >= 5:
                sr = sum(w[-5:]) / 5.0
                if sr > 0.2:
                    r = min(self.cfg.mut_rate_max, r * self.cfg.mut_rate_step)
                else:
                    r = max(self.cfg.mut_rate_min, r / self.cfg.mut_rate_step)
            new_rates.append(r)
            new_windows.append(list(w))
            new_parent_fits.append(self.pop[i].fitness)
        # Per-cluster offspring allocation (proportional to cluster mean adjusted fitness)
        cluster_scores = []
        for cl in clusters:
            if not cl:
                cluster_scores.append(0.0)
                continue
            m = float(np.mean([self.pop[i].adjusted_fitness for i in cl]))
            cluster_scores.append(max(m, 0.0))
        total = sum(cluster_scores) or 1.0
        n_remaining = self.cfg.pop_size - len(new_pop)
        allocs = [int(round(n_remaining * s / total)) for s in cluster_scores]
        while sum(allocs) > n_remaining:
            allocs[allocs.index(max(allocs))] -= 1
        while sum(allocs) < n_remaining:
            allocs[allocs.index(min(allocs))] += 1
        for ci, cl in enumerate(clusters):
            if not cl:
                continue
            cl_sorted = sorted(cl, key=lambda i: self.pop[i].adjusted_fitness, reverse=True)
            n_parents = max(1, int(len(cl_sorted) * self.cfg.survival_threshold))
            parents = cl_sorted[:n_parents]
            for _ in range(allocs[ci]):
                if not parents:
                    break
                pidx = random.choice(parents)
                parent = self.pop[pidx]
                child = parent.copy()
                # Adapt mutation rate from parent's success window
                r = self.mut_rates[pidx]
                w = list(self.mut_success_window[pidx])
                if self.cfg.use_adaptive_rates and len(w) >= 5:
                    sr = sum(w[-5:]) / 5.0
                    if sr > 0.2:
                        r = min(self.cfg.mut_rate_max, r * self.cfg.mut_rate_step)
                    else:
                        r = max(self.cfg.mut_rate_min, r / self.cfg.mut_rate_step)
                self._mutate(child, r)
                new_pop.append(child)
                new_rates.append(r)
                new_windows.append(w)
                new_parent_fits.append(parent.fitness)  # child's fitness will be compared to this
        # Fill if underflow
        while len(new_pop) < self.cfg.pop_size:
            i = random.choice(order[:max(1, len(order)//2)])
            child = self.pop[i].copy()
            self._mutate(child, self.mut_rates[i])
            new_pop.append(child)
            new_rates.append(self.mut_rates[i])
            new_windows.append(list(self.mut_success_window[i]))
            new_parent_fits.append(self.pop[i].fitness)
        new_pop = new_pop[:self.cfg.pop_size]
        new_rates = new_rates[:self.cfg.pop_size]
        new_windows = new_windows[:self.cfg.pop_size]
        new_parent_fits = new_parent_fits[:self.cfg.pop_size]
        # Replace
        self.pop = new_pop
        self.mut_rates = new_rates
        self.mut_success_window = new_windows
        self._parent_fits = new_parent_fits
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
            'num_species': len(clusters),
            'avg_complexity': avg_complexity,
            'max_complexity': int(max_complexity),
            'archive_size': len(self.archive),
            'avg_mut_rate': float(np.mean(self.mut_rates)) if self.mut_rates else 0.0,
        }
        self.history.append(stats)
        return stats

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def _mutate(self, g: Genome, mut_rate: float) -> int:
        """Apply mutations scaled by per-genome mut_rate. Returns n mutations."""
        cfg = self.cfg
        n_mut = 0
        # Add node — criticality-guided if enabled
        if random.random() < cfg.p_add_node * mut_rate and g.conns and g.num_hidden() < cfg.max_hidden:
            enabled_cids = [cid for cid, c in g.conns.items() if c.enabled]
            if enabled_cids:
                if cfg.use_criticality_growth:
                    crits = self._conn_criticality(g)
                    if crits:
                        # Pick top-k most critical, weighted random
                        sorted_cids = sorted(crits.items(), key=lambda x: x[1], reverse=True)
                        top = sorted_cids[:max(1, len(sorted_cids)//3)]
                        cids = [c for c, _ in top]
                        weights = np.array([max(crits[c], 1e-6) for c in cids])
                        weights = weights / weights.sum()
                        cid_to_split = np.random.choice(cids, p=weights)
                    else:
                        cid_to_split = random.choice(enabled_cids)
                else:
                    cid_to_split = random.choice(enabled_cids)
                new_nid = self.innov.node_id_for_split(cid_to_split)
                if g.add_hidden_node(new_nid, cid_to_split, self.gen):
                    n_mut += 1
        # Add connection
        if random.random() < cfg.p_add_conn * mut_rate and g.num_enabled_conns() < cfg.max_conns:
            if self._try_add_conn(g):
                n_mut += 1
        # Mutate weights
        for c in g.conns.values():
            if random.random() < cfg.p_mut_weight * mut_rate:
                if random.random() < cfg.weight_reset_prob:
                    c.weight = float(np.random.randn() * cfg.weight_init_std)
                else:
                    c.weight += float(np.random.randn() * cfg.weight_perturb_std)
                n_mut += 1
        return n_mut

    def _try_add_conn(self, g: Genome, max_tries: int = 20) -> bool:
        for _ in range(max_tries):
            in_node = random.choice(list(g.nodes.keys()))
            out_node = random.choice(list(g.nodes.keys()))
            if g.nodes[in_node].kind == 'output':
                continue
            if g.nodes[out_node].kind == 'input':
                continue
            if in_node == out_node:
                continue
            dup = any(c.in_node == in_node and c.out_node == out_node
                      for c in g.conns.values())
            if dup:
                continue
            cid = self.innov.conn_id_for(in_node, out_node)
            w = float(np.random.randn() * self.cfg.weight_init_std)
            if g.add_connection(cid, in_node, out_node, w, self.gen):
                return True
        return False
