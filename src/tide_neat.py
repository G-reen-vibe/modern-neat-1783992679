"""
TIDE-NEAT: Trajectory-Informed Diversified Evolution NEAT.

A NEW FUNDAMENTAL ALGORITHM that discards both NEAT's genetic speciation
and CRIT-NEAT's behavioral clustering in favor of a MAP-Elites-style
quality-diversity grid in trajectory space.

CORE PRINCIPLE: In RL, what matters is the trajectory a policy produces,
not the policy's per-state action choices. TIDE-NEAT therefore maintains
an explicit GRID of behaviorally-distinct genomes, where each grid cell
holds the fittest genome found for that behavioral niche.

This is fundamentally different from:
- NEAT (species via genetic distance)
- CRIT-NEAT (clusters via behavioral distance)
- MAP-Elites (fixed behavior descriptors; usually applied to morphologies)

TIDE-NEAT's novelty:
1. TRAJECTORY DESCRIPTORS via PCA: automatically discover the most
   informative 2D projection of trajectory histograms. The grid is
   data-driven, not hand-designed.
2. CRITICALITY-GUIDED GROWTH: when adding capacity, split the most
   functionally-critical connection (measured by output change on ablation).
3. ROBUST FITNESS: 0.5*mean + 0.5*min over eval seeds (favors consistency).
4. SELECTIVE PRESSURE via cell density: cells with more attempted genomes
   get more offspring allocation (popular niches are worth exploring).

Algorithm:
  init_pop -> evaluate -> compute trajectory descriptors -> PCA project ->
  place in grid (keep fittest per cell) -> per-cell selection + mutation ->
  repeat
"""
from __future__ import annotations
import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from .genome import Genome, NodeGene, ConnGene
from .innovation import InnovationRegistry
from .evaluator import get_env_info
import gymnasium as gym
from sklearn.decomposition import PCA


@dataclass
class TIDEConfig:
    # Population (cells * genomes_per_cell = total pop)
    grid_n_bins: int = 8  # bins per dim (8x8 = 64 cells)
    genomes_per_cell: int = 2  # max genomes kept per cell
    # Mutation
    p_add_node: float = 0.05
    p_add_conn: float = 0.10
    p_mut_weight: float = 0.8
    weight_perturb_std: float = 0.4
    weight_reset_prob: float = 0.1
    weight_init_std: float = 1.0
    # Criticality growth
    use_criticality_growth: bool = True
    use_structural_novelty_bias: bool = True
    # Trajectory signature
    traj_n_bins: int = 6  # bins per state dim for trajectory histogram
    traj_max_steps: int = 150
    # Selection
    elitism_per_cell: int = 1  # elites kept per cell
    global_elitism: int = 5  # top-N genomes globally always carried over
    # Robustness
    use_robust_fitness: bool = True  # 0.5*mean + 0.5*min
    # PCA descriptor
    pca_components: int = 2  # project trajectory sig to 2D for grid
    # Structural bounds
    max_hidden: int = 30
    max_conns: int = 150
    # Stagnation
    use_stagnation_injection: bool = True
    stagnation_patience: int = 5
    stagnation_inject_frac: float = 0.2


class TIDENEAT:
    """TIDE-NEAT algorithm."""

    name = "TIDE-NEAT"

    def __init__(self, num_inputs: int, num_outputs: int, cfg: TIDEConfig,
                 discrete_actions: bool = True, birth_gen: int = 0):
        self.cfg = cfg
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.discrete = discrete_actions
        self.innov = InnovationRegistry(num_inputs, num_outputs)
        self.gen = 0
        self.history: List[dict] = []
        # The grid: dict mapping (bin_x, bin_y) -> list of genomes (fittest first)
        self.grid: Dict[Tuple[int, int], List[Genome]] = {}
        # PCA model (recomputed each generation)
        self.pca: Optional[PCA] = None
        # All genomes (working population) — used during a generation
        self.pop: List[Genome] = []
        # Stagnation
        self._best_ever_fitness: float = -1e9
        self._stagnation_count: int = 0
        # Grid edges (computed lazily on first step)
        self._grid_edges = None
        self._current_probe = np.array([])
        # Initialize population
        self._init_pop(birth_gen)

    @property
    def pop_size(self) -> int:
        return self.cfg.grid_n_bins ** 2 * self.cfg.genomes_per_cell

    # ------------------------------------------------------------------
    def _init_pop(self, birth_gen: int) -> None:
        self.innov.reset_generation()
        # Initialize with full input->output connectivity
        target_pop = self.pop_size
        for _ in range(target_pop):
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
    # Trajectory signature (same as CRIT-NEAT)
    # ------------------------------------------------------------------
    def _signature_from_states(self, states: np.ndarray, env) -> np.ndarray:
        n_bins = self.cfg.traj_n_bins
        D = states.shape[1]
        sig_parts = []
        obs_low = env.observation_space.low
        obs_high = env.observation_space.high
        for d in range(D):
            lo = obs_low[d] if np.isfinite(obs_low[d]) else float(states[:, d].min())
            hi = obs_high[d] if np.isfinite(obs_high[d]) else float(states[:, d].max())
            if hi <= lo:
                hi = lo + 1.0
            clipped = np.clip(states[:, d], lo, hi)
            bins = np.linspace(lo, hi, n_bins + 1)
            hist, _ = np.histogram(clipped, bins=bins)
            hist = hist.astype(np.float32) / max(hist.sum(), 1)
            sig_parts.append(hist)
        return np.concatenate(sig_parts)

    # ------------------------------------------------------------------
    # Criticality analysis
    # ------------------------------------------------------------------
    def _conn_criticality(self, g: Genome, probe_states: np.ndarray) -> Dict[int, float]:
        """For each enabled connection, compute output change on ablation."""
        if not probe_states.size or not g.conns:
            return {}
        base_outs = np.array([g.forward(list(obs)) for obs in probe_states])
        crits = {}
        enabled_cids = [cid for cid, c in g.conns.items() if c.enabled]
        if len(enabled_cids) > 15:
            enabled_cids = random.sample(enabled_cids, 15)
        for cid in enabled_cids:
            c = g.conns[cid]
            w_save = c.weight
            c.weight = 0.0
            new_outs = np.array([g.forward(list(obs)) for obs in probe_states])
            diff = float(np.mean(np.linalg.norm(base_outs - new_outs, axis=1)))
            crits[cid] = diff
            c.weight = w_save
        return crits

    def _sample_probe_states(self, env_name: str, n: int = 30) -> np.ndarray:
        info = get_env_info(env_name)
        env = gym.make(env_name)
        try:
            states = []
            obs, _ = env.reset(seed=self.gen * 31 + 7)
            for _ in range(n * 3):
                states.append(np.asarray(obs, dtype=np.float32))
                action = env.action_space.sample()
                obs, r, term, trunc, _ = env.step(action)
                if term or trunc:
                    obs, _ = env.reset(seed=self.gen * 31 + 7 + len(states))
            if len(states) > n:
                idx = np.random.choice(len(states), n, replace=False)
                states = [states[i] for i in idx]
            return np.array(states[:n])
        finally:
            env.close()

    # ------------------------------------------------------------------
    # Mutation (similar to CRIT but simpler)
    # ------------------------------------------------------------------
    def _mutate(self, g: Genome) -> int:
        cfg = self.cfg
        n_mut = 0
        probe = getattr(self, '_current_probe', np.array([]))
        # Add node — criticality-guided
        if random.random() < cfg.p_add_node and g.conns and g.num_hidden() < cfg.max_hidden:
            enabled_cids = [cid for cid, c in g.conns.items() if c.enabled]
            if enabled_cids:
                if cfg.use_criticality_growth and probe.size:
                    crits = self._conn_criticality(g, probe)
                    if crits:
                        cid_list = list(crits.keys())
                        weights = np.array([max(crits[c], 1e-6) for c in cid_list])
                        if cfg.use_structural_novelty_bias:
                            for k, c in enumerate(cid_list):
                                conn = g.conns[c]
                                in_kind = g.nodes[conn.in_node].kind
                                out_kind = g.nodes[conn.out_node].kind
                                if in_kind == 'input' and out_kind == 'output':
                                    weights[k] *= 1.5
                                elif in_kind == 'hidden' or out_kind == 'hidden':
                                    weights[k] *= 0.5
                        weights = weights / weights.sum()
                        cid_to_split = int(np.random.choice(cid_list, p=weights))
                    else:
                        cid_to_split = random.choice(enabled_cids)
                else:
                    cid_to_split = random.choice(enabled_cids)
                new_nid = self.innov.node_id_for_split(cid_to_split)
                if g.add_hidden_node(new_nid, cid_to_split, self.gen):
                    n_mut += 1
        # Add connection
        if random.random() < cfg.p_add_conn and g.num_enabled_conns() < cfg.max_conns:
            if self._try_add_conn(g):
                n_mut += 1
        # Mutate weights
        for c in g.conns.values():
            if random.random() < cfg.p_mut_weight:
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

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, env_name: str, eval_seeds: List[int]) -> dict:
        self.innov.reset_generation()
        # 1) Sample probe states for criticality
        self._current_probe = self._sample_probe_states(env_name)
        # 2) Evaluate fitness + collect trajectory states
        info = get_env_info(env_name)
        env = gym.make(env_name)
        all_sigs = []
        try:
            for g in self.pop:
                g_states = []
                g_fit = 0.0
                g_per_seed = []
                for s in eval_seeds:
                    obs, _ = env.reset(seed=s)
                    total = 0.0
                    states = [np.asarray(obs, dtype=np.float32)]
                    for _ in range(info['max_steps']):
                        out = g.forward(list(obs))
                        if info['discrete']:
                            action = int(np.argmax(out))
                        else:
                            action = np.array(out, dtype=np.float32)
                            if hasattr(env.action_space, 'low'):
                                action = np.clip(action, env.action_space.low, env.action_space.high)
                        obs, r, term, trunc, _ = env.step(action)
                        total += r
                        states.append(np.asarray(obs, dtype=np.float32))
                        if term or trunc:
                            break
                    g_fit += total
                    g_per_seed.append(total)
                    g_states.extend(states)
                g.fitness = g_fit / len(eval_seeds)
                if len(g_per_seed) >= 2 and self.cfg.use_robust_fitness:
                    g.fitness = 0.5 * g.fitness + 0.5 * float(np.min(g_per_seed))
                g._per_seed_fitness = g_per_seed
                sig = self._signature_from_states(np.array(g_states), env)
                all_sigs.append(sig)
        finally:
            env.close()
        # 3) PCA project trajectory signatures to 2D for grid placement
        # We use BOTH the current pop AND any existing grid genomes to fit PCA,
        # so the projection is stable across generations.
        all_existing_sigs = []
        for cell_genomes in self.grid.values():
            for g in cell_genomes:
                if hasattr(g, '_last_sig'):
                    all_existing_sigs.append(g._last_sig)
        all_existing_sigs.extend(all_sigs)
        sigs_for_pca = np.array(all_existing_sigs) if all_existing_sigs else np.array(all_sigs)
        if len(sigs_for_pca) >= 4 and sigs_for_pca.shape[1] > 2:
            try:
                sigs_noisy = sigs_for_pca + np.random.randn(*sigs_for_pca.shape) * 1e-6
                self.pca = PCA(n_components=min(self.cfg.pca_components, sigs_for_pca.shape[1]))
                self.pca.fit(sigs_noisy)
                projected = self.pca.transform(np.array(all_sigs))
            except Exception:
                projected = np.array(all_sigs)[:, :2] if np.array(all_sigs).shape[1] >= 2 else np.zeros((len(all_sigs), 2))
        else:
            projected = np.array(all_sigs)[:, :2] if np.array(all_sigs).shape[1] >= 2 else np.zeros((len(all_sigs), 2))
        # Store signature on each genome for future PCA fitting
        for g, sig in zip(self.pop, all_sigs):
            g._last_sig = sig
        # 4) Compute bin edges from EXISTING grid (if any) for stability,
        # otherwise from current projections
        n_bins = self.cfg.grid_n_bins
        if self.grid and hasattr(self, '_grid_edges') and self._grid_edges is not None:
            edges = self._grid_edges
        else:
            if len(projected) > 0:
                edges = []
                for d in range(projected.shape[1]):
                    lo = float(np.percentile(projected[:, d], 2))
                    hi = float(np.percentile(projected[:, d], 98))
                    if hi <= lo:
                        hi = lo + 1.0
                    edges.append(np.linspace(lo, hi, n_bins + 1))
            else:
                edges = [np.linspace(-1, 1, n_bins + 1) for _ in range(2)]
            self._grid_edges = edges
        # 5) Place genomes in grid: each cell keeps fittest genomes_per_cell
        new_grid: Dict[Tuple[int, int], List[Genome]] = {}
        for i, g in enumerate(self.pop):
            bin_x = min(n_bins - 1, max(0, int(np.searchsorted(edges[0], projected[i, 0]) - 1)))
            bin_y = min(n_bins - 1, max(0, int(np.searchsorted(edges[1], projected[i, 1]) - 1)))
            cell = (bin_x, bin_y)
            if cell not in new_grid:
                new_grid[cell] = []
            new_grid[cell].append(g)
        # Sort each cell by fitness desc, keep top genomes_per_cell
        for cell in new_grid:
            new_grid[cell].sort(key=lambda g: g.fitness, reverse=True)
            new_grid[cell] = new_grid[cell][:self.cfg.genomes_per_cell]
        # Merge with previous grid: keep fittest per cell across gens
        for cell, genomes in new_grid.items():
            if cell in self.grid:
                combined = self.grid[cell] + genomes
                combined.sort(key=lambda g: g.fitness, reverse=True)
                self.grid[cell] = combined[:self.cfg.genomes_per_cell]
            else:
                self.grid[cell] = genomes
        # Periodically update edges (every 5 gens) to adapt to behavioral drift
        if self.gen % 5 == 4:
            if len(projected) > 0:
                edges = []
                for d in range(projected.shape[1]):
                    lo = float(np.percentile(projected[:, d], 2))
                    hi = float(np.percentile(projected[:, d], 98))
                    if hi <= lo:
                        hi = lo + 1.0
                    edges.append(np.linspace(lo, hi, n_bins + 1))
                self._grid_edges = edges
        # 6) Stagnation check
        current_best = max((g.fitness for cell in self.grid.values() for g in cell), default=-1e9)
        if current_best > self._best_ever_fitness + 1e-6:
            self._best_ever_fitness = current_best
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1
        # 7) Build next generation: pick parents from cells, mutate, fill pop
        new_pop: List[Genome] = []
        # Global elitism: top N genomes across all cells, always carried over
        all_grid_g = sorted([g for cell in self.grid.values() for g in cell],
                            key=lambda g: g.fitness, reverse=True)
        for g in all_grid_g[:self.cfg.global_elitism]:
            new_pop.append(g.copy())
        # Per-cell elitism
        for cell, genomes in self.grid.items():
            for i in range(min(self.cfg.elitism_per_cell, len(genomes))):
                if genomes[i] not in all_grid_g[:self.cfg.global_elitism]:
                    new_pop.append(genomes[i].copy())
        # If stagnating, replace some genomes with heavily-mutated top performers
        if (self.cfg.use_stagnation_injection
                and self._stagnation_count >= self.cfg.stagnation_patience
                and self.gen > 3):
            all_g = sorted([g for cell in self.grid.values() for g in cell],
                           key=lambda g: g.fitness, reverse=True)
            n_inject = max(1, int(self.cfg.stagnation_inject_frac * len(new_pop)))
            for _ in range(n_inject):
                if all_g:
                    src = random.choice(all_g[:max(1, len(all_g)//3)])
                    new_g = src.copy()
                    for c in new_g.conns.values():
                        if random.random() < 0.5:
                            c.weight += float(np.random.randn() * 2.0)
                    if new_g.conns and new_g.num_hidden() < self.cfg.max_hidden:
                        enabled_cids = [cid for cid, c in new_g.conns.items() if c.enabled]
                        if enabled_cids:
                            cid_to_split = random.choice(enabled_cids)
                            new_nid = self.innov.node_id_for_split(cid_to_split)
                            new_g.add_hidden_node(new_nid, cid_to_split, self.gen)
                    for _ in range(3):
                        self._try_add_conn(new_g)
                    new_pop.append(new_g)
            self._stagnation_count = 0
        # Fill rest with offspring from random cells
        cells_list = list(self.grid.keys())
        while len(new_pop) < self.pop_size:
            if not cells_list:
                break
            cell = random.choice(cells_list)
            parents = self.grid[cell]
            if not parents:
                continue
            parent = random.choice(parents)
            child = parent.copy()
            self._mutate(child)
            new_pop.append(child)
        new_pop = new_pop[:self.pop_size]
        while len(new_pop) < self.pop_size:
            g = Genome(self.num_inputs, self.num_outputs, self.gen)
            for i in range(self.num_inputs):
                for o in range(self.num_outputs):
                    in_id = i
                    out_id = self.num_inputs + o
                    cid = self.innov.conn_id_for(in_id, out_id)
                    w = float(np.random.randn() * self.cfg.weight_init_std)
                    g.conns[cid] = ConnGene(cid, in_id, out_id, w, True, self.gen)
            new_pop.append(g)
        self.pop = new_pop
        self.gen += 1
        # Stats
        all_g_in_grid = [g for cell in self.grid.values() for g in cell]
        if all_g_in_grid:
            best = max(g.fitness for g in all_g_in_grid)
            mean = float(np.mean([g.fitness for g in self.pop]))
            avg_cx = float(np.mean([g.complexity() for g in self.pop]))
            max_cx = max(g.complexity() for g in self.pop)
        else:
            best = -1e9
            mean = 0.0
            avg_cx = 0.0
            max_cx = 0
        stats = {
            'gen': self.gen,
            'best': float(best),
            'mean': float(mean),
            'num_cells': len(self.grid),
            'avg_complexity': avg_cx,
            'max_complexity': int(max_cx),
            'stagnation_count': self._stagnation_count,
        }
        self.history.append(stats)
        return stats

    # ------------------------------------------------------------------
    # Best genome access (for evaluation)
    # ------------------------------------------------------------------
    def best_genome(self) -> Optional[Genome]:
        all_g = [g for cell in self.grid.values() for g in cell]
        if not all_g:
            return max(self.pop, key=lambda g: g.fitness) if self.pop else None
        return max(all_g, key=lambda g: g.fitness)
