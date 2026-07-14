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
    use_adaptive_rates: bool = False  # ablation showed this HURTS — too noisy with 5-sample window
    use_novelty_bonus: bool = True  # ablation showed marginal/negative impact; re-enabled
    use_adaptive_threshold: bool = True  # data-driven behavioral threshold
    use_structural_novelty_bias: bool = True  # bias growth toward un-split connections
    use_soft_sharing: bool = True  # soft behavioral fitness sharing (vs hard clustering)
    use_intercluster_crossover: bool = False  # ablation: didn't help, adds complexity
    intercluster_crossover_prob: float = 0.3  # fraction of offspring from inter-cluster mating
    use_robust_fitness: bool = True  # use 0.5*mean + 0.5*min for fitness (favors consistency)
    use_genetic_sharing: bool = False  # ablation: hurts because punishes structural growth
    genetic_sharing_weight: float = 0.3  # weight on genetic component (0-1)
    use_stagnation_injection: bool = True  # inject diversity when population stagnates
    stagnation_patience: int = 5  # generations without improvement before injection
    stagnation_inject_frac: float = 0.2  # fraction of pop to replace
    use_elite_archive: bool = True  # persistent archive of best-per-cell genomes
    archive_grid_bins: int = 5  # bins per dim for archive grid (5x5 = 25 cells)
    archive_as_parents: bool = True  # use archive genomes as additional parents
    archive_parent_frac: float = 0.15  # fraction of offspring from archive parents
    # Behavioral speciation
    n_probe_states: int = 50
    behavioral_threshold: float = 0.5  # used only if use_adaptive_threshold=False
    adaptive_threshold_percentile: float = 0.25  # threshold = 25th pct of pairwise dists
    # Trajectory signature (replaces per-state signature for RL)
    use_trajectory_sig: bool = True  # use trajectory-based behavioral signature
    traj_n_bins: int = 6  # bins per state dim for trajectory signature (smaller=faster)
    traj_max_steps: int = 120  # cap rollout length for signature
    traj_n_seeds: int = 2  # number of trajectory seeds to average for signature
    # Functional pruning
    silent_node_var: float = 0.01
    prune_min_hidden: int = 4  # only prune when genome has >= this many hidden nodes
    # Novelty
    novelty_k: int = 5
    novelty_coef: float = 0.05  # starting weight
    novelty_decay: float = 0.97  # decay per generation (slower than 0.99)
    novelty_coef_min: float = 0.005
    archive_size: int = 200
    archive_add_percentile: float = 0.5  # add if farther than median archive distance
    # Selection
    elitism: int = 3
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
        # Stagnation tracking
        self._best_ever_fitness: float = -1e9
        self._stagnation_count: int = 0
        # Persistent elite archive: dict (bin_x, bin_y) -> (genome, fitness)
        self.elite_archive: Dict[Tuple[int, int], Tuple[Genome, float]] = {}
        self._archive_edges = None  # bin edges for archive grid
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
            # Seed success window with random booleans to break cold-start symmetry
            # This is NOT a hack — it's a reasonable prior: ~20% success rate.
            self.mut_success_window.append([random.random() < 0.2 for _ in range(5)])
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

    def _behavioral_signature(self, g: Genome, env_name: str = None,
                              eval_seed: int = 0) -> np.ndarray:
        """Compute behavioral signature.
        If use_trajectory_sig: rollout a short trajectory and bin the states
            visited (this captures episode-level behavior, not just per-state
            action choices — important for RL where two policies can agree on
            every individual state but produce very different trajectories).
        Otherwise: per-state action distribution on probe states.
        """
        if self.cfg.use_trajectory_sig and env_name is not None:
            return self._trajectory_signature(g, env_name, eval_seed)
        sigs = []
        for obs in self.probe_states:
            out = g.forward(list(obs))
            if self.discrete:
                o = np.array(out)
                o = o - np.max(o)
                e = np.exp(o)
                p = e / (e.sum() + 1e-12)
                sigs.append(p)
            else:
                sigs.append(np.array(out, dtype=np.float32))
        return np.concatenate(sigs)

    def _trajectory_signature(self, g: Genome, env_name: str,
                              eval_seed: int) -> np.ndarray:
        """Roll out the genome on multiple seeds and produce an averaged
        histogram of visited states. Multiple seeds reduce noise and produce
        a more representative behavioral fingerprint.
        """
        info = get_env_info(env_name)
        env = gym.make(env_name)
        all_states = []
        try:
            for s_off in range(self.cfg.traj_n_seeds):
                seed = eval_seed + s_off * 17
                obs, _ = env.reset(seed=seed)
                states = [np.asarray(obs, dtype=np.float32)]
                for _ in range(self.cfg.traj_max_steps):
                    out = g.forward(list(obs))
                    if info['discrete']:
                        action = int(np.argmax(out))
                    else:
                        action = np.array(out, dtype=np.float32)
                        if hasattr(env.action_space, 'low'):
                            action = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, r, term, trunc, _ = env.step(action)
                    states.append(np.asarray(obs, dtype=np.float32))
                    if term or trunc:
                        break
                all_states.extend(states)
            states = np.array(all_states)  # (T, D)
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
        finally:
            env.close()

    def _signature_from_states(self, states: np.ndarray, env) -> np.ndarray:
        """Build a histogram signature from a pre-collected set of states.
        This avoids duplicate rollouts when fitness eval already collected states.
        """
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

    def _genetic_distance(self, g1: Genome, g2: Genome) -> float:
        """Simple genetic distance: 1 - Jaccard similarity of connection sets.
        This is a fast alternative to NEAT's compatibility distance — it
        captures structural similarity without needing excess/disjoint/weight
        coefficients.
        """
        ids1 = set(g1.conns.keys())
        ids2 = set(g2.conns.keys())
        if not ids1 and not ids2:
            return 0.0
        union = ids1 | ids2
        inter = ids1 & ids2
        jaccard = len(inter) / len(union) if union else 1.0
        # Also factor in weight differences on matching conns
        if inter:
            wdiff = np.mean([abs(g1.conns[i].weight - g2.conns[i].weight) for i in inter])
        else:
            wdiff = 1.0
        # Combine: structural distance + weighted weight distance
        return (1.0 - jaccard) + 0.2 * wdiff

    def _behavioral_distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """Distance between two behavioral signatures.
        Uses Jensen-Shannon divergence (symmetric, bounded) of the histogram
        representations. Works for both trajectory and per-state signatures.
        """
        # If trajectory sig: treat as a probability distribution
        # (each dim block sums to 1; we average JS divergence across blocks)
        if self.cfg.use_trajectory_sig:
            n_bins = self.cfg.traj_n_bins
            D = len(s1) // n_bins
            js_divs = []
            for d in range(D):
                p = s1[d*n_bins:(d+1)*n_bins] + 1e-12
                q = s2[d*n_bins:(d+1)*n_bins] + 1e-12
                p = p / p.sum()
                q = q / q.sum()
                m = 0.5 * (p + q)
                # JS = 0.5 * KL(p||m) + 0.5 * KL(q||m)
                kl_pm = np.sum(p * np.log(p / m))
                kl_qm = np.sum(q * np.log(q / m))
                js_divs.append(0.5 * (kl_pm + kl_qm))
            return float(np.mean(js_divs))
        # Per-state signature: use KL or L2 as before
        if self.discrete:
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
        # Compute adaptive threshold from pairwise distances if enabled
        if self.cfg.use_adaptive_threshold and len(sigs) >= 4:
            # Sample pairwise distances (cap to 100 pairs for speed)
            n = len(sigs)
            idxs = list(range(n))
            random.shuffle(idxs)
            sample = idxs[:min(n, 20)]
            dists = []
            for i in range(len(sample)):
                for j in range(i+1, len(sample)):
                    dists.append(self._behavioral_distance(sigs[sample[i]], sigs[sample[j]]))
            if dists:
                threshold = float(np.percentile(dists, self.cfg.adaptive_threshold_percentile * 100))
                threshold = max(threshold, 1e-3)
            else:
                threshold = self.cfg.behavioral_threshold
        else:
            threshold = self.cfg.behavioral_threshold
        clusters: List[List[int]] = []
        reps: List[int] = []
        for i, sig in enumerate(sigs):
            placed = False
            for ci, ridx in enumerate(reps):
                d = self._behavioral_distance(sig, sigs[ridx])
                if d < threshold:
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
        # Compute archive distance threshold (data-driven)
        if self.archive and len(self.archive) >= 5:
            # Sample distances from existing archive members
            sample_a = random.sample(self.archive, min(len(self.archive), 10))
            internal = []
            for i in range(len(sample_a)):
                for j in range(i+1, len(sample_a)):
                    internal.append(self._behavioral_distance(sample_a[i], sample_a[j]))
            thresh = float(np.percentile(internal, self.cfg.archive_add_percentile * 100)) if internal else self.cfg.behavioral_threshold
        else:
            thresh = 0.0
        for sig in sigs:
            if not self.archive:
                self.archive.append(sig.copy())
                continue
            dists = [self._behavioral_distance(sig, a) for a in self.archive]
            if min(dists) > thresh:
                self.archive.append(sig.copy())
        if len(self.archive) > self.cfg.archive_size:
            self.archive = self.archive[-self.cfg.archive_size:]

    def _update_elite_archive(self, sigs: List[np.ndarray]) -> None:
        """Update persistent elite archive (best-per-cell across all generations).
        Uses PCA projection of trajectory signatures to 2D, then bins into a grid.
        Each cell keeps the fittest genome ever observed there.
        """
        from sklearn.decomposition import PCA
        n_bins = self.cfg.archive_grid_bins
        # Need at least 4 sigs and dim > 2 for PCA
        if len(sigs) < 4 or sigs[0].shape[0] <= 2:
            return
        try:
            sigs_arr = np.array(sigs)
            sigs_noisy = sigs_arr + np.random.randn(*sigs_arr.shape) * 1e-6
            # If we have existing archive, include their sigs for stable PCA
            if self.elite_archive:
                existing_sigs = []
                for _, (g, _) in [(c, v) for c, v in self.elite_archive.items()]:
                    if hasattr(g, '_last_archive_sig'):
                        existing_sigs.append(g._last_archive_sig)
                if existing_sigs:
                    all_sigs = np.vstack([sigs_noisy, np.array(existing_sigs)])
                else:
                    all_sigs = sigs_noisy
            else:
                all_sigs = sigs_noisy
            pca = PCA(n_components=2)
            pca.fit(all_sigs)
            projected = pca.transform(sigs_arr)
        except Exception:
            return
        # Compute or reuse bin edges
        if self._archive_edges is None:
            edges = []
            for d in range(2):
                lo = float(np.percentile(projected[:, d], 2))
                hi = float(np.percentile(projected[:, d], 98))
                if hi <= lo:
                    hi = lo + 1.0
                edges.append(np.linspace(lo, hi, n_bins + 1))
            self._archive_edges = edges
        # Periodically update edges (every 5 gens)
        if self.gen % 5 == 4:
            edges = []
            for d in range(2):
                lo = float(np.percentile(projected[:, d], 2))
                hi = float(np.percentile(projected[:, d], 98))
                if hi <= lo:
                    hi = lo + 1.0
                edges.append(np.linspace(lo, hi, n_bins + 1))
            self._archive_edges = edges
        edges = self._archive_edges
        # Place genomes in cells, keep fittest
        for i, g in enumerate(self.pop):
            bin_x = min(n_bins - 1, max(0, int(np.searchsorted(edges[0], projected[i, 0]) - 1)))
            bin_y = min(n_bins - 1, max(0, int(np.searchsorted(edges[1], projected[i, 1]) - 1)))
            cell = (bin_x, bin_y)
            g_copy = g.copy()
            g_copy._last_archive_sig = sigs[i].copy()
            if cell not in self.elite_archive or g.fitness > self.elite_archive[cell][1]:
                self.elite_archive[cell] = (g_copy, g.fitness)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, env_name: str, eval_seeds: List[int]) -> dict:
        self.innov.reset_generation()
        # 1) Sample probe states (only used for non-trajectory signatures and criticality)
        self._sample_probe_states(env_name)
        # 2) Evaluate fitness AND collect trajectory states in a single rollout
        info = get_env_info(env_name)
        env = gym.make(env_name)
        all_states_per_genome: List[np.ndarray] = []
        try:
            for g in self.pop:
                g_states = []
                g_fit = 0.0
                g_per_seed = []  # per-seed fitness for robustness tracking
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
                # Robustness-aware fitness: weight min (worst-case) heavily
                # to favor genomes that solve ALL eval seeds over those that
                # ace one and fail others. This addresses the overfitting
                # problem on environments with bimodal rewards (MountainCar).
                # We use: 0.5 * mean + 0.5 * min, which favors consistency.
                if len(g_per_seed) >= 2 and self.cfg.use_robust_fitness:
                    g.fitness = 0.5 * g.fitness + 0.5 * float(np.min(g_per_seed))
                # Track robustness: store min and max per-seed fitness
                if not hasattr(g, '_per_seed_fitness'):
                    g._per_seed_fitness = []
                g._per_seed_fitness = g_per_seed
                all_states_per_genome.append(np.array(g_states))
        finally:
            env.close()
        # 2c) Stagnation detection
        current_best = max(g.fitness for g in self.pop)
        if current_best > self._best_ever_fitness + 1e-6:
            self._best_ever_fitness = current_best
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1
        stagnated = (self.cfg.use_stagnation_injection
                     and self._stagnation_count >= self.cfg.stagnation_patience
                     and self.gen > 3)
        if stagnated:
            # Inject diversity: replace bottom X% with mutated copies of top performers
            # PLUS large-weight mutations to encourage exploration
            sorted_idx = sorted(range(len(self.pop)),
                                key=lambda i: self.pop[i].fitness)
            n_inject = max(1, int(self.cfg.stagnation_inject_frac * len(self.pop)))
            bottom = sorted_idx[:n_inject]
            top = sorted_idx[-n_inject:]
            for bi, ti in zip(bottom, top):
                # Reset bottom genome to a heavily-mutated copy of a top genome
                new_g = self.pop[ti].copy()
                # Heavy mutation: large weight perturbation + structural mutations
                for c in new_g.conns.values():
                    if random.random() < 0.5:
                        c.weight += float(np.random.randn() * 2.0)  # large perturbation
                # Force structural mutation
                if new_g.conns and new_g.num_hidden() < self.cfg.max_hidden:
                    enabled_cids = [cid for cid, c in new_g.conns.items() if c.enabled]
                    if enabled_cids:
                        cid_to_split = random.choice(enabled_cids)
                        new_nid = self.innov.node_id_for_split(cid_to_split)
                        new_g.add_hidden_node(new_nid, cid_to_split, self.gen)
                # Try add connection
                for _ in range(3):
                    self._try_add_conn(new_g)
                self.pop[bi] = new_g
            self._stagnation_count = 0  # reset after injection
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
        # 3) Compute behavioral signatures (reuse states from fitness eval if trajectory sig)
        if self.cfg.use_trajectory_sig:
            env_temp = gym.make(env_name)
            try:
                sigs = [self._signature_from_states(states, env_temp)
                        for states in all_states_per_genome]
            finally:
                env_temp.close()
        else:
            sigs = [self._behavioral_signature(g) for g in self.pop]
        # 4) Compute novelty bonus
        novelties = [self._novelty(s) for s in sigs]
        info = get_env_info(env_name)
        reward_range = info['r_max'] - info['r_min']
        # Combined fitness
        # Use current effective novelty coef (decayed over generations)
        eff_novelty_coef = max(self.cfg.novelty_coef_min,
                               self.cfg.novelty_coef * (self.cfg.novelty_decay ** self.gen))
        for i, g in enumerate(self.pop):
            nf = eff_novelty_coef * reward_range * novelties[i] if self.cfg.use_novelty_bonus else 0.0
            g.adjusted_fitness = g.fitness + nf
        # 5) Update archive
        self._update_archive(sigs)
        # 5a) Update persistent elite archive (best-per-cell across all generations)
        if self.cfg.use_elite_archive:
            self._update_elite_archive(sigs)
        # 5b) Soft behavioral fitness sharing: divide each genome's adjusted
        # fitness by the sum of its behavioral similarities to others.
        # This is like NEAT's explicit fitness sharing but in BEHAVIOR space
        # and SOFT (no hard clusters).
        # Additionally, if use_genetic_sharing, also factor in genetic distance
        # so we preserve diversity along BOTH axes (multi-axis diversity).
        if self.cfg.use_soft_sharing and self.cfg.use_behavioral_spec:
            # Compute pairwise behavioral distances (cap at 60 genomes for speed)
            n = len(self.pop)
            sample_idx = list(range(n)) if n <= 60 else random.sample(range(n), 60)
            shares = [0.0] * n
            for ii, i in enumerate(sample_idx):
                for jj in range(ii+1, len(sample_idx)):
                    j = sample_idx[jj]
                    d_beh = self._behavioral_distance(sigs[i], sigs[j])
                    s_beh = 1.0 / (1.0 + d_beh * 5.0)
                    if self.cfg.use_genetic_sharing:
                        d_gen = self._genetic_distance(self.pop[i], self.pop[j])
                        s_gen = 1.0 / (1.0 + d_gen * 2.0)
                        # Combined sharing: weighted max of behavioral and genetic
                        # (use max because we want to penalize similarity on EITHER axis)
                        s = (1 - self.cfg.genetic_sharing_weight) * s_beh + \
                            self.cfg.genetic_sharing_weight * s_gen
                    else:
                        s = s_beh
                    shares[i] += s
                    shares[j] += s
                shares[i] += 1.0  # self
            for i in range(n):
                if i not in sample_idx:
                    shares[i] = 1.0  # fallback: no sharing computed
                # Divide adjusted fitness by share count (with floor to avoid div-by-zero)
                self.pop[i].adjusted_fitness = self.pop[i].adjusted_fitness / max(shares[i], 0.5)
        # 6) Speciate / cluster
        if self.cfg.use_behavioral_spec:
            clusters = self._behavioral_speciate(sigs)
        else:
            clusters = [list(range(len(self.pop)))]  # single cluster
        # 7) Functional pruning (only when genome is non-trivial)
        if self.cfg.use_functional_pruning:
            for g in self.pop:
                if g.num_hidden() < self.cfg.prune_min_hidden:
                    continue
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
        # Sort population by combined fitness (for parent selection)
        order = sorted(range(len(self.pop)),
                       key=lambda i: self.pop[i].adjusted_fitness, reverse=True)
        # Global elites (carry over unchanged) — selected by RAW fitness
        # to ensure we never lose the best-performing genome due to novelty
        # bonus or sharing adjustments.
        elite_order = sorted(range(len(self.pop)),
                             key=lambda i: self.pop[i].fitness, reverse=True)
        for i in elite_order[:self.cfg.elitism]:
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
                # With probability intercluster_crossover_prob, mate with a
                # parent from a DIFFERENT cluster. This recombines behavioral
                # innovations across niches — analogous to NEAT's interspecies
                # mating but driven by behavioral (not genetic) distance.
                if (self.cfg.use_intercluster_crossover
                        and len(clusters) > 1
                        and random.random() < self.cfg.intercluster_crossover_prob):
                    # Pick another cluster, prefer one with high mean fitness
                    other_clusters = [c for k, c in enumerate(clusters)
                                       if k != ci and c]
                    if other_clusters:
                        weights = np.array([
                            max(np.mean([self.pop[i].adjusted_fitness for i in c]), 1e-6)
                            for c in other_clusters
                        ])
                        weights = weights / weights.sum()
                        other_cl = other_clusters[int(np.random.choice(len(other_clusters), p=weights))]
                        other_sorted = sorted(other_cl, key=lambda i: self.pop[i].adjusted_fitness, reverse=True)
                        other_parents = other_sorted[:max(1, int(len(other_sorted) * self.cfg.survival_threshold))]
                        p1_idx = random.choice(parents)
                        p2_idx = random.choice(other_parents)
                        child = self._crossover(self.pop[p1_idx], self.pop[p2_idx])
                        # Still mutate the crossover child
                        r = self.mut_rates[p1_idx]
                        self._mutate(child, r)
                        new_pop.append(child)
                        new_rates.append(r)
                        new_windows.append(list(self.mut_success_window[p1_idx]))
                        new_parent_fits.append(self.pop[p1_idx].fitness)
                        continue
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
        # Archive-as-parents: spawn some offspring from archive genomes.
        # This re-introduces historically-good behavioral strategies into the
        # current population, allowing them to be recombined with current best.
        if (self.cfg.use_elite_archive and self.cfg.archive_as_parents
                and self.elite_archive and len(new_pop) < self.cfg.pop_size):
            n_archive_parents = max(1, int(self.cfg.archive_parent_frac * self.cfg.pop_size))
            archive_genomes = [g for g, _ in self.elite_archive.values()]
            for _ in range(n_archive_parents):
                if len(new_pop) >= self.cfg.pop_size or not archive_genomes:
                    break
                parent = random.choice(archive_genomes)
                child = parent.copy()
                self._mutate(child, self.cfg.init_mut_rate)
                new_pop.append(child)
                new_rates.append(self.cfg.init_mut_rate)
                new_windows.append([random.random() < 0.2 for _ in range(5)])
                new_parent_fits.append(parent.fitness if hasattr(parent, 'fitness') else 0.0)
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
            'eff_novelty_coef': float(eff_novelty_coef) if self.cfg.use_novelty_bonus else 0.0,
        }
        self.history.append(stats)
        return stats

    # ------------------------------------------------------------------
    # Crossover (used for inter-cluster mating)
    # ------------------------------------------------------------------
    def _crossover(self, g1: Genome, g2: Genome) -> Genome:
        """Crossover two genomes. Matching genes: random parent.
        Excess/disjoint: from fitter parent. Same structure as NEAT crossover
        but used here for INTER-BEHAVIORAL-CLUSTER recombination.
        """
        if g1.fitness < g2.fitness:
            g1, g2 = g2, g1
        child = Genome(self.num_inputs, self.num_outputs, self.gen)
        ids1 = set(g1.conns.keys())
        ids2 = set(g2.conns.keys())
        all_ids = ids1 | ids2
        for cid in all_ids:
            in1 = cid in g1.conns
            in2 = cid in g2.conns
            if in1 and in2:
                src = g1.conns[cid] if random.random() < 0.5 else g2.conns[cid]
                enabled = src.enabled
                if (not g1.conns[cid].enabled) or (not g2.conns[cid].enabled):
                    if random.random() < 0.75:
                        enabled = False
                child.conns[cid] = ConnGene(cid, src.in_node, src.out_node,
                                             src.weight, enabled, self.gen)
            elif in1:
                c = g1.conns[cid]
                child.conns[cid] = ConnGene(cid, c.in_node, c.out_node,
                                             c.weight, c.enabled, self.gen)
            # Skip excess from less-fit parent
        # Inherit all nodes referenced
        needed = set()
        for c in child.conns.values():
            needed.add(c.in_node)
            needed.add(c.out_node)
        for i in range(self.num_inputs + self.num_outputs):
            needed.add(i)
        for nid in needed:
            if nid in g1.nodes:
                n = g1.nodes[nid]
                child.nodes[nid] = NodeGene(nid, n.kind, self.gen)
            elif nid in g2.nodes:
                n = g2.nodes[nid]
                child.nodes[nid] = NodeGene(nid, n.kind, self.gen)
        return child

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
                        # Combine criticality with structural novelty:
                        # bias away from recently-split connections.
                        # Encode each cid by its (in_node, out_node) and check
                        # if either endpoint is a hidden node that already exists
                        # (i.e., has been split before).
                        cid_list = list(crits.keys())
                        weights = np.array([max(crits[c], 1e-6) for c in cid_list])
                        if cfg.use_structural_novelty_bias:
                            # Penalize cids whose in_node or out_node is a hidden node
                            # (these have been split before); prefer input->hidden or hidden->output
                            for k, c in enumerate(cid_list):
                                conn = g.conns[c]
                                in_kind = g.nodes[conn.in_node].kind
                                out_kind = g.nodes[conn.out_node].kind
                                # If both endpoints are input/output, this is a "fresh" split
                                if in_kind == 'input' and out_kind == 'output':
                                    weights[k] *= 1.5
                                elif in_kind == 'hidden' or out_kind == 'hidden':
                                    weights[k] *= 0.5
                        weights = weights / weights.sum()
                        cid_to_split = np.random.choice(cid_list, p=weights)
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
