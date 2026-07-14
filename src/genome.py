"""
Genome representation shared by all algorithms in this study.

A genome is a directed graph of nodes with weighted connections.
We support feed-forward activation only (recurrent connections are
not allowed to keep the search space tractable for RL control).

Node activations: tanh for hidden/output, identity for input.
This keeps the network bounded and differentiable-ish for analysis.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator


@dataclass
class NodeGene:
    nid: int
    kind: str  # 'input' | 'output' | 'hidden'
    # When the node was first created (innovation-like marker)
    birth_gen: int = 0
    # Functional stats updated during evaluation
    activation_variance: float = 0.0
    usage_count: int = 0


@dataclass
class ConnGene:
    cid: int
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True
    birth_gen: int = 0
    # Tracks whether this connection changed output recently (for pruning)
    impact: float = 0.0


class Genome:
    """A genome = set of node genes + set of connection genes."""

    def __init__(self, num_inputs: int, num_outputs: int, birth_gen: int = 0):
        self.nodes: Dict[int, NodeGene] = {}
        self.conns: Dict[int, ConnGene] = {}

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Input nodes have ids 0..num_inputs-1
        for i in range(num_inputs):
            self.nodes[i] = NodeGene(i, 'input', birth_gen)
        # Output nodes have ids num_inputs..num_inputs+num_outputs-1
        for o in range(num_outputs):
            nid = num_inputs + o
            self.nodes[nid] = NodeGene(nid, 'output', birth_gen)

        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0
        self.birth_gen: int = birth_gen

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def add_connection(self, cid: int, in_node: int, out_node: int,
                       weight: float, birth_gen: int) -> bool:
        """Add a connection if it does not create a cycle or duplicate."""
        if in_node == out_node:
            return False
        if in_node not in self.nodes or out_node not in self.nodes:
            return False
        # No cycles: out_node must not be an ancestor of in_node
        if self._creates_cycle(in_node, out_node):
            return False
        # No duplicate enabled connection
        for c in self.conns.values():
            if c.in_node == in_node and c.out_node == out_node and c.enabled:
                return False
        self.conns[cid] = ConnGene(cid, in_node, out_node, weight, True, birth_gen)
        return True

    def _creates_cycle(self, in_node: int, out_node: int) -> bool:
        """Return True if adding in_node -> out_node would create a cycle.
        A cycle exists if out_node can already reach in_node.
        """
        # BFS from out_node; if we reach in_node, it's a cycle
        stack = [out_node]
        seen = set()
        adj: Dict[int, List[int]] = {}
        for c in self.conns.values():
            if c.enabled:
                adj.setdefault(c.in_node, []).append(c.out_node)
        while stack:
            cur = stack.pop()
            if cur == in_node:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adj.get(cur, []))
        return False

    def add_hidden_node(self, new_nid: int, cid_to_split: int,
                        birth_gen: int) -> bool:
        """Split an existing connection, inserting a hidden node.
        Original weight w becomes: in->hidden weight=1.0, hidden->out weight=w.
        This preserves function (Stanley & Miikkulainen 2002).
        """
        if cid_to_split not in self.conns:
            return False
        c = self.conns[cid_to_split]
        in_node, out_node, w = c.in_node, c.out_node, c.weight
        c.enabled = False
        self.nodes[new_nid] = NodeGene(new_nid, 'hidden', birth_gen)
        # Use new cids; caller should pass new_nid-derived ids
        self.conns[10**9 + new_nid * 2] = ConnGene(
            10**9 + new_nid * 2, in_node, new_nid, 1.0, True, birth_gen)
        self.conns[10**9 + new_nid * 2 + 1] = ConnGene(
            10**9 + new_nid * 2 + 1, new_nid, out_node, w, True, birth_gen)
        return True

    # ------------------------------------------------------------------
    # Forward pass (feed-forward topological sort)
    # ------------------------------------------------------------------
    def forward(self, x: List[float]) -> List[float]:
        """Compute output activations for input x.
        Assumes feed-forward structure (guaranteed by add_connection cycle check).
        Uses Kahn's algorithm for topological sort.
        """
        # Build adjacency + in-degree over *enabled* conns
        adj: Dict[int, List[int]] = {nid: [] for nid in self.nodes}
        in_degree: Dict[int, int] = {nid: 0 for nid in self.nodes}
        for c in self.conns.values():
            if c.enabled:
                adj[c.in_node].append(c.out_node)
                in_degree[c.out_node] += 1

        # Activation table
        act: Dict[int, float] = {}
        for i in range(self.num_inputs):
            act[i] = x[i]

        # Kahn's algorithm
        from collections import deque
        queue = deque([nid for nid in self.nodes if in_degree[nid] == 0])
        order: List[int] = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for m in adj[n]:
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    queue.append(m)

        # Process in topological order
        for n in order:
            if self.nodes[n].kind == 'input':
                continue
            total = 0.0
            for c in self.conns.values():
                if c.enabled and c.out_node == n:
                    total += c.weight * act.get(c.in_node, 0.0)
            act[n] = math.tanh(total)

        return [act.get(self.num_inputs + o, 0.0) for o in range(self.num_outputs)]

    # ------------------------------------------------------------------
    # Misc utilities
    # ------------------------------------------------------------------
    def num_hidden(self) -> int:
        return sum(1 for n in self.nodes.values() if n.kind == 'hidden')

    def num_enabled_conns(self) -> int:
        return sum(1 for c in self.conns.values() if c.enabled)

    def complexity(self) -> int:
        return self.num_hidden() + self.num_enabled_conns()

    def copy(self) -> 'Genome':
        g = Genome(self.num_inputs, self.num_outputs, self.birth_gen)
        g.nodes = {nid: NodeGene(n.nid, n.kind, n.birth_gen,
                                 n.activation_variance, n.usage_count)
                   for nid, n in self.nodes.items()}
        g.conns = {cid: ConnGene(c.cid, c.in_node, c.out_node, c.weight,
                                  c.enabled, c.birth_gen, c.impact)
                   for cid, c in self.conns.items()}
        g.fitness = self.fitness
        g.adjusted_fitness = self.adjusted_fitness
        return g

    def __repr__(self) -> str:
        return (f"Genome(inputs={self.num_inputs}, outputs={self.num_outputs}, "
                f"hidden={self.num_hidden()}, conns={self.num_enabled_conns()}, "
                f"fit={self.fitness:.1f})")
