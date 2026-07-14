"""
Innovation registry shared by all algorithms.
Acts as a global innovation counter (Stanley & Miikkulainen 2002 style),
but allows each algorithm to choose how to use it.
"""
from __future__ import annotations
import threading


class InnovationRegistry:
    """Global counter for structural innovations.
    Two genomes that mutate the same structure get the same innovation id.
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        self._lock = threading.Lock()
        self._next_node_id: int = num_inputs + num_outputs
        self._next_conn_id: int = 1
        # Map (in_node, out_node) -> conn_id, to dedup within a generation
        self._conn_lookup: dict[tuple[int, int], int] = {}
        self._node_lookup: dict[int, int] = {}  # split_cid -> new_nid

    def reset_generation(self) -> None:
        """Reset the per-generation dedup tables (call between generations)."""
        with self._lock:
            self._conn_lookup.clear()
            self._node_lookup.clear()

    def new_node_id(self) -> int:
        with self._lock:
            nid = self._next_node_id
            self._next_node_id += 1
            return nid

    def conn_id_for(self, in_node: int, out_node: int) -> int:
        """Get or create a connection innovation id for (in_node, out_node)."""
        with self._lock:
            key = (in_node, out_node)
            if key not in self._conn_lookup:
                self._conn_lookup[key] = self._next_conn_id
                self._next_conn_id += 1
            return self._conn_lookup[key]

    def node_id_for_split(self, cid_to_split: int) -> int:
        """Get a node id for splitting a connection. Same cid -> same node id
        (only within a generation)."""
        with self._lock:
            if cid_to_split not in self._node_lookup:
                self._node_lookup[cid_to_split] = self._next_node_id
                self._next_node_id += 1
            return self._node_lookup[cid_to_split]
