"""
JAX-based utils for converting bins -> int and back
"""

from typing import List, Tuple
import warnings

import numpy as np
import jax.numpy as jnp

class OutOfBoundsBinWarning(Warning):
    pass


class BinHandler:
    def __init__(self, bin_edges: List[List[float]]):
        self.bin_edges = [jnp.array(b, dtype=jnp.float64) for b in bin_edges]
        self._bin_edge_dims = [len(b) - 1 for b in bin_edges]
        self._bin_edge_tensor, self._bin_edge_indices = self._generate_bin_tuples()

    def _generate_bin_tuples(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Now we want to get all coordinate tuples from self.bin_edges

        i.e. [(0,0,0), (0,0,1), ...]
        """
        idx_list = jnp.array(list(np.ndindex(*self._bin_edge_dims)), dtype=jnp.int32)

        # Now put BACK in the bin edges
        bin_list = jnp.zeros((len(idx_list), len(self.bin_edges), 2), dtype=jnp.float64)
        index_list = jnp.zeros((len(idx_list), len(self.bin_edges)), dtype=jnp.int32)
        
        for i, indices in enumerate(idx_list):
            for j, idx in enumerate(indices):
                idx_int = int(idx)  # Convert to Python int to avoid dtype issues
                bin_list = bin_list.at[i, j].set(self.bin_edges[j][idx_int : idx_int + 2])
                index_list = index_list.at[i, j].set(idx_int)
        return bin_list, index_list

    def get_bin_from_int(self, index: jnp.ndarray) -> jnp.ndarray:
        """
        Get the bin edges from the index
        """
        if jnp.any(index >= len(self._bin_edge_tensor)):
            warnings.warn(
                "Index out of bounds, returning [-1, -1] for invalid indices."
            )

        return_tensor = jnp.zeros(
            (index.shape[0], len(self.bin_edges), 2),
            dtype=jnp.float64,
        )

        # Handle valid indices
        valid_mask = index < len(self._bin_edge_tensor)
        valid_indices = index[valid_mask]
        
        return_tensor = return_tensor.at[valid_mask].set(self._bin_edge_tensor[valid_indices])
        
        # Handle invalid indices
        invalid_mask = ~valid_mask
        return_tensor = return_tensor.at[invalid_mask].set(jnp.array([-1, -1], dtype=jnp.float64))

        return return_tensor

    def find_bin(self, kinematic: jnp.ndarray) -> jnp.ndarray:
        """
        Find the bin for each event in the kinematic tensor using JAX
        """
        if kinematic.shape[1] != len(self.bin_edges):
            raise ValueError(f"Kinematic tensor has {kinematic.shape[1]} columns, but {len(self.bin_edges)} bins expected")

        bin_indices = jnp.zeros(kinematic.shape, dtype=jnp.int32)

        for i, edge in enumerate(self.bin_edges):
            # JAX searchsorted for finding bin indices
            indices = jnp.searchsorted(edge[:-1], kinematic[:, i], side='right') - 1
            bin_indices = bin_indices.at[:, i].set(indices)

        return bin_indices

    @property
    def bin_indices(self):
        return self._bin_edge_indices
        
    @property
    def bin_edge_tensor(self):
        return self._bin_edge_tensor


# For backward compatibility
# BinHandler = JAXBinHandler
