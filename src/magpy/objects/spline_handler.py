from typing import List, Optional
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from scipy.interpolate import CubicSpline


class MagpySplineException(Exception):
    pass


@dataclass(frozen=True)
class SplineData:
    is_flat: bool
    spline_stack: jnp.ndarray  # shape: (n_segments, 5) if not flat else (5,)

def build_spline(x: jnp.ndarray, y: jnp.ndarray) -> SplineData:
    is_flat = len(jnp.unique(y)) < 2
    if is_flat:
        return SplineData(True, jnp.array([0, 0, 0, 0, 0], dtype=jnp.float64))
    spline = CubicSpline(x, y)
    knots = jnp.array(spline.x)
    coefs = jnp.array(spline.c)
    spline_stack = jnp.vstack((knots[:-1], coefs)).T
    return SplineData(False, spline_stack)


@dataclass(frozen=True)
class MonolithState:
    n_splines: int
    flat_mask: jnp.ndarray  # shape: (n_splines,)
    indices: jnp.ndarray  # shape: (n_nonflat_splines,)
    knot_ranges: jnp.ndarray  # shape: (n_nonflat_splines, 2)
    spline_monolith: jnp.ndarray  # shape: (sum of segments, 5)
    flat_spline_mask: jnp.ndarray
    spline_syst_map: jnp.ndarray  # shape: (n_flat_splines, 2)
    par_spline_starts: jnp.ndarray  # shape: (n_syst,)
    par_spline_counts: jnp.ndarray  # shape: (n_syst,)
    flat_par_splines: jnp.ndarray  # shape: (total_spline_count,)
    flat_par_knots: jnp.ndarray  # shape: (n_syst, n_knots)
    knot_indices: jnp.ndarray  # shape: (n_splines,)
    par_arr: jnp.ndarray  # shape: (n_splines,)
    weights: jnp.ndarray  # shape: (n_splines,)


# Register MonolithState as a JAX PyTree
def _monolith_state_tree_flatten(state):
    arrays = (state.flat_mask, state.indices, state.knot_ranges, state.spline_monolith,
              state.flat_spline_mask, state.spline_syst_map, state.par_spline_starts,
              state.par_spline_counts, state.flat_par_splines, state.flat_par_knots,
              state.knot_indices, state.par_arr, state.weights)
    aux_data = (state.n_splines,)
    return arrays, aux_data


def _monolith_state_tree_unflatten(aux_data, arrays):
    n_splines, = aux_data
    (flat_mask, indices, knot_ranges, spline_monolith, flat_spline_mask,
     spline_syst_map, par_spline_starts, par_spline_counts, flat_par_splines,
     flat_par_knots, knot_indices, par_arr, weights) = arrays
    return MonolithState(
        n_splines=n_splines,
        flat_mask=flat_mask,
        indices=indices,
        knot_ranges=knot_ranges,
        spline_monolith=spline_monolith,
        flat_spline_mask=flat_spline_mask,
        spline_syst_map=spline_syst_map,
        par_spline_starts=par_spline_starts,
        par_spline_counts=par_spline_counts,
        flat_par_splines=flat_par_splines,
        flat_par_knots=flat_par_knots,
        knot_indices=knot_indices,
        par_arr=par_arr,
        weights=weights
    )


jax.tree_util.register_pytree_node(
    MonolithState,
    _monolith_state_tree_flatten,
    _monolith_state_tree_unflatten
)

def init_monolith(splines: List[SplineData], spline_syst_map: jnp.ndarray) -> MonolithState:
    n_splines = len(splines)
    flat_mask = jnp.array([s.is_flat for s in splines])
    non_flat_splines = [s for s in splines if not s.is_flat]

    if len(non_flat_splines) == 0:
        raise MagpySplineException("No non-flat splines")

    lens = jnp.array([len(s.spline_stack) for s in non_flat_splines])
    indices = jnp.cumsum(lens)

    knot_ranges = jnp.zeros((indices.shape[0], 2), dtype=jnp.int64)
    knot_ranges = knot_ranges.at[0].set(jnp.array([0, indices[0]]))
    for i in range(1, indices.shape[0]):
        low, high = indices[i - 1], indices[i]
        knot_ranges = knot_ranges.at[i].set(jnp.array([low, high]))

    spline_monolith = jnp.vstack([s.spline_stack for s in non_flat_splines])
    flat_spline_mask = jnp.where(~flat_mask, size=spline_syst_map.shape[0])[0]

    non_flat_spline_syst_map = spline_syst_map[flat_spline_mask]

    n_syst = len(jnp.unique(non_flat_spline_syst_map[:, 0]))

    # Create a padded matrix for efficient vectorized processing
    max_splines_per_syst = max(int(jnp.sum(non_flat_spline_syst_map[:, 0] == i)) for i in range(n_syst))
    
    # Create padded arrays for vectorized processing
    syst_spline_indices = jnp.full((n_syst, max_splines_per_syst), -1, dtype=jnp.int64)
    syst_spline_counts = jnp.zeros(n_syst, dtype=jnp.int64)
    
    knot_sequences = [spline_monolith[low:high, 0] for (low, high) in knot_ranges]
    flat_par_knots = []

    for i in range(n_syst):
        mask = non_flat_spline_syst_map[:, 0] == i
        splines_for_par = non_flat_spline_syst_map[:, 1][mask]
        count = len(splines_for_par)
        syst_spline_counts = syst_spline_counts.at[i].set(count)
        syst_spline_indices = syst_spline_indices.at[i, :count].set(splines_for_par)
        flat_par_knots.append(knot_sequences[splines_for_par[0]])

    # Pad knot sequences to the same length for stacking
    if flat_par_knots:
        max_knots = max(len(knots) for knots in flat_par_knots)
        padded_knots = []
        for knots in flat_par_knots:
            # Pad with the last knot value to avoid issues with searchsorted
            if len(knots) < max_knots:
                padding = jnp.full(max_knots - len(knots), knots[-1])
                padded_knots.append(jnp.concatenate([knots, padding]))
            else:
                padded_knots.append(knots)
        flat_par_knots_array = jnp.stack(padded_knots)
    else:
        flat_par_knots_array = jnp.array([])

    return MonolithState(
        n_splines=n_splines,
        flat_mask=flat_mask,
        indices=indices,
        knot_ranges=knot_ranges,
        spline_monolith=spline_monolith,
        flat_spline_mask=flat_spline_mask,
        spline_syst_map=spline_syst_map,
        par_spline_starts=syst_spline_indices,  # Reuse this field for the padded matrix
        par_spline_counts=syst_spline_counts,
        flat_par_splines=jnp.array([]),  # No longer needed
        
        flat_par_knots=flat_par_knots_array,
        knot_indices=jnp.zeros(n_splines, dtype=jnp.int64),
        par_arr=jnp.zeros(n_splines, dtype=jnp.float64),
        weights=jnp.ones(n_splines, dtype=jnp.float64),
    )

@jax.jit
def get_knots_grouped_fn(state: MonolithState, x: jnp.ndarray) -> MonolithState:
    """
    OPTIMIZED: Reduced scatter operations while maintaining JAX compatibility.
    Uses vectorized operations where possible, static loops where necessary.
    """
    n_syst = len(x)
    
    # Start with current state arrays
    new_knot_indices = state.knot_indices
    new_par_arr = state.par_arr
    
    # Process each systematic with vectorized updates
    for i in range(n_syst):
        count = state.par_spline_counts[i]
        spline_indices = state.par_spline_starts[i]  # Row for this systematic
        
        # Get parameter value and knot index for this systematic
        par_val = x[i]
        knot = state.flat_par_knots[i]
        knot_index = jnp.maximum(jnp.searchsorted(knot, par_val) - 1, 0)
        
        # Vectorized operations for this systematic
        max_size = state.par_spline_starts.shape[1]
        valid_mask = jnp.arange(max_size) < count
        
        # Get offsets and compute final indices for all potential splines
        offsets = state.knot_ranges[spline_indices, 0]
        final_knot_indices = knot_index + offsets
        
        # Use where to conditionally update only valid entries
        current_knot_vals = new_knot_indices[spline_indices]
        current_par_vals = new_par_arr[spline_indices]
        
        update_knot_vals = jnp.where(valid_mask, final_knot_indices, current_knot_vals)
        update_par_vals = jnp.where(valid_mask, par_val, current_par_vals)
        
        # Single scatter operation per systematic (instead of per spline)
        new_knot_indices = new_knot_indices.at[spline_indices].set(update_knot_vals)
        new_par_arr = new_par_arr.at[spline_indices].set(update_par_vals)
    
    # Compute final weights (vectorized)
    masked_splines = state.flat_spline_mask
    idx = new_knot_indices[masked_splines]
    coefs = state.spline_monolith[idx]
    dx = new_par_arr[masked_splines] - coefs[:, 0]
    weights = (
        (coefs[:, 1] * dx + coefs[:, 2]) * dx + coefs[:, 3]
    ) * dx + coefs[:, 4]

    # Final weight update (single scatter for weights)
    new_weights = state.weights.at[masked_splines].set(weights)
    
    return replace(state, knot_indices=new_knot_indices, par_arr=new_par_arr, weights=new_weights)


# ---------------------
# Class Wrapper
# ---------------------
class Spline():
    def __init__(self, x: jnp.ndarray, y: jnp.ndarray):
        self._data = build_spline(x, y)
        
    @property
    def spline(self):
        return self._data.spline_stack

    @property
    def data(self):
        return self._data

    @property
    def is_flat(self):
        return self._data.is_flat
    
    def __len__(self):
        return len(self._data.spline_stack)

class SplineMonolith:
    def __init__(self, splines: List[Spline]):
        self._splines = [s.data for s in splines]
        self._state: Optional[MonolithState] = None  # Will be initialized in map_splines_to_syst

    def add_spline(self, new_spline: Spline):
        self._splines.append(new_spline.data)

    def map_splines_to_syst(self, spline_syst_map: jnp.ndarray):
        # Need to make normalisation splines
        self._state = init_monolith(self._splines, spline_syst_map)
    
    def __len__(self):
        return self._state.n_splines if self._state is not None else len(self._splines)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self._state is not None and self._state.spline_syst_map is not None:
            self._state = get_knots_grouped_fn(self._state, x)
            assert self._state is not None  # Type checker hint
            return self._state.weights
        else:
            raise NotImplementedError("Ungrouped mode not implemented in JAX version.")

    def get_knots_grouped(self, x: jnp.ndarray) -> jnp.ndarray:
        if self._state is None:
            raise ValueError("SplineMonolith state not initialized. Call map_splines_to_syst first.")
        self._state = get_knots_grouped_fn(self._state, x)
        assert self._state is not None  # Type checker hint
        return self._state.weights

    def get_knot_indices(self, i, par):
        if self._state is None:
            raise ValueError("SplineMonolith state not initialized. Call map_splines_to_syst first.")
        # For single parameter processing, create an array and use the vectorized function
        x = jnp.array([par])
        self._state = get_knots_grouped_fn(self._state, x)
