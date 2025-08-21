"""
Bins splines and get a big indexing tensor - JAX version
"""

from typing import List, Union

import numpy as np
import jax.numpy as jnp
import jax
from tqdm import tqdm

from magpy.file_io.spline_file import SplineFile
from magpy.file_io.systematic_file import SystematicFile, SystematicHandler
from magpy.utils.modes import SplineModes, bins_to_spline_name
from magpy.objects.mc_event import MCEventMonolith, MCEventIndices
from magpy.objects.spline_handler import SplineMonolith

@jax.jit
def _reweight_jit(
    monolith: jnp.ndarray, 
    combined_event_weights: jnp.ndarray,
    valid_events: jnp.ndarray
) -> jnp.ndarray:
    """JIT-compiled reweighting function for ultra-fast performance with multi-spline events"""
    # Apply combined weights to valid events
    monolith = monolith.at[valid_events, MCEventIndices.WEIGHT.value].multiply(combined_event_weights)
    
    return monolith

class SplineSystematicModel:

    def __init__(self, spline_file: SplineFile, systematics: Union[SystematicFile, SystematicHandler]):
        self.spline_file = spline_file
        if isinstance(systematics, SystematicFile):
            self.systematic_handler = systematics.systematic_handler
        else:
            self.systematic_handler = systematics
        self._spline_monolith = self.spline_file.monolith
        self.setup_splines()
        self._spline_idx_monolith = None


    def setup_splines(self):
        """Setup the splines from the spline file.
        We want a unique associate for each bin, mode and syst to each spline
        """
        out_list = []
        self._bins_handler = self.spline_file.get_bin_handler()

        # Get inidices
        self.SYST_INDEX = 0
        self.SPLINE_INDEX = 1
        self.MODE_INDEX = 2
        self.BIN_INDEX = [i + 3 for i in range(len(self._bins_handler.bin_indices[0]))]

        for isyst, syst in tqdm(
            enumerate(self.systematic_handler.systematics),
            desc="Processing systematics",
            total=len(self.systematic_handler.systematics)
        ):
            for mode in syst.modes:
                mode_name = SplineModes(mode).spline_name()
                for bins in self._bins_handler.bin_indices:
                    spline_name = bins_to_spline_name(
                        syst.spline_name, mode_name, bins.tolist()
                    )
                    # Get spline
                    try:
                        spline_idx = self.spline_file.spline_names.index(spline_name)
                        output = [isyst, spline_idx, mode]
                        output.extend(bins.tolist())
                        out_list.append(output)
                    except ValueError:
                        # Spline not found, skip
                        continue
                        
        self._index_tensor = jnp.array(out_list, dtype=jnp.int64)
        self._spline_monolith.map_splines_to_syst(self._index_tensor[:,self.SYST_INDEX:self.SPLINE_INDEX+1])
        

    def get_monolith_splines(
        self, mc_event_monolith: MCEventMonolith, bin_indices: jnp.ndarray
    ) -> jnp.ndarray:
        """Find ALL splines for each event - supporting multiple splines per event.
        Each event can have multiple splines (one per systematic) and we multiply their weights.
        """
        use_dummy = MCEventIndices.DUMMY.value in bin_indices

        self._bins = jnp.concatenate([jnp.array([MCEventIndices.INTERACTION_MODE.value]), bin_indices], axis=0)

        monolith_kinematics = mc_event_monolith.monolith[
            :, bin_indices[bin_indices != MCEventIndices.DUMMY.value]
        ]

        if use_dummy:
            monolith_kinematics = jnp.concatenate(
                (
                    monolith_kinematics,
                    jnp.ones(
                        (len(monolith_kinematics), 1),
                        dtype=jnp.float64,
                    ),
                ),
                axis=1,
            )

        # Now we get the index of the full indices in the index tensor
        # Use JAX-compatible bin handler directly
        monolith_kinematic_bins = self._bins_handler.find_bin(monolith_kinematics)
        
        modes = (
            mc_event_monolith._mc_event_monolith[
                :, int(MCEventIndices.INTERACTION_MODE.value)
            ]
            .reshape(-1, 1)
            .astype(jnp.int32)
        )

        full_index_array = jnp.concatenate((modes, monolith_kinematic_bins), axis=1)

        # JAX OPTIMIZED: Truly vectorized matching
        index_keys = self._index_tensor[:, self.MODE_INDEX:]
        
        # Vectorized matching using broadcasting - all events at once
        # Shape: (n_events, n_splines, n_dims)
        matches = jnp.all(
            full_index_array[:, None, :] == index_keys[None, :, :], 
            axis=2
        )
        
        # NEW ARCHITECTURE: Store ALL matching splines per event, not just the first one
        # We'll create a sparse representation of event -> spline mappings
        
        # Find all (event_idx, spline_idx) pairs where there's a match
        event_indices, spline_indices = jnp.where(matches)
        
        # Store the mapping: each entry is (event_idx, spline_idx)
        self._event_spline_pairs = jnp.column_stack([event_indices, spline_indices])
        
        # For each event, we need to know which entries in _event_spline_pairs belong to it
        # Sort by event index to group splines by event
        sort_indices = jnp.argsort(event_indices)
        self._event_spline_pairs = self._event_spline_pairs[sort_indices]
        
        # Get unique events (those that have at least one spline)
        self._valid_events = jnp.unique(self._event_spline_pairs[:, 0])
        
        # For compatibility, store the spline indices
        spline_indices_only = self._event_spline_pairs[:, 1]
        
        # Saves rebuilding every loop
        self._spline_value_arr = jnp.zeros(len(self._index_tensor), dtype=jnp.float64)
        
        return spline_indices_only  # Return for compatibility

    def get_weights_only(self, syst_values: jnp.ndarray) -> jnp.ndarray:
        """Get systematic weights for all events - OPTIMIZED: multiply weights from ALL splines per event"""
        # Evaluate spline weights for all splines
        spline_weights = self._spline_monolith(syst_values)
        
        # Get weights for all event-spline pairs
        pair_weights = spline_weights[self._event_spline_pairs[:, 1]]
        
        # HIGHLY OPTIMIZED: Use scatter_mul for direct multiplication 
        n_total_events = jnp.max(self._event_spline_pairs[:, 0]) + 1
        
        # Start with all ones, then multiply in the weights for each event
        final_weights = jnp.ones(n_total_events, dtype=jnp.float64)
        
        # Use scatter multiplication - this is the most efficient approach
        event_ids = self._event_spline_pairs[:, 0]
        final_weights = final_weights.at[event_ids].multiply(pair_weights)
        
        # Return weights only for events that have splines (valid events)
        return final_weights[self._valid_events]

    def reweight(self, syst_values: jnp.ndarray, monolith: jnp.ndarray) -> jnp.ndarray:
        """MULTI-SPLINE REWEIGHTING: Multiply weights from all splines affecting each event"""
        # Get combined weights for all events (already multiplied together per event)
        combined_event_weights = self.get_weights_only(syst_values)
        
        return _reweight_jit(
            monolith, 
            combined_event_weights,
            self._valid_events
        )
