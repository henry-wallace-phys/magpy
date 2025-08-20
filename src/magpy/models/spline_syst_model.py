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
    spline_weights: jnp.ndarray,
    event_to_spline_map: jnp.ndarray, 
    valid_event_indices: jnp.ndarray
) -> jnp.ndarray:
    """JIT-compiled reweighting function for ultra-fast performance"""
    # Get weights for valid events
    event_weights = spline_weights[event_to_spline_map]
    
    # Apply weights only to valid events
    monolith = monolith.at[valid_event_indices, MCEventIndices.WEIGHT.value].multiply(event_weights)
    
    return monolith


class SplineSystematicModel:

    def __init__(self, spline_file: SplineFile, systematics: Union[SystematicFile, SystematicHandler]):
        self.spline_file = spline_file
        if isinstance(systematics, SystematicFile):
            self.systematic_handler = systematics.systematic_handler
        else:
            self.systematic_handler = systematics
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
        self.spline_file.monolith.map_splines_to_syst(self._index_tensor[:,self.SYST_INDEX:self.SPLINE_INDEX+1])

    def get_monolith_splines(
        self, mc_event_monolith: MCEventMonolith, bin_indices: jnp.ndarray
    ) -> jnp.ndarray:
        """Find the bin for each event in the monolith, bins must be in order x,y,z,..."""
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
        
        # Find first matching spline for each event (-1 if no match)
        # Shape: (n_events,)
        self._event_to_spline_map = jnp.where(
            jnp.any(matches, axis=1),  # Has any match
            jnp.argmax(matches, axis=1),  # Index of first match
            -1  # No match found
        )

        # Remove events that don't have matching splines
        valid_mask = self._event_to_spline_map >= 0
        self._event_to_spline_map = self._event_to_spline_map[valid_mask]
        self._valid_event_indices = jnp.where(valid_mask)[0]
        
        # Saves rebuilding every loop
        self._spline_value_arr = jnp.zeros(len(self._index_tensor), dtype=jnp.float64)
        return self._event_to_spline_map

    def reweight(self, syst_values: jnp.ndarray, monolith: jnp.ndarray) -> jnp.ndarray:
        """Ultra-fast reweighting for sub-millisecond performance"""
        # Evaluate spline weights outside of JIT (since SplineMonolith isn't JIT-compatible)
        spline_weights = self.spline_monolith(syst_values)
        
        return _reweight_jit(
            monolith, 
            spline_weights,
            self._event_to_spline_map, 
            self._valid_event_indices
        )

    @property
    def index_tensor(self) -> jnp.ndarray:
        return self._index_tensor

    @property
    def spline_monolith(self) -> SplineMonolith:
        return self.spline_file.monolith

    @property
    def mc_indices(self) -> jnp.ndarray:
        return self._event_to_spline_map
