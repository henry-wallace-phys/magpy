"""
Bins splines and get a big indexing tensor
"""

from typing import List

import numpy as np
import torch
from tqdm import tqdm

from magpy.file_io.spline_file import SplineFile
from magpy.file_io.systematic_file import SystematicFile, SystematicHandler
from magpy.utils.modes import SplineModes, bins_to_spline_name
from magpy.objects.mc_event import MCEventMonolith, MCEventIndices
from magpy.objects.spline_handler import SplineMonolith

class SplineSystematicModel:
    def __init__(self, spline_file: SplineFile, systematics: SystematicFile | SystematicHandler):
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
                    spline_idx = self.spline_file.spline_names.index(spline_name)
                    output = [isyst, spline_idx, mode]
                    output.extend(bins.tolist())
                    out_list.append(output)
        self._index_tensor = torch.tensor(out_list, dtype=torch.int)
        self.spline_file.monolith.map_splines_to_syst(self._index_tensor[:,self.SYST_INDEX:self.SPLINE_INDEX+1])

    def get_monolith_splines(
        self, mc_event_monolith: MCEventMonolith, bin_indices: torch.Tensor
    ) -> torch.Tensor:
        """Find the bin for each event in the monolith, bins must be in order x,y,z,..."""
        use_dummy = MCEventIndices.DUMMY.value in bin_indices

        self._bins = torch.cat([torch.tensor([MCEventIndices.INTERACTION_MODE.value]), bin_indices], dim=0)

        monolith_kinematics = mc_event_monolith.monolith[
            :, bin_indices[bin_indices != MCEventIndices.DUMMY.value]
        ]

        if use_dummy:
            monolith_kinematics = torch.cat(
                (
                    monolith_kinematics,
                    torch.ones(
                        (len(monolith_kinematics), 1),
                        dtype=torch.float64,
                        device=monolith_kinematics.device,
                    ),
                ),
                dim=1,
            )

        # Now we get the index of the full indices in the index tensor
        monolith_kinematic_bins = self._bins_handler.find_bin(monolith_kinematics)
        modes = (
            mc_event_monolith._mc_event_monolith[
                :, int(MCEventIndices.INTERACTION_MODE.value)
            ]
            .unsqueeze(-1)
            .to(torch.int32)
        )

        full_index_array = torch.cat((modes, monolith_kinematic_bins), dim=1)

        # OPTIMIZED: Create lookup table for fast matching
        # Convert index tensor to hashable format for lookup
        index_keys = self._index_tensor[:, self.MODE_INDEX:].contiguous()
        
        # Create mapping from index tuple to spline indices
        self._event_to_spline_map = torch.full((len(full_index_array),), -1, dtype=torch.long)
                
        # Vectorized matching - much faster than the loop
        for i, event_idx in enumerate(full_index_array):
            # Find matching rows using broadcasting
            matches = torch.all(index_keys == event_idx.unsqueeze(0), dim=1)
            if torch.any(matches):
                # Get the first matching spline index
                spline_idx = torch.where(matches)[0][0]
                self._event_to_spline_map[i] = spline_idx


        # Remove events that don't have matching splines
        valid_mask = self._event_to_spline_map >= 0
        self._event_to_spline_map = self._event_to_spline_map[valid_mask]
        self._valid_event_indices = torch.where(valid_mask)[0]
        
        # Saves rebuilding every loop
        self._spline_value_arr = torch.zeros(len(self._index_tensor), dtype=torch.float64)


        return self._event_to_spline_map

    
    # @torch.jit.script
    def reweight(self, syst_values: torch.Tensor, monolith: torch.Tensor) -> torch.Tensor:
        """Ultra-fast reweighting for sub-millisecond performance"""
        # Get weights for valid events
        event_weights =self.spline_monolith(syst_values)[self._event_to_spline_map]
        
        # Apply weights only to valid events
        monolith[self._valid_event_indices, MCEventIndices.WEIGHT.value] *= event_weights
        
        return monolith

    @property
    def index_tensor(self) -> torch.Tensor:
        return self._index_tensor

    @property
    def spline_monolith(self) -> SplineMonolith:
        return self.spline_file.monolith

    @property
    def mc_indices(self) -> torch.Tensor:
        return self._event_to_spline_map