'''
Bins splines and get a big indexing tensor
'''
from typing import List

import torch
from tqdm import tqdm
import numpy as np


from magpy.file_io.spline_file import SplineFile
from magpy.objects.systematic_handler import SystematicHandler
from magpy.utils.modes import SplineModes, bins_to_spline_name
from magpy.objects.mc_event import MCEventMonolith, MCEventIndices


class SplineSystematicModel:
    def __init__(self, spline_file: SplineFile, systematic_handler: SystematicHandler):
        self.spline_file = spline_file
        self.systematic_handler = systematic_handler
        self.setup_splines()
    
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
        self.BIN_INDEX = [i+3 for i in range(len(self._bins_handler.bin_indices[0]))]

        for isyst, syst in tqdm(enumerate(self.systematic_handler.systematics), desc="Processing systematics"):
            for imode, mode in enumerate(syst.modes):
                mode_name = SplineModes(mode).spline_name()
                for bins in self._bins_handler.bin_indices:
                    spline_name = bins_to_spline_name(syst.spline_name, mode_name, bins.tolist())
                    # Get spline
                    spline_idx = self.spline_file.spline_names.index(spline_name)
                    output = [isyst, spline_idx, mode]
                    output.extend(bins.tolist())
                    out_list.append(output)
        

        self._index_tensor = torch.tensor(out_list, dtype=torch.int)

    def get_monolith_splines(self, mc_event_monolith: MCEventMonolith, bin_indices: torch.Tensor) -> List[torch.Tensor]:
        """Find the bin for each event in the monolith, bins must be in order x,y,z,...
        """
        use_dummy = MCEventIndices.DUMMY.value in bin_indices

        monolith_kinematics = mc_event_monolith.monolith[:,bin_indices[bin_indices != MCEventIndices.DUMMY.value]]

        if use_dummy:
            monolith_kinematics = torch.cat((monolith_kinematics, torch.ones((len(monolith_kinematics), 1), dtype=torch.float64, device=monolith_kinematics.device)), dim=1)        
        
        
        if monolith_kinematics.shape[1] != len(bin_indices):
            raise ValueError(f"Monolith indices shape {monolith_kinematics.shape} does not match index tensor shape {self._index_tensor.shape}")
        
        # Now we get the index of the full indices in the index tensor
        monolith_kinematic_bins = self._bins_handler.find_bin(monolith_kinematics)
        modes = mc_event_monolith._mc_event_monolith[:,int(MCEventIndices.INTERACTION_MODE.value)].unsqueeze(-1).to(torch.int32)

        full_index_array = torch.cat((modes, monolith_kinematic_bins), dim=1)

        # Now we get rows which match full index array as a list
        return_rows =[]

        for idx in full_index_array:
            # Get the row in the index tensor which matches
            matching_rows = torch.all(self._index_tensor[:, self.MODE_INDEX:] == idx, dim=1)
            if not torch.any(matching_rows):
                raise ValueError(f"No matching spline found for index {idx}")
            return_rows.append(self._index_tensor[matching_rows].tolist())
        
        return return_rows

    @property
    def index_tensor(self) -> torch.Tensor:
        return self._index_tensor
