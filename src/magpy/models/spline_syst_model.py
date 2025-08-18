'''
Bins splines
'''
from typing import List

import torch
import numpy as np
import pandas as pd

from magpy.file_io.spline_file import SplineFile
from magpy.objects.systematic_handler import SystematicHandler
from magpy.utils.modes import SplineModes, bins_to_spline_name

class SplineSystematicModel:
    def __init__(self, spline_file: SplineFile, systematic_handler: SystematicHandler):
        self.spline_file = spline_file
        self.systematic_handler = systematic_handler
        self._index_tensor = None
        
    def setup_splines(self, sample_binning: List[float]):
        """Setup the splines from the spline file.
            We want a unique associate for each bin, mode and syst to each spline         
        """

        # Write as dataframe for my own sanity first
        index_frame = pd.DataFrame(columns=["syst", "mode", "spline_index"])
        for i in range(len(sample_binning)):
            index_frame[f"bin_{i}"] = pd.Series(dtype='int')
            
        for i_syst, systematic in enumerate(self.systematic_handler.systematics):
            # Skip non-spline systematics
            if systematic.syst_type != "spline":
                continue

            syst_spline_name = systematic.spline_name
            
            for m in systematic.modes:
                mode_name = SplineModes(m).name

                for isamp, sample_bins in enumerate(sample_binning):
                    for ibin, _ in enumerate(sample_bins):
                        spline_name = bins_to_spline_name(syst_spline_name, mode_name, spline_bins)
                        spline_index = self.spline_file.spline_names.index(spline_name)

                        index_frame.loc[-1] = [i_syst, m, spline_index]
                        for i, b in enumerate(spline_bins):
                            index_frame.loc[-1][f"bin_{i}"] = b
                            
                        index_frame.index += 1
                
                self._index_tensor = torch.tensor(index_frame.values, dtype=torch.int64)
            
if __name__ == "__main__":
    
    from magpy.objects.systematic_handler import Systematic

    spline_file_name = "/Users/henrywallace/software/magpy/converted_splines.root"
    spline_file = SplineFile(spline_file_name)
    
    syst_a = Systematic(
        syst_name="syst1", 
        spline_name="mysyst1",
        modes=[0],
        syst_type="spline",
        nominal=1.0,
        error=0.1,
        range=(0.0, 999.0)
    )
    
    handler = SystematicHandler([syst_a])

    model = SplineSystematicModel(spline_file, handler)
    sample_binning = [[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                                5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
                               
                               [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                                5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5],
                                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]

    model.setup_splines(sample_binning)