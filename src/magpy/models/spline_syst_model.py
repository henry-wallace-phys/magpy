'''
Bins splines and get a big indexing tensor
'''
import torch
from tqdm import tqdm

from magpy.file_io.spline_file import SplineFile
from magpy.objects.systematic_handler import SystematicHandler
from magpy.utils.modes import SplineModes, bins_to_spline_name

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
        bins_handler = self.spline_file.get_bin_handler()

        for isyst, syst in tqdm(enumerate(self.systematic_handler.systematics), desc="Processing systematics"):
            for imode, mode in enumerate(syst.modes):
                mode_name = SplineModes(mode).spline_name()
                for bins in bins_handler.bin_indices:
                    spline_name = bins_to_spline_name(syst.spline_name, mode_name, bins.tolist())
                    # Get spline
                    spline_idx = self.spline_file.spline_names.index(spline_name)
                    output = [isyst, imode, spline_idx]
                    output.extend(bins.tolist())
                    out_list.append(output)
        
        self._index_tensor = torch.tensor(out_list, dtype=torch.int)

    @property
    def index_tensor(self) -> torch.Tensor:
        return self._index_tensor
