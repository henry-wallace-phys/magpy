import uproot
from tqdm import tqdm
import torch

from magpy.file_io.root_file import RootFile
from magpy.objects.spline_handler import Spline, SplineMonolith
from magpy.utils.bin_handler import BinHandler
from magpy.Exceptions import MagpyInvalidObjectError

"""
Note: This file needs to be in the ROOT file format expected in the magpy ecosystem
For now this is simply:

file -> syst_name -> TGraphs
"""


class SplineFile(RootFile):
    """Spline file handler for the magpy ecosystem."""
    BINNING_HIST_STR = "dev_tmp.0.0;1"
    def __init__(self, file_path: str):
        """Initialize the SplineFile class.

        :param file_path: Path to the ROOT file.
        :type file_path: str
        """
        print(f"Loading spline file: {file_path}")
        super().__init__(file_path)

        # self.systematic_names = [
        #     key.rstrip("/").rstrip(";")
        #     for key in self.file.keys()
        #     if isinstance(self.file[key], uproot.ReadOnlyDirectory)
        # ]
        print("Extracting splines")
        # We now convert the TGraphs to CubicSpline objects
        spline_array = []
        self._spline_names = []
        
        self.binning_hist = None
        
        for graph_name in tqdm(self.file.keys(), desc="Systematic"):

            gr = self.file[graph_name]

            try:
                x, y = gr.values()
                spline_array.append(Spline(torch.tensor(x), torch.tensor(y)))
                
                # Remove backup
                graph_name = graph_name.split(";")[0]
                
                self._spline_names.append(graph_name)
            except Exception:
                continue

        # We now get the binning hist
        
        
        # Use hist to get axis
        self.binning_hist = self.file.get(self.BINNING_HIST_STR)
        if self.binning_hist is not None:
            x_bins = self.binning_hist.axis(0).edges()
            y_bins = self.binning_hist.axis(1).edges()
            z_bins = self.binning_hist.axis(2).edges()
            self.bins = BinHandler([x_bins, y_bins, z_bins])
        else:
            raise MagpyInvalidObjectError(f"Binning histogram {self.BINNING_HIST_STR} not found")

        self._monolith = SplineMonolith(spline_array)

    @property
    def monolith(self):
        return self._monolith

    @property
    def spline_names(self):
        return self._spline_names


    def get_bin_handler(self):
        return self.bins
    
