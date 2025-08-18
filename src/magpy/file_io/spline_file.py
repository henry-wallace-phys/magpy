import uproot
from tqdm import tqdm
import torch

from magpy.file_io.root_file import RootFile
from magpy.objects.spline_handler import Spline, SplineMonolith

"""
Note: This file needs to be in the ROOT file format expected in the magpy ecosystem
For now this is simply:

file -> syst_name -> TGraphs
"""


class SplineFile(RootFile):
    """Spline file handler for the magpy ecosystem."""

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
        for graph_name in tqdm(self.file.keys(), desc="Systematic"):
            gr = self.file[graph_name]

            try:
                x, y = gr.values()
                spline_array.append(Spline(torch.tensor(x), torch.tensor(y)))
                self._spline_names.append(graph_name)
            except Exception:
                continue

        self._monolith = SplineMonolith(spline_array)

    @property
    def monolith(self):
        return self._monolith

    @property
    def spline_names(self):
        return self._spline_names
