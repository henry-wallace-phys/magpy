from typing import List, Tuple

import uproot
import numpy as np
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
    """Spline file handler for the magpy ecosystem.
    """

    def __init__(self, file_path: str):
        """Initialize the SplineFile class.

        :param file_path: Path to the ROOT file.
        :type file_path: str
        """
        print(f"Loading spline file: {file_path}")
        super().__init__(file_path)
        
        self.systematic_names = [key.rstrip('/').rstrip(';') for key in self.file.keys() if isinstance(self.file[key], uproot.ReadOnlyDirectory)]
        print("Extracting splines")
        
        self._n_systs = len(self.systematic_names)
        
        # We now convert the TGraphs to CubicSpline objects
        spline_array = []

        for syst in tqdm(self.systematic_names, total=len(self.systematic_names), desc="Loading splines"):
            for spline in self.file[syst].values():                
                x, y = spline.values()
                spline_array.append(Spline(torch.tensor(x), torch.tensor(y)))


        self._monolith = SplineMonolith(spline_array)
        
    @property
    def monolith(self):
        return self._monolith

