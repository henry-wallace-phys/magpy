from typing import List, Dict

import torch

from magpy.objects.mc_event import MCEventMonolith
from magpy.objects.kinematic_handler import KinematicBinning
from magpy.objects.spline_handler import SplineMonolith
from magpy.objects.systematic_handler import Systematic


class MagpySample:
    def __init__(
        self,
        MCEvents: MCEventMonolith,
        spline: SplineMonolith,
        systematics: List[Systematic],
        binning: Dict[str, List[float]],
    ):
        self.MCEvents = MCEvents
        self.spline = spline
        self.systematics = systematics
        self.binning = binning

        # Set up kinematic handlers
        self._kinematic_bins = [
            KinematicBinning(name, torch.tensor(binning))
            for name, binning in self.binning.items()
        ]
