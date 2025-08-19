from typing import List, Optional

from scipy.interpolate import CubicSpline
import torch

from magpy.Exceptions import MagpySplineException
from magpy.utils.device_manager import DeviceManager

# '''
# Code for defining spline objects in Torch
# '''


# # ---------------------------------------------------------------------------
class Spline:
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self._is_flat = len(set(y.cpu().numpy())) < 2

        if self._is_flat:
            self._spline_stack = torch.tensor(
                [0, 0, 0, 0, 0], dtype=torch.float64, device=DeviceManager().get_device()
            )
            self._spline_stack = self._spline_stack.to(DeviceManager().get_device())
        else:
            spline = CubicSpline(x.cpu().numpy(), y.cpu().numpy())
            knots = torch.tensor(spline.x).to(device=x.device, dtype=torch.float64)
            coefs = torch.tensor(spline.c).to(device=x.device, dtype=torch.float64)
            self._spline_stack = torch.vstack((knots[:-1], coefs)).T
            self._spline_stack.to(torch.float64).to(DeviceManager().get_device())

    @property
    def spline(self):
        return self._spline_stack

    def __len__(self):
        if self._is_flat:
            return 0
        else:
            return self._spline_stack.shape[0]

    @property
    def is_flat(self):
        return self._is_flat


class SplineMonolith:
    DEVICE = DeviceManager().get_device()
    FLAT_SPLINE = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float64, device=DEVICE)

    def __init__(self, splines: List[Spline]):
        # Indices
        self._n_splines = len(splines)

        if self._n_splines == 0:
            raise MagpySplineException("No splines provided to monolith")

        self._indices = torch.cumsum(
            torch.tensor([len(spline) for spline in splines], dtype=torch.int64, device=self.DEVICE), dim=0,
        )
        # flat splines

        self._flat_splines = torch.tensor(
            [spline._is_flat for spline in splines], dtype=torch.bool,
            device=self.DEVICE
        )

        # Spline monolith

        self._spline_monolith = torch.vstack(
            [spline.spline for spline in splines if not spline.is_flat],
        )
        self._spline_monolith = self._spline_monolith.to(self.DEVICE)

        self._spline_syst_map = None

        # PRE-CALCULATE: Move expensive computations out of __call__
        self._setup_fast_lookup()
        

    def _setup_fast_lookup(self):
        """Pre-calculate lookup structures for fast spline evaluation"""
        # Pre-calculate knot ranges for each non-flat spline
        self._knot_ranges = torch.zeros((self._indices.shape[0], 2), dtype=torch.int64, device=self.DEVICE)
        for i in range(self._indices.shape[0]):
            if i == 0:
                low, high = 0, self._indices[0]
            else:
                low, high = self._indices[i - 1], self._indices[i]
            self._knot_ranges[i] = torch.tensor([low, high], device=self.DEVICE)

        # Pre-extract all knot sequences for faster access
        self._knot_sequences = []
        for low, high in self._knot_ranges:
            knots = self._spline_monolith[low:high, 0].contiguous()
            self._knot_sequences.append(knots)

    def map_splines_to_syst(self, spline_syst_map: torch.Tensor):
        self._spline_syst_map = spline_syst_map.to(self.DEVICE)
        self._dim = len(spline_syst_map)
        self._n_syst = len(torch.unique(self._spline_syst_map[:, 0]))

        self._n_non_flat = self._dim - len(self._flat_splines)

        # We can also cache the number of splines for each index
        self._par_splines = []
        # Remove flat splines from spline syst map
        non_flat_spline_syst_map = self._spline_syst_map[self._spline_syst_map[:,1][~self._flat_splines]]

        for i in range(self._n_syst):
            # Get all splines for this systematic
            splines_for_par = non_flat_spline_syst_map[non_flat_spline_syst_map[:, 0] == i][:,1]
            self._par_splines.append(splines_for_par)

        self._weights = torch.ones(self._dim, dtype=torch.float64, device=self.DEVICE)
        self._par_arr = torch.zeros(self._dim, dtype=torch.float64, device=self.DEVICE)
        self._knot_indices = torch.zeros(self._dim, dtype=torch.int64, device=self.DEVICE)
    
        
    def __getitem__(self, item: int):
        if item >= len(self._indices):
            raise IndexError("Index out of range")

        if self.is_flat(item):
            return SplineMonolith.FLAT_SPLINE

        # Return item
        if item == len(self._indices) - 1:
            return self._spline_monolith[self._indices[item - 1] :]
        else:
            return self._spline_monolith[self._indices[item - 1] : self._indices[item]]


    def is_flat(self, item: int) -> bool:
        return self._flat_splines[item].item()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate a vector of splines and return the weights - OPTIMIZED
        """
        if self._spline_syst_map is not None:
            return self.get_knots_grouped(x)
        else:
            return self.get_knots_ungrouped(x)

    def get_knots_grouped(self, x: torch.Tensor) -> torch.Tensor:
        '''
        ULTRA-OPTIMIZED: For when you know what maps to what
        All non-flat splines for a given systematic share the same knots!
        '''
        if len(x) != self._n_syst:
            raise MagpySplineException(
                f"Input tensor x must have length {self._n_syst} (number of systematics)."
            )
           
        # We now loop over the systematic parameters
        for i, par in enumerate(x):
            # Get the splines for this systematic
            splines = self._par_splines[i]
            # Knots are shared so we just need the first value
            knot = self._knot_sequences[splines[0]]
            # Get the index
            self._knot_indices[splines] =  max(torch.searchsorted(knot, par).item() - 1, 0)
            self._knot_indices[splines] += self._knot_ranges[splines, 0]
            self._par_arr[splines] = par.to(torch.float64)

        coefs = self._spline_monolith[self._knot_indices][~self._flat_splines]
        reduced_par_arr = self._par_arr[~self._flat_splines]

        dx = reduced_par_arr - coefs[:, 0]
        self._weights[~self._flat_splines] = (
            (coefs[:, 1] * dx + (coefs[:, 2])) * dx + coefs[:, 3]
        ) * dx + coefs[:, 4]

        return self._weights

    def get_knots_ungrouped(self, x: torch.Tensor)->torch.Tensor:
        '''
        For when you just have a load of spline parameters
        '''
        
        weights = torch.ones(len(x), dtype=torch.float64, device=x.device)
        
        # Use pre-calculated non-flat data
        x_non_flat = x[~self._flat_splines]

        knot_indices = torch.zeros(len(x_non_flat), dtype=torch.int64, device=x_non_flat.device)
        for i, (s, knots) in enumerate(zip(x_non_flat, self._knot_sequences)):
            knot_indices[i] = max(torch.searchsorted(knots, s).item() - 1, 0)
            # Adjust for global monolith indexing
            knot_indices[i] += self._knot_ranges[i][0]

        # Get the coefficients for the segments
        coefs = self._spline_monolith[knot_indices]

        if len(x_non_flat) == 0:
            return weights

        # OPTIMIZED: Use pre-calculated knot sequences

        # Calculate polynomial value
        dx = x_non_flat - coefs[:, 0]
        weights[~self._flat_splines] = (
            (coefs[:, 1] * dx + (coefs[:, 2])) * dx + coefs[:, 3]
        ) * dx + coefs[:, 4]

        return weights

