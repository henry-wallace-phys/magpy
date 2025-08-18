from typing import List

from scipy.interpolate import CubicSpline
import torch

from magpy.Exceptions import MagpySplineException

# '''
# Code for defining spline objects in Torch
# '''

# # ---------------------------------------------------------------------------
class Spline:
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self._is_flat = (len(set(y.cpu().numpy())) < 2) 
        
        if self._is_flat:
            self._spline_stack = torch.tensor([0,0,0,0,0], dtype=torch.float64, device=x.device)
        else:
            spline = CubicSpline(x.cpu().numpy(), y.cpu().numpy())
            knots = torch.tensor(spline.x).to(device=x.device, dtype=torch.float64)
            coefs = torch.tensor(spline.c).to(device=x.device, dtype=torch.float64)
            self._spline_stack = torch.vstack((knots[:-1], coefs)).T
            
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
    FLAT_SPLINE = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float64)

    def __init__(self, splines: List[Spline]):
        # Indices
        self._n_splines = len(splines)
        
        if self._n_splines == 0:
            raise MagpySplineException("No splines provided to monolith")
        
        
        self._indices = torch.cumsum(torch.tensor([len(spline) for spline in splines]), dim=0)
        # flat splines
        
        self._flat_splines = torch.tensor([spline._is_flat for spline in splines], dtype=torch.bool)
        
        # Spline monolith
        
        self._spline_monolith = torch.vstack([spline.spline for spline in splines if not spline.is_flat]) 
        

    @property
    def indices(self):
        '''
        Get indices
        '''
        return self._indices

    @property
    def monolith(self):
        '''
        Get the monolith
        '''
        return self._spline_monolith
    
    def is_flat(self, item: int) -> bool:
        return self._flat_splines[item]
    
    def __len__(self):
        return self._n_splines
    
    def __getitem__(self, item: int):
        if item >= len(self._indices):
            raise IndexError("Index out of range")

        if self.is_flat(item):
            return SplineMonolith.FLAT_SPLINE


        # Return item
        if item == len(self._indices) - 1:
            return self._spline_monolith[self._indices[item-1]:]
        else:            
            return self._spline_monolith[self._indices[item-1]:self._indices[item]]

    def flat_indices(self)->torch.Tensor:
        return self._flat_splines

    def __call__(self, x: torch.Tensor)->torch.Tensor:
        '''
        Evaluate a vector of splines and return the weights
        '''
        weights = torch.ones(len(x), dtype=torch.float64, device=x.device)
        # Reduce to non-flat
        x_non_flat = x[~self._flat_splines]
        
        non_flat_indices = self._indices[~self._flat_splines]
        
        if len(x_non_flat) != len(non_flat_indices):
            raise ValueError("Input tensor x must have the same length as the number of spline segments.")


        knot_indices = torch.zeros(len(x_non_flat), dtype=torch.int64)
        for i, s in enumerate(x_non_flat):
            if i == 0:
                low, high = 0, non_flat_indices[0]
            else:
                low, high = non_flat_indices[i-1], non_flat_indices[i]
    
            knots = self._spline_monolith[low:high, 0].contiguous()
            
            knot_indices[i] = max(torch.searchsorted(knots, s).item()-1, 0)

        # Get the coefficients for the segments        
        coefs = self._spline_monolith[knot_indices]
        
        
        # Now we calculate the polynomial value

        # Get the differences
        dx = x_non_flat - coefs[:, 0]

        # Get the weights
        weights[~self._flat_splines] = ((coefs[:, 1]*dx + (coefs[:, 2]))*dx + coefs[:, 3])*dx + coefs[:, 4]

        return weights
    
    