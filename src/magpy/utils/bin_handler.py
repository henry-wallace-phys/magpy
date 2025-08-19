'''
Some utils for converting bins -> int and back
'''

from typing import List, Tuple
import warnings

import numpy as np
import torch


class OutOfBoundsBinWarning(Warning):
    pass

class BinHandler:
    def __init__(self, bin_edges: List[List[float]]):
        
        # Whole thing can't be tensor but we can make this one
        self.bin_edges = [torch.tensor(b) for b in bin_edges]
        self._bin_edge_dims = [len(b)-1 for b in bin_edges]
        self._bin_edge_tensor, self._bin_edge_indices = self._generate_bin_tuples()

    def _generate_bin_tuples(self)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        Now we want to get all coordinate tuples from self.bin_edges
        
        i.e. [(0,0,0), (0,0,1), ...]
        '''
        idx_list = torch.tensor(list(np.ndindex(*self._bin_edge_dims)))
        
        # Now put BACK in the bin edges
        bin_list = torch.zeros((len(idx_list), len(self.bin_edges),2), dtype=torch.float64)
        index_list = torch.zeros((len(idx_list), len(self.bin_edges)), dtype=torch.int)
        for i, indices in enumerate(idx_list):
            for j, idx in enumerate(indices):
                bin_list[i, j] = self.bin_edges[j][idx:idx+2]
                index_list[i, j] = idx
        return bin_list, index_list

    def get_bin_from_int(self, index: torch.Tensor) -> torch.Tensor:
        '''
        Get the bin edges from the index
        '''
        if torch.any(index >= len(self._bin_edge_tensor)):
            warnings.warn("Index out of bounds, returning [-1, -1] for invalid indices.")

        return_tensor = torch.zeros((len(index), len(self.bin_edges), 2), dtype=torch.float64, device=index.device)

        return_tensor[index<len(self._bin_edge_tensor)] = self._bin_edge_tensor[index[index<len(self._bin_edge_tensor)-1]]
        return_tensor[index>=len(self._bin_edge_tensor)] = torch.tensor([-1.0, -1.0],dtype=return_tensor.dtype, device=return_tensor.device)

        return return_tensor

    def find_bin(self, kinematic: torch.Tensor) -> torch.Tensor:
        '''
        Get the index from the bin edges
        '''
        # if kinematic.shape[1] != self._indices.shape[1]:
        #     raise MagpyBinException(f"Input kinematic shape {kinematic.shape} does not match bin edges shape {self._indices.shape}.")

        bin_indices = torch.zeros((kinematic.shape), dtype=torch.int, device=kinematic.device)
        

        for i, edge in enumerate(self.bin_edges):            
            bin_indices[:,i] = torch.searchsorted(edge[:-1], (kinematic[:,i].contiguous()))-1
            bin_indices[:,i][kinematic[:,i] < edge[0]] = -1
            bin_indices[:,i][kinematic[:,i] >= edge[-1]] = len(edge) - 1

            # if torch.any(bin_indices[:,i] < 0) or torch.any(bin_indices[:,i] >= len(self.bin_edges[i]) - 1):
            #     warnings.warn("Some kinematic values are out of bounds for the bin edges.", OutOfBoundsBinWarning)
        return bin_indices


    @property
    def bin_indices(self)->torch.Tensor:
        return self._bin_edge_indices
    
    @property
    def bin_edges_tensor(self)->torch.Tensor:
        return self._bin_edge_tensor