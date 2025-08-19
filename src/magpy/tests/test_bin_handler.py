from typing import List 

import pytest
import torch

import logging
from magpy.utils.bin_handler import BinHandler

TEST_KINEMATIC_INDICES = torch.tensor([0.1, 2,100,1], dtype=torch.float32)

EXPECTED_BIN_INDICES = torch.tensor([0, 2, 5, 1], dtype=torch.int)

EXPECTED_BINS = torch.tensor([
        [[0.0000, 1.5000],
         [2.5000, 3.5000],
         [0.0000, 1.5000]],

        [[-1.0, -1.0],
         [-1.0, -1.0],
         [-1.0, -1.0]],

        [[0.0000, 1.5000],
         [1.5000, 2.5000],
         [0.0000, 1.5000]]], dtype=torch.float64)



def test_bin_index_conversion():
    '''
    Test bin handler's ability to test indices
    '''
    bin_edges = [
        [0, 1.5, 2.5, 3.5],
        [0, 1.5, 2.5, 3.5, 4.5, 5.5],
        [0, 1.5]
    ]
    
    indices = torch.tensor([0, 1, 2])
    
    expected_bins = torch.tensor(
       [[[0.0000, 1.5000],
         [0.0000, 1.5000],
         [0.0000, 1.5000]],

        [[0.0000, 1.5000],
         [1.5000, 2.5000],
         [0.0000, 1.5000]],

        [[0.0000, 1.5000],
         [2.5000, 3.5000],
         [0.0000, 1.5000]]], dtype=torch.float64)

    
    handler = BinHandler(bin_edges)
    
    output_bins = handler.get_bin_from_int(indices)
    
    print(output_bins)
    assert torch.allclose(output_bins, expected_bins), f"Expected {expected_bins}, got {output_bins}"



def test_bin_finding():
    '''
    Test bin handler's ability to find indices from kinematic data
    '''
    bin_edges = [
        [0, 1.5, 2.5, 3.5],
        [0, 1.5, 2.5, 3.5, 4.5, 5.5],
        [0, 1.5]
    ]

    kinematics = torch.tensor([[0.1, 6, -0.5],
                                [2, 4, 1]
                                ], dtype=torch.float64)
    
    expected_output = torch.tensor([[0, 6, -1], [1, 3, 0]], dtype=torch.int)
    
    handler = BinHandler(bin_edges)
    
    output_indices = handler.find_bin(kinematics)
    print("HI", output_indices)
    
    assert torch.equal(output_indices, expected_output), f"Expected {expected_output}, got {output_indices}"
    
