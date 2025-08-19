# Benchmarking script for the reweighting process

import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from magpy.models.sample_model import SampleModel
from magpy.file_io.spline_file import SplineFile
from magpy.models.spline_syst_model import SplineSystematicModel
from magpy.oscillator.oscillator import Oscillator
from magpy.file_io.systematic_file import SystematicFile
from magpy.file_io.mc_file import MCFile
from magpy.objects.mc_event import MCEventIndices

def benchmark_reweight(n_iter: int = 50):
    """
    Benchmark the reweighting process.
    """
    # Load the spline file
    data_folder = Path(__file__).parent.parent / "tests" / "data"

    spline_file_path = data_folder / "converted_splines.root"
    spline_file = SplineFile(spline_file_path)
    
    # load in mc
    mc_file_path = data_folder / "NuWro_FlatTree.root"
    mc_file = MCFile(mc_file_path, "FlatTree_VARS")
    
    mc_file.set_mc_branch(MCEventIndices.TRUE_NEUTRINO_ENERGY, "Enu_true")
    mc_file.set_mc_branch(MCEventIndices.TRUE_Q2, "Q2")
    mc_file.set_mc_const(MCEventIndices.RECO_NEUTRINO_ENERGY, 0)
    mc_file.set_mc_branch(MCEventIndices.INTERACTION_MODE, "Mode")
    mc_file.set_mc_branch(MCEventIndices.TARGET, "tgt")
    # consts
    mc_file.set_mc_const(MCEventIndices.START_NU, 12)
    mc_file.set_mc_const(MCEventIndices.END_NU, 14)
    mc_file.set_mc_const(MCEventIndices.WEIGHT, 0)
    mc_file.fill_monolith()

    
    # load in systematics
    syst_file_path = data_folder / "syst_file.yml"
    syst_file = SystematicFile(syst_file_path)

    
    oscillator = Oscillator(1300, 0.5, 3, 0)    

    bin_vars = torch.tensor([MCEventIndices.TRUE_NEUTRINO_ENERGY.value,
                             MCEventIndices.TRUE_Q2.value,
                             MCEventIndices.DUMMY.value])

    handler = SampleModel(mc_file, spline_file, syst_file, oscillator=oscillator)

    handler.set_bin_variables(bin_vars)
    handler.initialise_mc_indices()
    osc_reweight = torch.tensor([0.31,0.02, 0.55, 0.7 * torch.pi, 7.5e-5, 2.5e-3])
    syst_vals = torch.tensor([1.1, 1.1, 1.1, 1.1, 1.1])


    times = np.zeros(n_iter)

    for i in range(n_iter):
        osc_mod = osc_reweight.clone() * (
                1.0001 + (np.random.uniform(0, 1) / (n_iter * 100))
            )
    
        syst_mod = syst_vals.clone() * (
                1.0001 + (np.random.uniform(0, 1) / (n_iter * 100))
            )

        
        start = time.time()
        handler.reweight(osc_mod, syst_mod)
        times[i] = time.time() - start

    print(f"Average time per iteration: {np.mean(times):.6f} seconds")
    
    return times

def plot_hist(times, plot_file="reweight_histogram.png"):
    """
    Plot a histogram of the reweighting times.
    """

    plt.hist(times * 1000, bins=50, alpha=0.7)
    plt.xlabel("Reweight Time (milliseconds)")
    plt.title("Reweighting Time Histogram")
    plt.grid()
    plt.savefig(plot_file)
    plt.close()