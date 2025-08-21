# Benchmarking script for the reweighting process

import time
from pathlib import Path
from cProfile import Profile
from pstats import Stats


from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from magpy.models.sample_model import SampleModel
from magpy.file_io.spline_file import SplineFile
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

    bin_vars = jnp.array([MCEventIndices.TRUE_NEUTRINO_ENERGY.value,
                             MCEventIndices.TRUE_Q2.value,
                             MCEventIndices.DUMMY.value])

    handler = SampleModel(mc_file, spline_file, syst_file, oscillator=oscillator)

    print("Setting up bin variables...")
    handler.set_bin_variables(bin_vars)
    print("Initialising MC indices...")
    handler.initialise_mc_indices()
    osc_reweight = jnp.array([0.31,0.02, 0.55, 0.7 * jnp.pi, 7.5e-5, 2.5e-3])
    syst_vals = jnp.array([1.1, 1.1, 1.1, 1.1, 1.1])

    times = np.zeros(n_iter)
    # Reweight once to compile
    handler.reweight(osc_reweight, syst_vals)

    with Profile() as pr:
        pr.disable()
        for i in (pbar:=tqdm(range(n_iter))):
            osc_mod = osc_reweight.clone() * (
                    1.0001 + (np.random.uniform(0, 1) / (n_iter * 100))
                )
        
            syst_mod = syst_vals.clone() * (
                    1.0001 + (np.random.uniform(0, 1) / (n_iter * 100))
                )
            
            
            pr.enable()
            start = time.time()
            handler.reweight(osc_mod, syst_mod)
            times[i] = time.time() - start
            pr.disable()
            
            
            if i>10:
                pbar.set_description(f"Reweighting iteration {i+1}/{n_iter}, Time: {np.mean(times[:i]):.4f}s")
            
        Stats(pr).strip_dirs().sort_stats("cumulative").print_stats()

    print(f"Average time per iteration: {np.mean(times[2:]):.6f} seconds")
    
    return times

def plot_hist(times, plot_file="reweight_histogram.png"):
    """
    Plot a histogram of the reweighting times.
    """

    plt.hist(times * 1000, bins=100, alpha=0.7)
    plt.xlabel("Reweight Time (milliseconds)")
    plt.title("Reweighting Time Histogram")
    plt.grid()
    plt.savefig(plot_file)
    plt.close()

def plot_times_per_iter(times, plot_file="reweight_times_per_iter.png"):
    """
    Plot the times per iteration for the reweighting process.
    """

    plt.figure()
    plt.plot(times[10:] * 1000, label="Reweight Time per Iteration")
    
    plt.xlabel("Iteration")
    plt.ylabel("Reweight Time (milliseconds)")
    plt.grid()
    plt.legend()
    plt.savefig(plot_file)
    plt.close()