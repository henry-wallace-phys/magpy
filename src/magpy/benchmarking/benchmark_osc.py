import time
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from rich import print
from cProfile import Profile
from pstats import Stats

from tqdm import tqdm
from magpy.oscillator.oscillator import Oscillator
import jax.numpy as jnp
from magpy.oscillator.nu_types import NuType   

def benchmark_osc(n_iter: int = 50, n_scales: int = 100000, points: int = 100):

    osc_pars = jnp.array(
        [0.3, 0.02, 0.55, 0.7 * np.pi, 7.5e-5, 2.5e-3],
        dtype=jnp.float64,
    )

    scales = np.arange(n_scales // points, n_scales, n_scales // points)

    times = np.zeros((n_iter, len(scales)), dtype=jnp.float64)

    # Warm up to ensure compilation of everything etc.


    print("Starting osc benchmark...")
    with Profile() as pr:
        pr.disable()
        for j, event_scale in tqdm(
            enumerate(scales), desc="Event scale", total=len(scales)
        ):
            # Silly but helps
            if j == 0:
                print("Burning a few cycles in...")
            osc_in = jnp.array(
                np.random.choice(
                    [NuType.ELECTRON.value, NuType.MUON.value, NuType.TAU.value], size=event_scale
                ),
                dtype=jnp.int64,
            )
            osc_out = jnp.array(
                np.random.choice(
                    [NuType.ELECTRON.value, NuType.MUON.value, NuType.TAU.value], size=event_scale
                ),
                dtype=jnp.int64,
            )
            energies = jnp.array(
                np.linspace(0.1, 10, event_scale), dtype=jnp.float64
            )  # Example energy range

            oscillator = Oscillator(1300, 0.5, 3, 0)
            oscillator.set_energy_osc(energies, osc_in, osc_out)
            oscillator.calc_probability(osc_pars)

            
            for i in range(n_iter):
                osc_mod = osc_pars.clone() * (
                    1.0001 + (np.random.uniform(0, 1) / (n_iter * 100))
                )
                pr.enable()
                start_time = time.time()
                _ = oscillator.calc_probability(osc_mod)
                end_time = time.time()
                pr.disable()
                times[i, j] = end_time - start_time
        
        Stats(pr).strip_dirs().sort_stats("cumulative").print_stats()

    print("Benchmark complete.")
    return times, scales


def plot_scaling(times: np.ndarray, scales: np.ndarray, plot_file="osc_scaling.png"):
    mean_times = times[2:].mean(axis=0)
    std_times = times[2:].std(axis=0)

    plt.figure()
    plt.errorbar(scales, mean_times * 1_000, yerr=std_times * 1_000, fmt="o-")
    plt.xlabel("N Events")
    plt.ylabel("Time (ms)")
    plt.title("Oscillator Benchmark Scaling")
    plt.grid()
    plt.savefig(plot_file)
    plt.close()

    # Now we do times/scale
    plt.figure()
    plt.plot(scales, mean_times / scales * 1_000_000, label="Mean Time per event [μs]")
    plt.xlabel("Time/Event [μs]")
    plt.ylabel("Number of Events")
    plt.title("Mean Time per Event")
    plt.grid()

    plot_path = Path(plot_file)
    # Get extensionless path and extension
    ext = plot_path.suffix
    plot_path = plot_path.with_suffix("")

    plt.savefig(f"{plot_path}_per_event{ext}")
    plt.close()
    
    plt.figure()
    plt.hist(
        mean_times[
            np.abs(mean_times - np.mean(mean_times)) < 3 * np.std(mean_times)
        ]
        * 1_000_000,
        bins=50,
        alpha=0.7,
    )
    plt.xlabel("Oscillation Time (μs)")
    plt.title("Average time per oscillation per event")
    plt.grid()
    plt.savefig(plot_file)
    plt.close()
    
    print(
        f"Average time: ([bold]{np.mean(mean_times)*1_000_000}±{np.std(mean_times)*1_000_000} μs[/bold])"
    )



def plot_times_per_iter(times: np.ndarray, scales: np.ndarray, plot_file="osc_times_per_iter.png"):
    """
    Plot the average time per iteration for each scale.
    """
    for s in range(len(scales)):
        plt.plot(times[10:,s] * 1_000, label=f"N Events {scales[s]}")

    plt.xlabel("Iteration")
    plt.ylabel("Time (ms)")
    plt.title("Oscillator Benchmark Scaling")
    plt.grid()
    plt.legend()
    plt.savefig(plot_file)
    plt.close()



def plot_average_time_per_event(
    times: np.ndarray, scales: np.ndarray, plot_file="osc_average_time_event.png"
):

    times_avged = np.zeros(len(scales) * len(times), dtype=np.float64)
    for i in range(len(scales)):
        times_avged[i * len(times) : (i + 1) * len(times)] = times[:, i] / scales[i]

    print(
        f"Average time: ([bold]{np.mean(times_avged)*1_000_000}±{np.std(times_avged)*1_000_000} μs[/bold])"
    )

    plt.figure()
    plt.hist(
        times_avged[
            np.abs(times_avged - np.mean(times_avged)) < 3 * np.std(times_avged)
        ]
        * 1_000_000,
        bins=50,
        alpha=0.7,
    )
    plt.xlabel("Reweight Time (μs)")
    plt.title("Average time per reweight per event")
    plt.grid()
    plt.savefig(plot_file)
    plt.close()
