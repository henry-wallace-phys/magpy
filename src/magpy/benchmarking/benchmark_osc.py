import time
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from rich import print
import torch
from tqdm import tqdm
from magpy.oscillator.oscillator import Oscillator, NuType
from magpy.utils.device_manager import DeviceManager


def benchmark_osc(n_iter: int = 50, n_scales: int = 100000, points: int = 100):

    device = DeviceManager().get_device()
    osc_pars = torch.tensor(
        [0.3, 0.02, 0.55, 0.7 * np.pi, 7.5e-5, 2.5e-3],
        dtype=torch.float64,
        device=device,
    )

    scales = np.arange(n_scales // points, n_scales, n_scales // points)

    times = torch.zeros((n_iter, len(scales)), dtype=torch.float64, device=device)

    osc_in_tmp = torch.tensor(
        np.random.choice([NuType.E.value, NuType.Mu.value, NuType.Tau.value], size=10),
        dtype=torch.int64,
        device=device,
    )
    osc_out_tmp = torch.tensor(
        np.random.choice([NuType.E.value, NuType.Mu.value, NuType.Tau.value], size=10),
        dtype=torch.int64,
        device=device,
    )
    energies_tmp = torch.tensor(
        np.linspace(0.1, 10, 10), dtype=torch.float64, device=device
    )  # Example energy range

    # Warm up to ensure compilation of everything etc.

    print("Starting osc benchmark...")
    for j, event_scale in tqdm(
        enumerate(scales), desc="Event scale", total=len(scales)
    ):
        # Silly but helps
        if j == 0:
            print("Burning a few cycles in...")
            for i in range(100):
                oscillator_tmp = Oscillator(1300, 0.5, 3, 0)
                oscillator_tmp.calc_probability(
                    osc_pars, energies_tmp, osc_in_tmp, osc_out_tmp
                )

        osc_in = torch.tensor(
            np.random.choice(
                [NuType.E.value, NuType.Mu.value, NuType.Tau.value], size=event_scale
            ),
            dtype=torch.int64,
            device=device,
        )
        osc_out = torch.tensor(
            np.random.choice(
                [NuType.E.value, NuType.Mu.value, NuType.Tau.value], size=event_scale
            ),
            dtype=torch.int64,
            device=device,
        )
        energies = torch.tensor(
            np.linspace(0.1, 10, event_scale), dtype=torch.float64, device=device
        )  # Example energy range

        oscillator = Oscillator(1300, 0.5, 3, 0)

        for i in range(n_iter):
            osc_mod = osc_pars.clone() * (
                1.0001 + (np.random.uniform(0, 1) / (n_iter * 100))
            )
            start_time = time.time()
            _ = oscillator.calc_probability(osc_mod, energies, osc_in, osc_out)
            end_time = time.time()
            times[i, j] = end_time - start_time

    print("Benchmark complete.")
    return times.cpu().numpy(), scales


def plot_scaling(times: np.ndarray, scales: np.ndarray, plot_file="osc_scaling.png"):
    mean_times = times.mean(axis=0)
    std_times = times.std(axis=0)

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
    plt.plot(scales, mean_times / scales * 1_000_000, label="Mean Time per Scale [μs]")
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
