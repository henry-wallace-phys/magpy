import time

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from rich import print
import torch
from tqdm import tqdm

from magpy.oscillator.oscillator import Oscillator, NuType
from magpy.utils.device_manager import DeviceManager

def benchmark_osc(n_iter: int=50, n_scales: int=100000, points: int=100):
    
    device = DeviceManager().get_device()
    
    oscillator = Oscillator(1300, 0.5, 3, 0)
    osc_pars = torch.tensor([0.3, 0.02, 0.55, 0.7 * np.pi, 7.5e-5, 2.5e-3], dtype=torch.float64, device=device)
    
    scales = np.arange(1, n_scales, n_scales // points)
    
    times = torch.zeros((n_iter, len(scales)), dtype=torch.float64, device=device)

    print("Starting osc benchmark...")
    for j, event_scale in tqdm(enumerate(scales), desc="Event scale", total=len(scales)):
        osc_in = torch.tensor(np.random.choice([NuType.E.value, NuType.Mu.value, NuType.Tau.value], size=event_scale), dtype=torch.int64, device=device)
        osc_out = torch.tensor(np.random.choice([NuType.E.value, NuType.Mu.value, NuType.Tau.value], size=event_scale), dtype=torch.int64, device=device)
        energies = torch.tensor(np.linspace(0.1, 10, event_scale), dtype=torch.float64, device=device)  # Example energy range
        _ = oscillator.calc_probability(osc_pars, energies, osc_in, osc_out)
        
        
        for i in range(n_iter):
            e_mod = energies.clone()*(1.0001+(i/n_iter*100))
            start_time = time.time()
            _=oscillator.calc_probability(osc_pars, e_mod, osc_in, osc_out)
            end_time = time.time()
            times[i, j] = end_time - start_time

    print("Benchmark complete.")
    return times.cpu().numpy(), scales


def plot_scaling(times: np.ndarray, scales: np.ndarray, plot_file='osc_scaling.pdf'):
    mean_times = times.mean(axis=0)
    std_times = times.std(axis=0)

    plt.figure()
    plt.errorbar(scales[1:], mean_times[1:]*1000, yerr=std_times[1:]*1000, fmt='o-')
    plt.xlabel('Event Scale')
    plt.ylabel('Time (ms)')
    plt.title('Oscillator Benchmark Scaling')
    plt.grid()
    plt.savefig(plot_file)
    plt.close()

def plot_average_time_per_event(times: np.ndarray, scales: np.ndarray, plot_file='osc_average_time_event.pdf'):

    times_avged = np.zeros(len(scales)*len(times), dtype=np.float64)
    for i in range(len(scales)):
        times_avged[i*len(times):(i+1)*len(times)] = times[1:, i]/scales[i]
    
    plt.figure()
    plt.hist(times_avged, bins=50, alpha=0.7)
    plt.xlabel('Average Time (s)')
    plt.ylabel('Frequency')
    plt.title('Average Oscillator Calculation Time Distribution')
    plt.grid()
    plt.savefig(plot_file)
    plt.close()
    