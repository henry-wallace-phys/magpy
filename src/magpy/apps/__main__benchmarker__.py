from magpy.benchmarking.benchmark_osc import (
    benchmark_osc,
    plot_scaling,
    plot_average_time_per_event,
)
from magpy.benchmarking.benchmark_reweight import (
    benchmark_reweight,
    plot_hist,
)

import click
import numpy as np

from pathlib import Path


@click.command()
@click.option(
    "--n_iter", default=50, type=int, help="Number of iterations for the benchmark"
)
@click.option("--points", default=10, type=int, help="Number of points to sample")
@click.option(
    "--n_osc_scales",
    default=100000,
    type=int,
    help="Number of oscillation scales to sample",
)
@click.option(
    "--out_folder",
    default="benchmarks/results",
    type=str,
    help="Output folder for benchmark results",
)
@click.option(
    "--do_osc",
    default=True,
    type=bool,
    help="Whether to run oscillation benchmarks",
)
@click.option(
    "--do_reweight",
    default=True,
    type=bool,
    help="Whether to run reweighting benchmarks",
)
def main(
    n_iter: int = 500,
    points: int = 10,
    n_osc_scales: int = 100000,
    do_osc: bool = True,
    do_reweight: bool = True,
    out_folder: str = "benchmarks/results",
):
    if not Path(out_folder).exists():
        Path(out_folder).mkdir(parents=True, exist_ok=True)
    if do_osc:
        times, scales = benchmark_osc(n_iter=n_iter, points=points, n_scales=n_osc_scales)
        plot_scaling(times, scales, plot_file=f"{out_folder}/osc_scaling.png")
        plot_average_time_per_event(
            times, scales, plot_file=f"{out_folder}/osc_average_time_event.png"
        )

    if do_reweight:
        reweight_times = benchmark_reweight(n_iter=n_iter)
        plot_hist(reweight_times, plot_file=f"{out_folder}/reweight_histogram.png")

        reweight_times = np.array(reweight_times)

        reweight_times = reweight_times[100:]
        mean_time = np.mean(reweight_times)

        plot_hist(reweight_times[abs(reweight_times-mean_time) < np.std(reweight_times)], plot_file=f"{out_folder}/reweight_histogram_main.png")
