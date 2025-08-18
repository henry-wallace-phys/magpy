from magpy.benchmarking.benchmark_osc import benchmark_osc, plot_scaling, plot_average_time_per_event
import click

@click.command()
def main(n_iter: int = 50, points: int = 10, n_osc_scales: int = 100000):
    times, scales = benchmark_osc(n_iter=n_iter, points=points, n_scales=n_osc_scales)
    plot_scaling(times, scales)
    plot_average_time_per_event(times, scales)