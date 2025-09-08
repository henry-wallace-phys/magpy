"""
MCMC visualization module.
"""

import numpy as np
import matplotlib.pyplot as plt

class MCMCPlotter:
    """Basic MCMC plotting utilities."""
    
    def __init__(self):
        pass
    
    def plot_traces(self, chains, param_names):
        """Plot MCMC traces."""
        n_params = chains.shape[1]
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 2*n_params))
        
        for i in range(n_params):
            if n_params > 1:
                ax = axes[i]
            else:
                ax = axes
            ax.plot(chains[:, i])
            ax.set_ylabel(param_names[i])
        
        plt.tight_layout()
        return fig
