# MAGPY MCMC Module
"""
Adaptive Metropolis MCMC for neutrino oscillation parameter estimation.

Implements:
- Adaptive Metropolis sampler with covariance adaptation
- Binned Poisson likelihood for neutrino event data
- Prior distributions with bounds checking
- Convergence diagnostics and visualization
"""

from .adaptive_metropolis import AdaptiveMetropolis, MCMCResult
from .likelihood import BinnedPoissonLikelihood, create_likelihood_from_config
from .priors_yaml import PriorCollection, GaussianPrior, FlatPrior, load_priors_from_yaml
from .diagnostics import MCMCDiagnostics
from .visualization import MCMCPlotter

__all__ = [
    'AdaptiveMetropolis',
    'MCMCResult', 
    'BinnedPoissonLikelihood',
    'create_likelihood_from_config',
    'PriorCollection',
    'GaussianPrior',
    'FlatPrior',
    'load_priors_from_yaml',
    'MCMCDiagnostics',
    'MCMCPlotter'
]
