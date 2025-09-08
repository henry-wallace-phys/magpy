"""
Prior distributions for MCMC sampling.

This module provides support for Gaussian and flat priors with parameter
bounds checking, configurable from YAML files.
"""

import jax.numpy as jnp
from jax.scipy.stats import norm
from typing import Dict, List, Any, Union
import yaml
from pathlib import Path


class Prior:
    """Base class for priors."""
    
    def __init__(self, param_name: str, nominal: float, range_bounds: List[float]):
        self.param_name = param_name
        self.nominal = nominal
        self.range_bounds = range_bounds
    
    def log_prob(self, value: float) -> float:
        """Calculate log probability."""
        raise NotImplementedError
    
    def is_in_bounds(self, value: float) -> bool:
        """Check if value is within parameter bounds."""
        return self.range_bounds[0] <= value <= self.range_bounds[1]


class GaussianPrior(Prior):
    """Gaussian prior with optional bounds."""
    
    def __init__(self, param_name: str, nominal: float, error: float, range_bounds: List[float]):
        super().__init__(param_name, nominal, range_bounds)
        self.error = error
    
    def log_prob(self, value):
        """Log probability density."""
        # Check bounds
        in_bounds = (value >= self.range_bounds[0]) & (value <= self.range_bounds[1])
        
        # Gaussian log probability
        gaussian_logp = norm.logpdf(value, loc=self.nominal, scale=self.error)
        
        return jnp.where(in_bounds, gaussian_logp, -jnp.inf)


class FlatPrior(Prior):
    """Flat (uniform) prior within bounds."""
    
    def __init__(self, param_name: str, nominal: float, range_bounds: List[float]):
        super().__init__(param_name, nominal, range_bounds)
        self.width = range_bounds[1] - range_bounds[0]
    
    def log_prob(self, value):
        """Calculate log probability for flat prior."""
        # JAX-compatible bounds checking
        in_bounds = (value >= self.range_bounds[0]) & (value <= self.range_bounds[1])
        
        # Uniform log probability (constant within bounds)
        uniform_logp = -jnp.log(self.width)
        
        # Return -inf for out of bounds, uniform_logp otherwise
        return jnp.where(in_bounds, uniform_logp, -jnp.inf)


class PriorCollection:
    """Collection of priors for multiple parameters."""
    
    def __init__(self, priors: List[Prior]):
        self.priors = priors
        self.param_names = [prior.param_name for prior in priors]
        self.n_params = len(priors)
    
    def log_prob(self, params):
        """Calculate total log probability for all parameters."""
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        
        total_log_prob = 0.0
        for i, prior in enumerate(self.priors):
            log_prob = prior.log_prob(params[i])
            total_log_prob += log_prob
        
        return total_log_prob
    
    def are_all_in_bounds(self, params):
        """Check if all parameters are within their bounds using JAX operations."""
        # Start with True
        in_bounds = True
        
        for i, prior in enumerate(self.priors):
            param_in_bounds = (params[i] >= prior.range_bounds[0]) & (params[i] <= prior.range_bounds[1])
            in_bounds = in_bounds & param_in_bounds
            
        return in_bounds
    
    def get_nominal_values(self) -> jnp.ndarray:
        """Get nominal values for all parameters."""
        return jnp.array([prior.nominal for prior in self.priors])
    
    def get_bounds(self) -> List[List[float]]:
        """Get bounds for all parameters."""
        return [prior.range_bounds for prior in self.priors]


def load_priors_from_yaml(yaml_path: Union[str, Path]) -> PriorCollection:
    """
    Load priors from YAML configuration file.
    
    Args:
        yaml_path: Path to YAML file containing prior configuration
        
    Returns:
        PriorCollection with loaded priors
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'OscillationPriors' not in config:
        raise ValueError("YAML file must contain 'OscillationPriors' section")
    
    priors = []
    
    for prior_config in config['OscillationPriors']:
        param_name = prior_config['param_name']
        nominal = prior_config['nominal']
        range_bounds = prior_config['range']
        prior_type = prior_config['prior_type'].lower()
        
        if prior_type == 'gaussian':
            error = prior_config['error']
            prior = GaussianPrior(param_name, nominal, error, range_bounds)
        elif prior_type == 'flat':
            prior = FlatPrior(param_name, nominal, range_bounds)
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
        
        priors.append(prior)
    
    return PriorCollection(priors)


def create_default_priors() -> PriorCollection:
    """Create default oscillation parameter priors."""
    priors = [
        GaussianPrior("theta12", 0.3, 0.02, [0.1, 0.6]),
        GaussianPrior("theta13", 0.02, 0.002, [0.005, 0.05]),
        GaussianPrior("theta23", 0.55, 0.05, [0.3, 0.8]),
        FlatPrior("deltacp", 0.7 * jnp.pi, [0, 2 * jnp.pi]),
        GaussianPrior("dm21", 7.5e-5, 0.5e-5, [6.0e-5, 9.0e-5]),
        GaussianPrior("dm32", 2.5e-3, 0.1e-3, [2.0e-3, 3.0e-3])
    ]
    
    return PriorCollection(priors)
