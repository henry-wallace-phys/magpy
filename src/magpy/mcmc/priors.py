"""
Prior distributions for MCMC sampling.

This module implements various prior distributions including Gaussian and uniform (flat) priors
with proper range checking and log-probability evaluation.
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Dict, Any, Union
from abc import ABC, abstractmethod
import yaml
from pathlib import Path


class Prior(ABC):
    """Abstract base class for prior distributions."""
    
    @abstractmethod
    def log_prob(self, x: Union[float, Array]) -> Array:
        """Compute log probability of value x under this prior."""
        pass
    
    @abstractmethod
    def in_range(self, x: Union[float, Array]) -> bool:
        """Check if value x is within the valid range for this prior."""
        pass
    
    @abstractmethod
    def sample(self, key: Array) -> Array:
        """Sample from this prior distribution."""
        pass


class GaussianPrior(Prior):
    """Gaussian (normal) prior distribution."""
    
    def __init__(self, mean: float, std: float, range_min: float = -jnp.inf, range_max: float = jnp.inf):
        self.mean = mean
        self.std = std
        self.range_min = range_min
        self.range_max = range_max
        
    def log_prob(self, x: Union[float, Array]) -> Array:
        """Compute log probability under Gaussian prior."""
        if not self.in_range(x):
            return jnp.array(-jnp.inf)
        
        # Standard Gaussian log probability
        return -0.5 * jnp.log(2 * jnp.pi * self.std**2) - 0.5 * ((x - self.mean) / self.std)**2
    
    def in_range(self, x: Union[float, Array]) -> bool:
        """Check if value is within specified range."""
        return bool(jnp.all((self.range_min <= x) & (x <= self.range_max)))
    
    def sample(self, key: Array) -> Array:
        """Sample from Gaussian prior (with rejection if outside range)."""
        sample = jax.random.normal(key) * self.std + self.mean
        
        # Simple rejection sampling for range constraints
        # Note: In practice, might want truncated normal for efficiency
        if self.in_range(sample):
            return sample
        else:
            # Clip to range boundaries
            return jnp.clip(sample, self.range_min, self.range_max)


class FlatPrior(Prior):
    """Uniform (flat) prior distribution."""
    
    def __init__(self, range_min: float, range_max: float):
        if range_min >= range_max:
            raise ValueError("range_min must be less than range_max for flat prior")
        
        self.range_min = range_min
        self.range_max = range_max
        self.log_prob_value = -jnp.log(range_max - range_min)  # log(1/(b-a))
        
    def log_prob(self, x: Union[float, Array]) -> Array:
        """Compute log probability under flat prior."""
        if self.in_range(x):
            return jnp.array(self.log_prob_value)
        else:
            return jnp.array(-jnp.inf)
    
    def in_range(self, x: Union[float, Array]) -> bool:
        """Check if value is within prior range."""
        return bool(jnp.all((self.range_min <= x) & (x <= self.range_max)))
    
    def sample(self, key: Array) -> Array:
        """Sample uniformly from prior range."""
        return jax.random.uniform(key, minval=self.range_min, maxval=self.range_max)


class PriorCollection:
    """Collection of priors for multiple parameters."""
    
    def __init__(self, priors: Dict[str, Prior]):
        self.priors = priors
        self.param_names = list(priors.keys())
        
    def log_prob(self, params: jnp.ndarray) -> Array:
        """Compute total log prior probability for parameter vector."""
        if len(params) != len(self.param_names):
            raise ValueError(f"Expected {len(self.param_names)} parameters, got {len(params)}")
        
        total_log_prob = 0.0
        for i, param_name in enumerate(self.param_names):
            log_prob_i = self.priors[param_name].log_prob(params[i])
            if jnp.isinf(log_prob_i) and log_prob_i < 0:
                return jnp.array(-jnp.inf)  # Short-circuit if any parameter is out of bounds
            total_log_prob += log_prob_i
            
        return jnp.array(total_log_prob)
    
    def in_range(self, params: jnp.ndarray) -> bool:
        """Check if all parameters are within their prior ranges."""
        if len(params) != len(self.param_names):
            return False
            
        for i, param_name in enumerate(self.param_names):
            if not self.priors[param_name].in_range(params[i]):
                return False
        return True
    
    def sample(self, key: Array) -> jnp.ndarray:
        """Sample from all priors."""
        keys = jax.random.split(key, len(self.param_names))
        samples = []
        
        for i, param_name in enumerate(self.param_names):
            sample = self.priors[param_name].sample(keys[i])
            samples.append(sample)
            
        return jnp.array(samples)


def load_priors_from_yaml(yaml_path: Union[str, Path]) -> PriorCollection:
    """Load prior specifications from YAML file.
    
    Expected YAML format:
    ```yaml
    priors:
      theta_12:
        type: "gaussian"  # or "flat"
        nominal: 0.3
        error: 0.02
        range: [0.0, 1.0]  # optional, defaults to [-inf, inf]
      
      delta_m2_21:
        type: "flat"
        range: [6.0e-5, 9.0e-5]
        # nominal and error not used for flat priors
    ```
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'priors' not in config:
        raise ValueError("YAML file must contain 'priors' section")
    
    priors = {}
    
    for param_name, prior_config in config['priors'].items():
        prior_type = prior_config.get('type', 'gaussian').lower()
        
        # Get range (optional for Gaussian, required for flat)
        range_spec = prior_config.get('range', [-jnp.inf, jnp.inf])
        range_min, range_max = range_spec[0], range_spec[1]
        
        if prior_type == 'gaussian':
            nominal = prior_config.get('nominal')
            error = prior_config.get('error')
            
            if nominal is None or error is None:
                raise ValueError(f"Gaussian prior for {param_name} requires 'nominal' and 'error'")
            
            priors[param_name] = GaussianPrior(
                mean=nominal, 
                std=error, 
                range_min=range_min, 
                range_max=range_max
            )
            
        elif prior_type == 'flat':
            if jnp.isinf(range_min) or jnp.isinf(range_max):
                raise ValueError(f"Flat prior for {param_name} requires finite 'range'")
            
            priors[param_name] = FlatPrior(range_min=range_min, range_max=range_max)
            
        else:
            raise ValueError(f"Unknown prior type '{prior_type}' for parameter {param_name}")
    
    return PriorCollection(priors)


# JIT compile for performance
@jax.jit
def evaluate_log_priors(params: jnp.ndarray, prior_log_probs_func) -> Array:
    """JIT-compiled prior evaluation."""
    return prior_log_probs_func(params)


@jax.jit 
def check_param_ranges(params: jnp.ndarray, range_mins: jnp.ndarray, range_maxs: jnp.ndarray) -> Array:
    """JIT-compiled range checking."""
    return jnp.all((params >= range_mins) & (params <= range_maxs))
