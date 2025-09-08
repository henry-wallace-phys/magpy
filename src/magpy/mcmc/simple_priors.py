"""
Simplified priors for testing.
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

class SimplePriorCollection:
    """Simplified prior collection for testing."""
    
    def __init__(self):
        # Standard oscillation parameter priors
        self.param_names = ['theta12', 'theta13', 'theta23', 'deltacp', 'dm21', 'dm32']
        self.nominal_values = jnp.array([0.3, 0.02, 0.55, 0.7 * jnp.pi, 7.5e-5, 2.5e-3])
        self.errors = jnp.array([0.02, 0.002, 0.05, 0.1 * jnp.pi, 0.5e-5, 0.1e-3])
        
    def get_nominal_values(self):
        return self.nominal_values
    
    def log_prob(self, params):
        """Simple Gaussian priors for all parameters."""
        log_prob = 0.0
        for i in range(len(params)):
            if i == 3:  # deltacp - flat prior
                log_prob += -jnp.log(2 * jnp.pi)  # Flat on [0, 2π]
            else:  # Gaussian priors
                log_prob += norm.logpdf(params[i], loc=self.nominal_values[i], scale=self.errors[i])
        
        return log_prob
