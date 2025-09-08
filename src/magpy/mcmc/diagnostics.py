"""
MCMC diagnostics module.
"""

import jax.numpy as jnp
from typing import Dict, Any

class MCMCDiagnostics:
    """Basic MCMC diagnostics."""
    
    def __init__(self, acceptance_rate: float = 0.0, n_divergences: int = 0):
        self.acceptance_rate = acceptance_rate
        self.n_divergences = n_divergences
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'acceptance_rate': self.acceptance_rate,
            'n_divergences': self.n_divergences
        }
