"""
Adaptive Metropolis MCMC sampler for neutrino oscillation parameter inference.

This module implements an adaptive Metropolis-Hastings algorithm with:
- Adaptive proposal covariance matrix
- Parameter bounds checking
- Automatic step size adaptation
- Diagnostic statistics
"""

import jax.numpy as jnp
from jax import random
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class MCMCResult:
    """Results from MCMC sampling."""
    samples: jnp.ndarray
    log_prob: jnp.ndarray
    accept_rate: float
    param_names: List[str]
    covariance_matrix: jnp.ndarray
    final_step_size: float


class AdaptiveMetropolis:
    """
    Adaptive Metropolis-Hastings sampler.
    
    Implements the adaptive Metropolis algorithm from Haario et al. (2001)
    with parameter bounds checking and automatic step size adaptation.
    """
    
    def __init__(
        self,
        log_prob_fn,
        n_params: int,
        param_names=None,
        initial_step_size: float = 0.1,
        adaptation_interval: int = 100,
        bounds=None
    ):
        """
        Initialize the adaptive Metropolis sampler.
        
        Args:
            log_prob_fn: Function that computes log probability
            n_params: Number of parameters
            param_names: Parameter names for diagnostics
            initial_step_size: Initial step size for proposals
            adaptation_interval: How often to adapt covariance matrix
            bounds: Parameter bounds as list of (min, max) tuples
        """
        self.log_prob_fn = log_prob_fn
        self.n_params = n_params
        self.param_names = param_names or [f"param_{i}" for i in range(n_params)]
        self.step_size = initial_step_size
        self.adaptation_interval = adaptation_interval
        self.bounds = bounds
        
        # Adaptive parameters
        self.epsilon = 1e-8  # Small regularization
        self.adaptation_start = max(100, 2 * n_params)  # When to start adapting
        
        # Initialize covariance matrix as identity
        self.cov_matrix = jnp.eye(n_params) * (initial_step_size ** 2)
        
        # Target acceptance rate
        self.target_accept_rate = 0.234  # Optimal for high-dimensional problems
        
    def check_bounds(self, params: jnp.ndarray) -> bool:
        """Check if parameters are within bounds."""
        if self.bounds is None:
            return True
            
        for i, (low, high) in enumerate(self.bounds):
            if params[i] < low or params[i] > high:
                return False
        return True
        
    def propose_step(self, current_params: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """Generate proposal step using current covariance matrix."""
        # Generate random step from multivariate normal
        proposal_step = random.multivariate_normal(
            key, 
            mean=jnp.zeros(self.n_params),
            cov=self.cov_matrix
        )
        
        return current_params + proposal_step
        
    def adapt_covariance(self, samples: jnp.ndarray, iteration: int):
        """Adapt the covariance matrix based on sample history."""
        if iteration < self.adaptation_start:
            return
            
        # Compute empirical covariance
        sample_mean = jnp.mean(samples, axis=0)
        centered_samples = samples - sample_mean
        empirical_cov = jnp.cov(centered_samples.T)
        
        # Adaptive scaling factor (Haario et al. 2001)
        s_d = 2.38**2 / self.n_params
        
        # Update covariance matrix with regularization
        self.cov_matrix = s_d * (empirical_cov + self.epsilon * jnp.eye(self.n_params))
        
    def adapt_step_size(self, accept_rate: float, iteration: int):
        """Adapt step size based on acceptance rate."""
        if iteration < 100:  # Don't adapt too early
            return
            
        # Simple step size adaptation
        adaptation_rate = min(0.1, 1.0 / iteration**0.6)
        
        if accept_rate > self.target_accept_rate:
            self.step_size *= (1 + adaptation_rate)
        else:
            self.step_size *= (1 - adaptation_rate)
            
        # Keep step size reasonable
        self.step_size = jnp.clip(self.step_size, 1e-8, 1.0)
        
        # Update covariance matrix scaling
        scale_factor = (self.step_size / 0.1) ** 2
        self.cov_matrix = self.cov_matrix * scale_factor
        
    def sample(
        self,
        initial_params,
        n_samples: int,
        n_warmup: int = 1000,
        key=None,
        thin: int = 1
    ):
        """
        Run adaptive Metropolis sampling.
        
        Args:
            initial_params: Starting parameter values
            n_samples: Number of samples to collect (after thinning)
            n_warmup: Number of warmup samples for adaptation
            key: Random key
            thin: Thinning factor (keep every thin-th sample)
            
        Returns:
            MCMCResult with samples and diagnostics
        """
        if key is None:
            key = random.PRNGKey(42)
        
        assert key is not None  # Type hint for mypy
            
        # Total iterations needed
        total_iterations = n_warmup + n_samples * thin
        
        # Storage for all samples (including warmup for adaptation)
        all_samples = []
        all_log_probs = []
        
        # Current state
        current_params = initial_params.copy()
        current_log_prob = self.log_prob_fn(current_params)
        
        # Check if initial parameters are valid
        if not jnp.isfinite(current_log_prob):
            raise ValueError(f"Initial log probability is not finite: {current_log_prob}")
            
        if not self.check_bounds(current_params):
            raise ValueError("Initial parameters are outside bounds")
        
        # Counters
        n_accepted = 0
        
        print(f"🎯 Starting Adaptive Metropolis: {n_samples} samples + {n_warmup} warmup")
        print(f"Initial log probability: {current_log_prob:.6f}")
        
        # Main sampling loop
        for i in tqdm(range(total_iterations), desc="MCMC sampling"):
            key, subkey = random.split(key)
            
            # Generate proposal
            proposed_params = self.propose_step(current_params, subkey)
            
            # Check bounds
            if not self.check_bounds(proposed_params):
                # Reject proposal - stays at current state
                pass
            else:
                # Evaluate log probability at proposal
                try:
                    proposed_log_prob = self.log_prob_fn(proposed_params)
                    
                    if jnp.isfinite(proposed_log_prob):
                        # Metropolis acceptance criterion
                        log_alpha = proposed_log_prob - current_log_prob
                        alpha = jnp.exp(jnp.minimum(0.0, log_alpha))
                        
                        # Accept or reject
                        key, subkey = random.split(key)
                        if random.uniform(subkey) < alpha:
                            current_params = proposed_params
                            current_log_prob = proposed_log_prob
                            n_accepted += 1
                except Exception:
                    # If evaluation fails, reject proposal
                    pass
            
            # Store sample
            all_samples.append(current_params.copy())
            all_log_probs.append(current_log_prob)
            
            # Adaptation during warmup
            if i < n_warmup and (i + 1) % self.adaptation_interval == 0:
                # Adapt covariance matrix
                recent_samples = jnp.array(all_samples[max(0, i - 500):i + 1])
                self.adapt_covariance(recent_samples, i + 1)
                
                # Adapt step size
                recent_accept_rate = n_accepted / (i + 1)
                self.adapt_step_size(recent_accept_rate, i + 1)
        
        # Extract final samples (post-warmup, thinned)
        warmup_samples = all_samples[n_warmup:]
        warmup_log_probs = all_log_probs[n_warmup:]
        
        # Apply thinning
        final_samples = warmup_samples[::thin][:n_samples]
        final_log_probs = warmup_log_probs[::thin][:n_samples]
        
        # Convert to arrays
        samples = jnp.array(final_samples)
        log_probs = jnp.array(final_log_probs)
        
        # Calculate final acceptance rate
        final_accept_rate = n_accepted / total_iterations
        
        print(f"✅ Sampling completed!")
        print(f"   Acceptance rate: {final_accept_rate:.3f}")
        print(f"   Final step size: {self.step_size:.6f}")
        
        return MCMCResult(
            samples=samples,
            log_prob=log_probs,
            accept_rate=final_accept_rate,
            param_names=self.param_names,
            covariance_matrix=self.cov_matrix,
            final_step_size=float(self.step_size)
        )
