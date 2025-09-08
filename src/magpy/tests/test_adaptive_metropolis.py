"""
Test adaptive Metropolis MCMC sampler.
"""

import pytest
import jax.numpy as jnp
import jax
from jax import random
import numpy as np

from magpy.mcmc.adaptive_metropolis import AdaptiveMetropolis, MCMCResult
from magpy.mcmc.likelihood import create_likelihood_from_config
from magpy.mcmc.priors_yaml import PriorCollection, GaussianPrior, FlatPrior


class TestAdaptiveMetropolis:
    """Test the adaptive Metropolis sampler."""
    
    def test_simple_gaussian(self):
        """Test sampling from a simple Gaussian distribution."""
        # Define a simple 2D Gaussian log probability
        def log_prob(params):
            # Simple 2D Gaussian centered at [1, 2] with variance [0.5, 1.0]
            x, y = params[0], params[1]
            return -0.5 * ((x - 1.0)**2 / 0.5 + (y - 2.0)**2 / 1.0)
        
        sampler = AdaptiveMetropolis(
            log_prob_fn=log_prob,
            n_params=2,
            param_names=['x', 'y'],
            initial_step_size=0.1
        )
        
        # Sample
        key = random.PRNGKey(42)
        initial_params = jnp.array([0.0, 0.0])
        
        result = sampler.sample(
            initial_params=initial_params,
            n_samples=1000,
            n_warmup=500,
            key=key
        )
        
        # Check results
        assert isinstance(result, MCMCResult)
        assert result.samples.shape == (1000, 2)
        assert result.accept_rate > 0.1  # Should have reasonable acceptance
        assert result.accept_rate < 0.8  # But not too high
        
        # Check convergence to correct mean (approximately)
        sample_mean = jnp.mean(result.samples, axis=0)
        assert abs(sample_mean[0] - 1.0) < 0.2  # Within reasonable tolerance
        assert abs(sample_mean[1] - 2.0) < 0.3
    
    def test_bounds_checking(self):
        """Test that bounds are properly enforced."""
        def log_prob(params):
            return -0.5 * jnp.sum(params**2)  # Standard Gaussian
        
        # Define bounds that should constrain the samples
        bounds = [(0.0, 2.0), (-1.0, 1.0)]
        
        sampler = AdaptiveMetropolis(
            log_prob_fn=log_prob,
            n_params=2,
            bounds=bounds,
            initial_step_size=0.1
        )
        
        key = random.PRNGKey(123)
        initial_params = jnp.array([1.0, 0.0])  # Within bounds
        
        result = sampler.sample(
            initial_params=initial_params,
            n_samples=500,
            n_warmup=200,
            key=key
        )
        
        # Check that all samples are within bounds
        for i, (low, high) in enumerate(bounds):
            assert jnp.all(result.samples[:, i] >= low)
            assert jnp.all(result.samples[:, i] <= high)
    
    def test_invalid_initial_params(self):
        """Test that invalid initial parameters are caught."""
        def log_prob(params):
            # Function that returns NaN for negative values
            if params[0] < 0:
                return jnp.nan
            return -0.5 * jnp.sum(params**2)
        
        sampler = AdaptiveMetropolis(
            log_prob_fn=log_prob,
            n_params=2,
            initial_step_size=0.1
        )
        
        key = random.PRNGKey(42)
        invalid_params = jnp.array([-1.0, 0.0])  # Will cause NaN
        
        with pytest.raises(ValueError, match="Initial log probability is not finite"):
            sampler.sample(
                initial_params=invalid_params,
                n_samples=100,
                n_warmup=50,
                key=key
            )
    
    def test_oscillation_likelihood_integration(self):
        """Test integration with oscillation likelihood."""
        # Create a simple prior collection
        priors = [
            GaussianPrior("theta12", 0.3, 0.02, [0.0, 1.0]),
            GaussianPrior("theta13", 0.02, 0.005, [0.0, 1.0]),
            GaussianPrior("theta23", 0.55, 0.05, [0.0, 1.0]),
            FlatPrior("deltacp", 2.2, [0.0, 2*np.pi]),
            GaussianPrior("dm21", 7.5e-5, 1e-5, [0.0, 1e-3]),
            GaussianPrior("dm32", 2.5e-3, 5e-4, [-1e-2, 1e-2])
        ]
        
        prior_collection = PriorCollection(priors)
        likelihood = create_likelihood_from_config({}, "", "")
        
        def log_posterior(params):
            log_likelihood = likelihood(params)
            log_prior = prior_collection.log_prob(params)
            return log_likelihood + log_prior
        
        # Parameter bounds for oscillation parameters
        bounds = [
            (0.01, 1.0),      # theta12
            (0.001, 0.1),     # theta13  
            (0.01, 1.0),      # theta23
            (0.0, 2*np.pi),   # deltacp
            (1e-6, 1e-3),     # dm21
            (-1e-2, 1e-2)     # dm32
        ]
        
        sampler = AdaptiveMetropolis(
            log_prob_fn=log_posterior,
            n_params=6,
            param_names=['theta12', 'theta13', 'theta23', 'deltacp', 'dm21', 'dm32'],
            bounds=bounds,
            initial_step_size=1e-4  # Small step size for sensitive parameters
        )
        
        # Start at nominal values
        initial_params = prior_collection.get_nominal_values()
        
        key = random.PRNGKey(42)
        
        # Run a short test
        result = sampler.sample(
            initial_params=initial_params,
            n_samples=50,
            n_warmup=25,
            key=key
        )
        
        # Basic checks
        assert result.samples.shape == (50, 6)
        assert result.accept_rate > 0.05  # Some reasonable acceptance
        assert jnp.all(jnp.isfinite(result.samples))  # No NaN/inf values
        assert jnp.all(jnp.isfinite(result.log_prob))  # No NaN/inf log probs
        
        # Check bounds are respected
        for i, (low, high) in enumerate(bounds):
            assert jnp.all(result.samples[:, i] >= low)
            assert jnp.all(result.samples[:, i] <= high)


if __name__ == "__main__":
    # Run a simple test
    test = TestAdaptiveMetropolis()
    test.test_simple_gaussian()
    print("✅ Simple Gaussian test passed!")
    
    test.test_bounds_checking()
    print("✅ Bounds checking test passed!")
    
    print("🎯 All basic tests passed!")
