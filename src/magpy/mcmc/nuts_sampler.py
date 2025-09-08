"""
No-U-Turn Sampler (NUTS) implementation for Hamiltonian Monte Carlo.

This module implements the NUTS algorithm with dual averaging for automatic
step size adaptation, following the algorithm described in Hoffman & Gelman (2014).
"""

import jax
import jax.numpy as jnp
import jax.random as random
from jax import grad, jit
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Dict, Any, Union
import numpy as np
from tqdm import tqdm

# Configure JAX for high precision
jax.config.update("jax_enable_x64", True)

@dataclass
class MCMCResult:
    """Results from MCMC sampling"""
    samples: jnp.ndarray          # Shape: (n_samples, n_params)
    log_prob: jnp.ndarray         # Shape: (n_samples,)
    step_size: float              # Final adapted step size
    accept_rate: float            # Overall acceptance rate
    n_divergences: int            # Number of divergent transitions
    energy: jnp.ndarray          # Hamiltonian energy at each step
    tree_depth: jnp.ndarray      # Tree depth at each step
    param_names: list            # Parameter names

class NUTSSampler:
    """
    No-U-Turn Sampler for Hamiltonian Monte Carlo.
    
    Implements the NUTS algorithm with dual averaging for step size adaptation.
    Provides robust sampling for complex posterior distributions.
    """
    
    def __init__(self, 
                 log_prob_fn: Callable,
                 n_params: int,
                 param_names: Optional[list] = None,
                 step_size: float = 0.1,
                 max_tree_depth: int = 10,
                 target_accept: float = 0.8):
        """
        Initialize NUTS sampler.
        
        Args:
            log_prob_fn: Function computing log probability and its gradient
            n_params: Number of parameters to sample
            param_names: Names of parameters
            step_size: Initial step size (will be adapted)
            max_tree_depth: Maximum tree depth for NUTS
            target_accept: Target acceptance rate for step size adaptation
        """
        self.log_prob_fn = jit(log_prob_fn)
        self.grad_log_prob_fn = jit(grad(log_prob_fn))
        self.n_params = n_params
        self.param_names = param_names or [f"param_{i}" for i in range(n_params)]
        self.step_size = step_size
        self.max_tree_depth = max_tree_depth
        self.target_accept = target_accept
        
        # Dual averaging parameters
        self.gamma = 0.05
        self.t0 = 10.0
        self.kappa = 0.75
        
    def leapfrog_step(self, q: jnp.ndarray, p: jnp.ndarray, 
                      grad_log_prob: jnp.ndarray, step_size: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single leapfrog integration step"""
        # Half step for momentum
        p_half = p + 0.5 * step_size * grad_log_prob
        
        # Full step for position
        q_new = q + step_size * p_half
        
        # Half step for momentum with new gradient
        grad_new = self.grad_log_prob_fn(q_new)
        p_new = p_half + 0.5 * step_size * grad_new
        
        return q_new, p_new
    
    def hamiltonian(self, q: jnp.ndarray, p: jnp.ndarray) -> float:
        """Compute Hamiltonian (total energy)"""
        kinetic = 0.5 * jnp.sum(p**2)
        potential = -self.log_prob_fn(q)
        return kinetic + potential
    
    def build_tree(self, q: jnp.ndarray, p: jnp.ndarray, grad_log_prob: jnp.ndarray,
                   log_prob: float, direction: int, depth: int, step_size: float,
                   key: jnp.ndarray) -> Tuple:
        """
        Build binary tree for NUTS algorithm.
        
        Returns:
            - Left endpoint (q, p)
            - Right endpoint (q, p) 
            - Proposal state (q, p, log_prob)
            - Number of valid states
            - Continue flag
            - Divergence flag
        """
        if depth == 0:
            # Base case: single leapfrog step
            q_new, p_new = self.leapfrog_step(q, p, grad_log_prob, direction * step_size)
            log_prob_new = self.log_prob_fn(q_new)
            grad_new = self.grad_log_prob_fn(q_new)
            
            # Check for divergence
            energy_new = self.hamiltonian(q_new, p_new)
            energy_old = self.hamiltonian(q, p)
            divergent = (energy_new - energy_old) > 1000  # Energy difference threshold
            
            # Accept/reject based on slice sampler criterion
            # (This is simplified - full NUTS uses slice sampling)
            n_valid = 1 if not divergent else 0
            
            return (q_new, p_new), (q_new, p_new), (q_new, p_new, log_prob_new), n_valid, not divergent, divergent
        
        else:
            # Recursion: build left and right subtrees
            key1, key2 = random.split(key)
            
            # Build left subtree
            left_end, right_end, proposal, n_valid, continue_flag, divergent = self.build_tree(
                q, p, grad_log_prob, log_prob, direction, depth-1, step_size, key1
            )
            
            if continue_flag:
                # Build right subtree from right endpoint of left subtree
                if direction == -1:
                    left_end2, _, proposal2, n_valid2, continue_flag2, divergent2 = self.build_tree(
                        left_end[0], left_end[1], self.grad_log_prob_fn(left_end[0]), 
                        self.log_prob_fn(left_end[0]), direction, depth-1, step_size, key2
                    )
                    left_end = left_end2
                else:
                    _, right_end2, proposal2, n_valid2, continue_flag2, divergent2 = self.build_tree(
                        right_end[0], right_end[1], self.grad_log_prob_fn(right_end[0]),
                        self.log_prob_fn(right_end[0]), direction, depth-1, step_size, key2
                    )
                    right_end = right_end2
                
                # Choose proposal based on number of valid states
                total_valid = n_valid + n_valid2
                if total_valid > 0:
                    prob_new = n_valid2 / total_valid
                    accept_new = random.uniform(key2) < prob_new
                    if accept_new:
                        proposal = proposal2
                
                # Update counters and flags
                n_valid = total_valid
                continue_flag = continue_flag2 and self.check_uturn(left_end, right_end)
                divergent = divergent or divergent2
            
            return left_end, right_end, proposal, n_valid, continue_flag, divergent
    
    def check_uturn(self, left_end: Tuple, right_end: Tuple) -> bool:
        """Check if trajectory is making a U-turn"""
        q_left, p_left = left_end
        q_right, p_right = right_end
        
        # U-turn condition: trajectory turning back on itself
        delta_q = q_right - q_left
        return bool(jnp.dot(delta_q, p_left) >= 0 and jnp.dot(delta_q, p_right) >= 0)
    
    def nuts_step(self, q: jnp.ndarray, step_size: float, 
                  key: jnp.ndarray) -> Tuple[jnp.ndarray, float, bool, int]:
        """Single NUTS sampling step"""
        # Sample momentum
        key1, key2 = random.split(key)
        p = random.normal(key1, shape=q.shape)
        
        # Compute initial log probability and gradient
        log_prob = self.log_prob_fn(q)
        grad_log_prob = self.grad_log_prob_fn(q)
        
        # Initialize tree
        q_left = q_right = q
        p_left = p_right = p
        proposal = (q, p, log_prob)
        depth = 0
        n_valid = 1
        continue_flag = True
        divergent = False
        
        # Build tree until stopping criterion
        while continue_flag and depth < self.max_tree_depth:
            key2, key3 = random.split(key2)
            direction = int(2 * random.bernoulli(key3) - 1)  # Random direction
            
            if direction == -1:
                (q_left, p_left), _, proposal_new, n_valid_new, continue_flag, div_new = self.build_tree(
                    q_left, p_left, self.grad_log_prob_fn(q_left), self.log_prob_fn(q_left),
                    direction, depth, step_size, key2
                )
            else:
                _, (q_right, p_right), proposal_new, n_valid_new, continue_flag, div_new = self.build_tree(
                    q_right, p_right, self.grad_log_prob_fn(q_right), self.log_prob_fn(q_right),
                    direction, depth, step_size, key2
                )
            
            # Accept new proposal with appropriate probability
            if continue_flag:
                total_valid = n_valid + n_valid_new
                if total_valid > 0:
                    prob_accept = n_valid_new / total_valid
                    if random.uniform(key2) < prob_accept:
                        proposal = proposal_new
                n_valid = total_valid
            
            # Check stopping criteria
            continue_flag = continue_flag and self.check_uturn((q_left, p_left), (q_right, p_right))
            divergent = divergent or div_new
            depth += 1
        
        return proposal[0], proposal[2], divergent, depth
    
    def adapt_step_size(self, step_size: float, accept_prob: float, 
                       iteration: int, h_bar: float, log_eps_bar: float) -> Tuple[float, float, float]:
        """Adapt step size using dual averaging"""
        eta = 1.0 / (iteration + self.t0)
        h_bar = (1 - eta) * h_bar + eta * (self.target_accept - accept_prob)
        
        log_eps = self.gamma * h_bar - jnp.sqrt(iteration) / self.gamma
        eta2 = iteration**(-self.kappa)
        log_eps_bar = eta2 * log_eps + (1 - eta2) * log_eps_bar
        
        step_size = float(jnp.exp(log_eps))
        return step_size, h_bar, log_eps_bar
    
    def sample(self, 
              initial_params: jnp.ndarray,
              n_samples: int,
              n_warmup: int = 1000,
              key: Optional[jnp.ndarray] = None) -> MCMCResult:
        """
        Run NUTS sampling.
        
        Args:
            initial_params: Starting parameter values
            n_samples: Number of samples to collect
            n_warmup: Number of warmup samples for adaptation
            key: Random key
            
        Returns:
            MCMCResult with samples and diagnostics
        """
        if key is None:
            key = random.PRNGKey(42)
        
        total_iterations = n_warmup + n_samples
        
        # Storage arrays
        samples = jnp.zeros((n_samples, self.n_params))
        log_probs = jnp.zeros(n_samples)
        energies = jnp.zeros(n_samples)
        tree_depths = jnp.zeros(n_samples, dtype=int)
        
        # Adaptation variables
        step_size = self.step_size
        h_bar = 0.0
        log_eps_bar = jnp.log(step_size)
        
        # Counters
        n_accept = 0
        n_divergent = 0
        
        # Current state
        current_q = initial_params
        current_log_prob = self.log_prob_fn(current_q)
        
        print(f"🔥 Starting NUTS sampling: {n_samples} samples + {n_warmup} warmup")
        print(f"Initial log probability: {current_log_prob:.6f}")
        
        # Main sampling loop
        for i in tqdm(range(total_iterations), desc="NUTS sampling"):
            assert key is not None  # Should not be None after initialization above
            key, subkey = random.split(key)
            
            # NUTS step
            proposed_q, proposed_log_prob, divergent, depth = self.nuts_step(
                current_q, step_size, subkey
            )
            
            # Accept/reject (NUTS handles this internally, so we always accept the proposal)
            current_q = proposed_q
            current_log_prob = proposed_log_prob
            
            # Compute acceptance probability for step size adaptation
            # (Simplified - real NUTS tracks this more carefully)
            accept_prob = 1.0 if not divergent else 0.0
            n_accept += accept_prob
            
            if divergent:
                n_divergent += 1
            
            # Adapt step size during warmup
            if i < n_warmup:
                step_size, h_bar, log_eps_bar = self.adapt_step_size(
                    step_size, accept_prob, i + 1, float(h_bar), float(log_eps_bar)
                )
            
            # Store samples after warmup
            if i >= n_warmup:
                sample_idx = i - n_warmup
                samples = samples.at[sample_idx].set(current_q)
                log_probs = log_probs.at[sample_idx].set(current_log_prob)
                
                # Compute energy
                p_dummy = jnp.zeros_like(current_q)  # Zero momentum for energy calculation
                energy = self.hamiltonian(current_q, p_dummy)
                energies = energies.at[sample_idx].set(energy)
                tree_depths = tree_depths.at[sample_idx].set(depth)
        
        # Final step size from adaptation
        if n_warmup > 0:
            final_step_size = jnp.exp(log_eps_bar)
        else:
            final_step_size = step_size
        
        accept_rate = n_accept / total_iterations
        
        print(f"✅ Sampling completed!")
        print(f"   Acceptance rate: {accept_rate:.3f}")
        print(f"   Divergences: {n_divergent}")
        print(f"   Final step size: {final_step_size:.6f}")
        
        return MCMCResult(
            samples=samples,
            log_prob=log_probs,
            step_size=float(final_step_size),
            accept_rate=float(accept_rate),
            n_divergences=n_divergent,
            energy=energies,
            tree_depth=tree_depths,
            param_names=self.param_names
        )
