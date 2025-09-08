"""
Poisson likelihood implementation for neutrino oscillation parameter inference.

This module implements a binned Poisson likelihood for MCMC sampling of 
oscillation parameters, with support for systematic uncertainties.
"""

import jax.numpy as jnp
from typing import Dict, Any, Optional
import numpy as np

from magpy.oscillator.oscillator import Oscillator
from magpy.objects.mc_event import MCEventMonolith, MCEventIndices, MCEvent
from magpy.utils.bin_handler import BinHandler


class BinnedPoissonLikelihood:
    """
    Binned Poisson likelihood for neutrino oscillation analysis.
    
    This class handles:
    - Binning of MC events in energy/angle space
    - Oscillation probability calculation
    - Event reweighting with systematic uncertainties
    - Poisson likelihood calculation
    """
    
    def __init__(
        self,
        data_monolith: MCEventMonolith,
        mc_monolith: MCEventMonolith,
        energy_bins: jnp.ndarray,
        oscillator_config: Dict[str, Any],
        use_systematics: bool = False
    ):
        """
        Initialize the likelihood calculator.
        
        Args:
            data_monolith: "Data" events (for now, same as MC at nominal)
            mc_monolith: Monte Carlo events for prediction
            energy_bins: Energy bin edges
            oscillator_config: Oscillator configuration (L, rho, Y_e, n_layers)
            use_systematics: Whether to include systematic uncertainties
        """
        self.data_monolith = data_monolith
        self.mc_monolith = mc_monolith
        self.energy_bins = energy_bins
        self.oscillator_config = oscillator_config
        self.use_systematics = use_systematics
        
        # Set up binning  
        self.bin_handler = BinHandler([energy_bins.tolist()])
        self.n_bins = len(energy_bins) - 1
        
        # Pre-compute data histogram (fixed)
        self.data_histogram = self._compute_data_histogram()
        
        # Extract MC event information for oscillation calculation
        self.mc_energies = self.mc_monolith.monolith[:, MCEventIndices.TRUE_NEUTRINO_ENERGY.value]
        self.mc_start_nu = self.mc_monolith.monolith[:, MCEventIndices.START_NU.value].astype(int)
        self.mc_end_nu = self.mc_monolith.monolith[:, MCEventIndices.END_NU.value].astype(int)
        self.mc_weights = self.mc_monolith.monolith[:, MCEventIndices.WEIGHT.value]
        
        # Initialize oscillator
        self.oscillator = Oscillator(
            oscillator_config['L'],
            oscillator_config['rho'], 
            oscillator_config['Y_e'],
            oscillator_config['n_layers']
        )
        
        # Pre-compute bin indices for MC events
        self.mc_bin_indices = self._get_bin_indices(self.mc_energies)
        
        print(f"Initialized likelihood with {len(self.mc_energies)} MC events")
        print(f"Data histogram sum: {jnp.sum(self.data_histogram):.1f}")
        print(f"Energy range: {jnp.min(self.mc_energies):.2f} - {jnp.max(self.mc_energies):.2f} GeV")
        
    def _compute_data_histogram(self) -> jnp.ndarray:
        """Compute the data histogram (fixed for pseudo-experiments)."""
        data_energies = self.data_monolith.monolith[:, MCEventIndices.TRUE_NEUTRINO_ENERGY.value]
        data_weights = self.data_monolith.monolith[:, MCEventIndices.WEIGHT.value]
        
        # Use JAX histogram function
        histogram, _ = jnp.histogram(data_energies, bins=self.energy_bins, weights=data_weights)
        
        return histogram
    
    def _get_bin_indices(self, energies: jnp.ndarray) -> jnp.ndarray:
        """Get bin indices for given energies."""
        # Use searchsorted to find bin indices
        bin_indices = jnp.searchsorted(self.energy_bins[1:], energies)
        # Ensure indices are within bounds
        bin_indices = jnp.clip(bin_indices, 0, self.n_bins - 1)
        return bin_indices
    
    def predict_histogram(self, osc_params: jnp.ndarray, syst_params: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Predict the histogram for given oscillation and systematic parameters.
        
        Args:
            osc_params: Oscillation parameters [theta12, theta13, theta23, deltacp, dm21, dm32]
            syst_params: Systematic parameters (not implemented yet)
            
        Returns:
            Predicted histogram
        """
        # Set up oscillator with energies and neutrino types
        self.oscillator.set_energy_osc(self.mc_energies, self.mc_start_nu, self.mc_end_nu)
        
        # Calculate oscillation probabilities
        osc_probs = self.oscillator.calc_probability(osc_params)
        
        # Apply systematic weights (for now, just use nominal weights)
        if syst_params is not None and self.use_systematics:
            # TODO: Implement systematic reweighting
            total_weights = self.mc_weights * osc_probs
        else:
            total_weights = self.mc_weights * osc_probs
        
        # Create predicted histogram using JAX-compatible operations
        predicted_histogram = jnp.zeros(self.n_bins)
        
        # Use JAX histogram function instead of manual binning
        predicted_histogram, _ = jnp.histogram(
            self.mc_energies, 
            bins=self.energy_bins, 
            weights=total_weights
        )
        
        return predicted_histogram
    
    def log_likelihood(self, osc_params: jnp.ndarray, syst_params: Optional[jnp.ndarray] = None):
        """
        Calculate the log-likelihood for given parameters.
        
        Args:
            osc_params: Oscillation parameters
            syst_params: Systematic parameters
            
        Returns:
            Log-likelihood value
        """
        # Get predicted histogram
        predicted = self.predict_histogram(osc_params, syst_params)
        
        # Add small regularization to avoid log(0)
        predicted = predicted + 1e-10
        
        # Poisson log-likelihood: sum over bins of (data * log(pred) - pred - log(data!))
        # We can drop the log(data!) term as it's constant
        log_likelihood = jnp.sum(self.data_histogram * jnp.log(predicted) - predicted)
        
        return log_likelihood
    
    def __call__(self, params: jnp.ndarray):
        """
        Convenience method for likelihood evaluation.
        
        Args:
            params: All parameters (oscillation + systematic)
            
        Returns:
            Log-likelihood value
        """
        # For now, assume all parameters are oscillation parameters
        osc_params = params[:6]  # First 6 are oscillation parameters
        syst_params = params[6:] if len(params) > 6 else None
        
        return self.log_likelihood(osc_params, syst_params)


def create_likelihood_from_config(
    config: Dict[str, Any],
    data_file: str,
    mc_file: str
) -> BinnedPoissonLikelihood:
    """
    Create a likelihood object from configuration.
    
    Args:
        config: Configuration dictionary
        data_file: Path to data file
        mc_file: Path to MC file
        
    Returns:
        Configured likelihood object
    """
    # This is a placeholder - in a real implementation, you'd load the files
    # For now, we'll create dummy data
    
    # Create some test data
    n_events = 10000
    
    # Generate realistic neutrino energies (log-normal distribution)
    np.random.seed(42)
    log_energies = np.random.normal(np.log(2.0), 0.8, n_events)
    energies = np.exp(log_energies)
    energies = np.clip(energies, 0.1, 20.0)  # Clip to reasonable range
    
    # Create MC events as individual MCEvent objects first
    mc_event_objs = []
    data_event_objs = []
    
    for i in range(n_events):
        # Most events are muon neutrino -> muon neutrino
        start_nu = 14
        end_nu = 14
        
        # Create MCEvent objects
        mc_event = MCEvent(
            true_neutrino_energy=energies[i],
            true_q2=0.5,
            reco_neutrino_energy=energies[i] * (1 + 0.1 * np.random.normal()),
            interaction_mode=1,
            start_nu=start_nu,
            end_nu=end_nu,
            target=1000060120,
            weight=1.0
        )
        
        data_event = MCEvent(
            true_neutrino_energy=energies[i],
            true_q2=0.5,
            reco_neutrino_energy=energies[i] * (1 + 0.1 * np.random.normal()),
            interaction_mode=1,
            start_nu=start_nu,
            end_nu=end_nu,
            target=1000060120,
            weight=1.0
        )
        
        mc_event_objs.append(mc_event)
        data_event_objs.append(data_event)
    
    # Create monoliths
    mc_monolith = MCEventMonolith(mc_event_objs)
    data_monolith = MCEventMonolith(data_event_objs)
    
    # Energy binning
    energy_bins = jnp.linspace(0.1, 10.0, 21)  # 20 bins from 0.1 to 10 GeV
    
    # Oscillator configuration
    oscillator_config = {
        'L': 1300.0,     # km
        'rho': 0.5,      # g/cm³
        'Y_e': 3.0,      # electron fraction
        'n_layers': 1000 # integration layers
    }
    
    return BinnedPoissonLikelihood(
        data_monolith=data_monolith,
        mc_monolith=mc_monolith,
        energy_bins=energy_bins,
        oscillator_config=oscillator_config,
        use_systematics=False
    )
