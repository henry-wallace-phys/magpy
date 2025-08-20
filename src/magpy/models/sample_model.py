# Central model for handling everything with JAX acceleration...
from typing import List, Optional, Union
import jax
import jax.numpy as jnp
import numpy as np
from cProfile import Profile
from pstats import SortKey, Stats
from tqdm import tqdm

# Enable 64-bit precision for accurate calculations
jax.config.update("jax_enable_x64", True)

from magpy.file_io.mc_file import MCFile
from magpy.file_io.spline_file import SplineFile
from magpy.file_io.systematic_file import SystematicFile
from magpy.objects.mc_event import MCEventIndices
from magpy.models.spline_syst_model import SplineSystematicModel
from magpy.oscillator.oscillator import Oscillator
from magpy.Exceptions import MagpyInvalidObjectError

class SampleModel:
    def __init__(self, 
                 mc_file: MCFile, 
                 spline_file: SplineFile, 
                 systematic_file: SystematicFile, 
                 oscillator: Oscillator, 
                 bin_variables: Optional[List[MCEventIndices]] = None):
        """
        Handle the full event model with JAX acceleration, including MC events, splines and systematics.
        """
        self.mc_file = mc_file
        if mc_file.monolith is None:
            raise MagpyInvalidObjectError("MC file does not contain a valid monolith.")
        
        self.spline_file = spline_file
        if spline_file.monolith is None:
            raise MagpyInvalidObjectError("Spline file does not contain a valid monolith.")

        self.spline_syst_handler = SplineSystematicModel(spline_file, systematic_file)

        self._bin_variables = bin_variables
        
        if self._bin_variables is None:
            self._mc_indices = None
        else:
            self.initialise_mc_indices()

        self.oscillator = oscillator
        self.oscillator.set_energy_osc(
            self.mc_monolith[:, MCEventIndices.TRUE_NEUTRINO_ENERGY.value],
            self.mc_monolith[:, MCEventIndices.START_NU.value],
            self.mc_monolith[:, MCEventIndices.END_NU.value]
        )

    def set_bin_variables(self, events: Union[List[MCEventIndices], jnp.ndarray]):        
        if isinstance(events, jnp.ndarray):
            self._bin_variables = events
        else:
            self._bin_variables = jnp.array([e for e in events])

    def initialise_mc_indices(self):
        if self._bin_variables is None:
            raise MagpyInvalidObjectError("Bin variables not set. Please set bin variables before initialising MC indices.")

        self._mc_indices = self.spline_syst_handler.get_monolith_splines(self.mc_file.monolith, self._bin_variables)
        
    @property
    def mc_indices(self) -> Optional[jnp.ndarray]:
        return self._mc_indices
    
    @property
    def mc_monolith(self) -> jnp.ndarray:
        return self.mc_file.monolith.monolith
    
    def reweight(self, osc_pars: jnp.ndarray, syst_pars: jnp.ndarray) -> jnp.ndarray:
        """
        Reweight the MC events based on the spline systematics using JAX.
        """
        if self._mc_indices is None:
            raise MagpyInvalidObjectError("MC indices not initialised. Please initialise MC indices before reweighting.")
        
        # Reset weights to 1.0 - create new array instead of in-place modification
        monolith = self.mc_monolith.at[:, MCEventIndices.WEIGHT.value].set(1.0)
        
        # Calculate oscillation probabilities
        osc_weights = self.oscillator.calc_probability(osc_params=osc_pars)
        
        # Apply oscillation weights
        monolith = monolith.at[:, MCEventIndices.WEIGHT.value].set(osc_weights)
        
        # Apply systematic reweighting
        monolith = self.spline_syst_handler.reweight(syst_pars, monolith)
 
        return monolith