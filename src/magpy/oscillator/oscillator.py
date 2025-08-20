"""
Simplified JAX-based oscillator for performance testing.
Focus on vectorization without complex control flow.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from magpy.Exceptions import MagpyProbabilityException

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)
@jit
def _simplified_oscillation_jax(
    energies: jnp.ndarray,
    osc_in: jnp.ndarray,
    osc_out: jnp.ndarray,
    s12sq: float, s13sq: float, s23sq: float, delta_cp: float, dmsq21: float, dmsq31: float,
    L: float, ye: float, rho: float
) -> jnp.ndarray:
    """
    Simplified JAX implementation focusing on core vectorized calculations.
    No Newton iterations for now to avoid JAX compilation issues.
    """
    
    # Constants
    eVsqkm_to_GeV_over4 = 1e-9 / 1.97327e-7 * 1e3 / 4
    YerhoE2a = 1.52588e-4
    
    # Pre-compute frequently used terms (vectorized)
    c13sq = 1.0 - s13sq
    c12sq = 1.0 - s12sq
    c23sq = 1.0 - s23sq
    
    # Energy-dependent calculations (all vectorized)
    signed_energies = energies * jnp.sign(osc_in)
    Lover4E = eVsqkm_to_GeV_over4 * L / signed_energies
    Amatter = ye * rho * signed_energies * YerhoE2a
    
    # Initial values calculation (vectorized)
    Ue2sq = c13sq * s12sq
    Ue3sq = s13sq
    Um3sq = c13sq * s23sq
    Um2sq = c12sq * c23sq
    Ut2sq = s13sq * s12sq * s23sq
    
    # Matter effects
    Jrr = jnp.sqrt(Um2sq * Ut2sq)
    sind = jnp.sin(delta_cp)
    cosd = jnp.cos(delta_cp)
    Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd
    Jmatter = 8 * Jrr * c13sq * sind
    
    # Broadcast to match array size
    n_events = energies.shape[0]
    Ue2sq = jnp.full(n_events, Ue2sq) if jnp.isscalar(Ue2sq) else Ue2sq
    Ue3sq = jnp.full(n_events, Ue3sq) if jnp.isscalar(Ue3sq) else Ue3sq
    Um3sq = jnp.full(n_events, Um3sq) if jnp.isscalar(Um3sq) else Um3sq
    Um2sq = jnp.full(n_events, Um2sq) if jnp.isscalar(Um2sq) else Um2sq
    Jmatter = jnp.full(n_events, Jmatter) if jnp.isscalar(Jmatter) else Jmatter
    
    # Core oscillation calculations (all vectorized)
    Dmsqee = dmsq31 - s12sq * dmsq21
    A = dmsq21 + dmsq31
    See = A - dmsq21 * Ue2sq - dmsq31 * Ue3sq
    Tmm = dmsq21 * dmsq31
    Tee = Tmm * (1.0 - Ue3sq - Ue2sq)
    C = Amatter * Tee
    A = A + Amatter
    
    # Lambda3 calculation without Newton iterations (vectorized)
    xmat = Amatter / Dmsqee
    xmat_minus_1 = xmat - 1.0
    sqrt_term = jnp.sqrt(xmat_minus_1 * xmat_minus_1 + 4.0 * s13sq * xmat)
    lambda3 = dmsq31 + 0.5 * Dmsqee * (xmat_minus_1 + sqrt_term)
    
    # Lambda calculations (vectorized)
    A_minus_lambda3 = A - lambda3
    sqrt_term_lambda = jnp.sqrt(A_minus_lambda3 * A_minus_lambda3 - 4.0 * C / lambda3)
    Dlambda21 = sqrt_term_lambda
    lambda2 = 0.5 * (A - lambda3 + Dlambda21)
    Dlambda32 = lambda3 - lambda2
    Dlambda31 = Dlambda32 + Dlambda21
    
    # Rosetta calculations (vectorized)
    PiDlambdaInv = 1.0 / (Dlambda31 * Dlambda32 * Dlambda21)
    Xp3 = PiDlambdaInv * Dlambda21
    Xp2 = -PiDlambdaInv * Dlambda31
    
    # U matrix elements (vectorized)
    Ue3sq_final = (lambda3 * (lambda3 - See) + Tee) * Xp3
    Ue2sq_final = (lambda2 * (lambda2 - See) + Tee) * Xp2
    
    Smm = A - dmsq21 * Um2sq - dmsq31 * Um3sq
    See_plus_Smm_minus_A = See + Smm - A
    Tmm_final = Tmm * (1.0 - Um3sq - Um2sq) + Amatter * See_plus_Smm_minus_A
    
    Um3sq_final = (lambda3 * (lambda3 - Smm) + Tmm_final) * Xp3
    Um2sq_final = (lambda2 * (lambda2 - Smm) + Tmm_final) * Xp2
    
    Jmatter_final = (Jmatter * dmsq21 * dmsq31 * (dmsq31 - dmsq21) * PiDlambdaInv)
    
    # Calculate all U matrix elements (vectorized)
    Ue1sq = 1.0 - Ue3sq_final - Ue2sq_final
    Um1sq = 1.0 - Um3sq_final - Um2sq_final
    Ut3sq = 1.0 - Um3sq_final - Ue3sq_final
    Ut2sq = 1.0 - Um2sq_final - Ue2sq_final
    Ut1sq = 1.0 - Um1sq - Ue1sq
    
    # Kinematic terms (vectorized)
    D21 = Dlambda21 * Lover4E
    D32 = Dlambda32 * Lover4E
    D31 = D32 + D21
    
    # Trigonometric calculations (vectorized)
    sinD21 = jnp.sin(D21)
    sinD31 = jnp.sin(D31)
    sinD32 = jnp.sin(D32)
    triple_sin = sinD21 * sinD31 * sinD32
    
    sinsqD21_2 = 2.0 * sinD21 * sinD21
    sinsqD31_2 = 2.0 * sinD31 * sinD31
    sinsqD32_2 = 2.0 * sinD32 * sinD32
    
    # Probability calculations (vectorized)
    # pme_CPC calculation
    Ut_terms_0 = Ut3sq - Um2sq_final * Ue1sq - Um1sq * Ue2sq_final
    Ut_terms_1 = Ut2sq - Um3sq_final * Ue1sq - Um1sq * Ue3sq_final
    Ut_terms_2 = Ut1sq - Um3sq_final * Ue2sq_final - Um2sq_final * Ue3sq_final
    
    pme_CPC = (Ut_terms_0 * sinsqD21_2 + 
               Ut_terms_1 * sinsqD31_2 + 
               Ut_terms_2 * sinsqD32_2)
    
    # pmm calculation
    Um_terms_0 = Um2sq_final * Um1sq
    Um_terms_1 = Um3sq_final * Um1sq  
    Um_terms_2 = Um3sq_final * Um2sq_final
    
    pmm_sum = (Um_terms_0 * sinsqD21_2 + 
               Um_terms_1 * sinsqD31_2 + 
               Um_terms_2 * sinsqD32_2)
    pmm = 1.0 - 2.0 * pmm_sum
    
    # pee calculation
    Ue_terms_0 = Ue2sq_final * Ue1sq
    Ue_terms_1 = Ue3sq_final * Ue1sq
    Ue_terms_2 = Ue3sq_final * Ue2sq_final
    
    pee_sum = (Ue_terms_0 * sinsqD21_2 + 
               Ue_terms_1 * sinsqD31_2 + 
               Ue_terms_2 * sinsqD32_2)
    pee = 1.0 - 2.0 * pee_sum
    
    pme_CPV = -Jmatter_final * triple_sin
    pem = pme_CPC - pme_CPV
    pme = pme_CPC + pme_CPV
    
    # Tau probabilities
    pet = 1.0 - pee - pem
    pmt = 1.0 - pme - pmm
    ptm = 1.0 - pem - pmm
    pte = 1.0 - pee - pme
    ptt = 1.0 - pet - pmt
    
    # Vectorized probability assignment using JAX's where
    results = jnp.zeros(n_events)
    abs_osc_in = jnp.abs(osc_in)
    abs_osc_out = jnp.abs(osc_out)
    
    # Main oscillation channels
    results = jnp.where((abs_osc_in == 12) & (abs_osc_out == 14), pem, results)  # e->mu
    results = jnp.where((abs_osc_in == 14) & (abs_osc_out == 12), pme, results)  # mu->e
    results = jnp.where((abs_osc_in == 14) & (abs_osc_out == 14), pmm, results)  # mu->mu
    results = jnp.where((abs_osc_in == 12) & (abs_osc_out == 12), pee, results)  # e->e
    
    # Tau channels
    results = jnp.where((abs_osc_in == 12) & (abs_osc_out == 16), pet, results)  # e->tau
    results = jnp.where((abs_osc_in == 14) & (abs_osc_out == 16), pmt, results)  # mu->tau
    results = jnp.where((abs_osc_in == 16) & (abs_osc_out == 14), ptm, results)  # tau->mu
    results = jnp.where((abs_osc_in == 16) & (abs_osc_out == 12), pte, results)  # tau->e
    results = jnp.where((abs_osc_in == 16) & (abs_osc_out == 16), ptt, results)  # tau->tau
    
    return results


class Oscillator:
    """
    Simplified JAX-based oscillator focusing on vectorization performance.
    """
    
    def __init__(self, L: float, ye: float, rho: float, n_newton: int):
        self.L = L
        self.rho = rho
        self.ye = ye
        self.n_newton = n_newton  # Note: Newton iterations not implemented in simplified version
        
        self._energies = None
        self._osc_in = None
        self._osc_out = None
        self._setup = False
        
        # Pre-compile the core function
        self._compiled_calc = jit(_simplified_oscillation_jax)
    
    def set_energy_osc(self, energies, osc_in, osc_out):
        """Set energy and oscillation channel arrays"""
        # Convert to JAX arrays with optimal memory layout
        self._energies = jnp.asarray(energies, dtype=jnp.float64)
        self._osc_in = jnp.asarray(osc_in, dtype=jnp.int32)
        self._osc_out = jnp.asarray(osc_out, dtype=jnp.int32)
        
        if len(self._energies) != len(self._osc_in) or len(self._energies) != len(self._osc_out):
            raise MagpyProbabilityException(
                "All input arrays must have the same length."
            )
        
        self._setup = True
    
    def calc_probability(self, osc_params) -> jnp.ndarray:
        """Calculate oscillation probabilities using simplified JAX implementation"""
        if not self._setup:
            raise MagpyProbabilityException("Oscillator not set up. Please call set_energy_osc() first.")
        
        # Convert parameters to JAX format
        params_array = jnp.asarray(osc_params, dtype=jnp.float64)
        
        # Extract parameters
        s12sq, s13sq, s23sq, delta_cp, dmsq21, dmsq31 = params_array
        
        # Call the fully compiled and vectorized calculation
        results = self._compiled_calc(
            self._energies, self._osc_in, self._osc_out,
            float(s12sq), float(s13sq), float(s23sq), float(delta_cp), float(dmsq21), float(dmsq31),
            self.L, self.ye, self.rho
        )
        
        return results
