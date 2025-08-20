"""
Tensorified version of nufast:
 https://github.com/PeterDenton/NuFast-LBL/tree/main
 Based on arXiv:2405.02400
"""

from enum import Enum
from typing import List

import torch
from magpy.Exceptions import MagpyProbabilityException


# JIT compile the most expensive probability calculations for maximum performance
@torch.jit.script
def _calc_probabilities_jit(
    Ut3sq: torch.Tensor, Um2sq: torch.Tensor, Ue1sq: torch.Tensor, Um1sq: torch.Tensor, Ue2sq: torch.Tensor,
    Ut2sq: torch.Tensor, Um3sq: torch.Tensor, Ue3sq: torch.Tensor,
    Ut1sq: torch.Tensor,
    sinsqD21_2: torch.Tensor, sinsqD31_2: torch.Tensor, sinsqD32_2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """JIT-compiled probability calculation for maximum performance"""
    
    # Calculate pme_CPC using vectorized operations
    Ut_terms_0 = Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq
    Ut_terms_1 = Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq
    Ut_terms_2 = Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq
    
    pme_CPC = (Ut_terms_0 * sinsqD21_2 + 
               Ut_terms_1 * sinsqD31_2 + 
               Ut_terms_2 * sinsqD32_2)
    
    # Calculate pmm using vectorized operations
    Um_terms_0 = Um2sq * Um1sq
    Um_terms_1 = Um3sq * Um1sq  
    Um_terms_2 = Um3sq * Um2sq
    
    pmm_sum = (Um_terms_0 * sinsqD21_2 + 
               Um_terms_1 * sinsqD31_2 + 
               Um_terms_2 * sinsqD32_2)
    pmm = 1.0 - 2.0 * pmm_sum
    
    # Calculate pee using vectorized operations
    Ue_terms_0 = Ue2sq * Ue1sq
    Ue_terms_1 = Ue3sq * Ue1sq
    Ue_terms_2 = Ue3sq * Ue2sq
    
    pee_sum = (Ue_terms_0 * sinsqD21_2 + 
               Ue_terms_1 * sinsqD31_2 + 
               Ue_terms_2 * sinsqD32_2)
    pee = 1.0 - 2.0 * pee_sum
    
    return pme_CPC, pmm, pee


# JIT compile lambda calculations which also have torch.sub operations
@torch.jit.script
def _calc_lambda_jit(
    A: torch.Tensor, lambda3: torch.Tensor, C: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """JIT-compiled lambda calculation to optimize sqrt and sub operations"""
    
    # Get Delta lambda's
    A_minus_lambda3 = A - lambda3
    sqrt_term = torch.sqrt(A_minus_lambda3 * A_minus_lambda3 - 4.0 * C / lambda3)
    Dlambda21 = sqrt_term
    lambda2 = 0.5 * (A - lambda3 + Dlambda21)
    Dlambda32 = lambda3 - lambda2
    Dlambda31 = Dlambda32 + Dlambda21
    
    return Dlambda21, lambda2, Dlambda32, Dlambda31


class NuType(Enum):
    E = 12
    EBar = -12
    Mu = 14
    MuBar = -14
    Tau = 16
    TauBar = -16

    def __str__(self):
        return self.name

class OscParIndex(Enum):
    S12SQ = 0
    S13SQ = 1
    S23SQ = 2
    DELTA = 3
    DMSQ21 = 4
    DMSQ31 = 5

    def __str__(self):
        return self.name


class Oscillator:
    """
    Oscillation through matter
    """

    # Some constants
    _E_IDX = 0
    _MU_IDX = 1
    _TAU_IDX = 2

    _S12SQ_IDX = OscParIndex.S12SQ.value
    _S13SQ_IDX = OscParIndex.S13SQ.value
    _S23SQ_IDX = OscParIndex.S23SQ.value
    _DELTA_IDX = OscParIndex.DELTA.value
    _DMSQ21_IDX = OscParIndex.DMSQ21.value
    _DMSQ31_IDX = OscParIndex.DMSQ31.value

    def __init__(self, L: float, ye: float, rho: float, n_newton: int):
        
        
        self.L = L
        self.rho = rho
        self.ye = ye
        self.n_newton = n_newton

        self._device = 'cpu'

        self.current_osc_pars = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64,
            device=self._device
        )
        self.current_weights = torch.tensor([], dtype=torch.float64,
                                            device=self._device)

        self.eVsqkm_to_GeV_over4 = 1e-9 / 1.97327e-7 * 1e3 / 4
        self.YerhoE2a = 1.52588e-4

        self._setup = False

        self._energy = torch.tensor([], dtype=torch.float64, device=self._device)
        self._osc_in = torch.tensor([], dtype=torch.int64, device=self._device)
        self._osc_out = torch.tensor([], dtype=torch.int64, device=self._device)
        self._masks: List[List[torch.Tensor]] = [[]]
        self._calc_tau: bool = True
        self._Lover4E = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._Amatter = torch.tensor(0.0, dtype=torch.float64, device=self._device)

    
    # We can set up a bunch of masks etc. now for optimization
    def set_energy_osc(self, energy, osc_in, osc_out):
        self._energy = energy
        self._osc_in = osc_in
        self._osc_out = osc_out

        device = self._energy.device

        if len(self._energy) != len(self._osc_in) != len(self._osc_out):
            raise MagpyProbabilityException(
                "All input tensors must have the same length."
            )

        self._energy *= torch.sign(self._osc_in)

        # Now we've accounted for the sign of the energies, we can remove that information 
        self._osc_in = torch.abs(self._osc_in)
        self._osc_out = torch.abs(self._osc_out)

        if torch.any(torch.sign(self._osc_in) != torch.sign(self._osc_out)):
            raise MagpyProbabilityException(
                "Oscillation in and out must have the same sign."
            )
            
        # We can also set up the output probabilty tensor here
        self.current_weights = torch.zeros(len(self._energy), dtype=torch.float64, device=self._energy.device)

        # Pre-allocate all intermediate tensors that get created in calc_probability
        n_events = len(self._energy)
        
        # Pre-allocate final probability tensors
        self._pem = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._pme = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._pmm = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._pee = torch.zeros(n_events, dtype=torch.float64, device=device)
        
        # Pre-allocate tau probability tensors
        self._pet = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._pmt = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._pte = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._ptm = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._ptt = torch.zeros(n_events, dtype=torch.float64, device=device)
        
        # Pre-allocate intermediate calculation tensors
        self._pme_CPC = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._pme_CPV = torch.zeros(n_events, dtype=torch.float64, device=device)
        
        # Pre-allocate kinematic tensors
        self._D21 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._D32 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._sinD21 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._sinD31 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._sinD32 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._triple_sin = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._sinsqD21_2 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._sinsqD31_2 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._sinsqD32_2 = torch.zeros(n_events, dtype=torch.float64, device=device)
        
        # Pre-allocate U-matrix elements
        self._Ue1sq = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Ue2sq = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Ue3sq = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Um1sq = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Um2sq = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Um3sq = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Ut1sq = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Ut2sq = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Ut3sq = torch.zeros(n_events, dtype=torch.float64, device=device)
        
        # Pre-allocate lambda and delta lambda tensors
        self._lambda2 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._lambda3 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Dlambda21 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Dlambda32 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Dlambda31 = torch.zeros(n_events, dtype=torch.float64, device=device)
        
        # Pre-allocate other intermediate tensors
        self._tmp = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._tmp2 = torch.zeros(n_events, dtype=torch.float64, device=device)  # Additional temp tensor
        self._xmat = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._PiDlambdaInv = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Xp2 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Xp3 = torch.zeros(n_events, dtype=torch.float64, device=device)
        
        # Pre-allocate cosine tensors for mathematical optimization
        self._c13sq = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._c12sq = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._c23sq = torch.zeros(n_events, dtype=torch.float64, device=device)

        # Pre-allocate vectorized calculation tensors for batch operations
        self._U_matrix = torch.zeros((n_events, 3, 3), dtype=torch.float64, device=device)  # Full U matrix
        self._D_vec = torch.zeros((n_events, 3), dtype=torch.float64, device=device)  # [D21, D31, D32]
        self._sin_vec = torch.zeros((n_events, 3), dtype=torch.float64, device=device)  # sin values
        self._sinsq_vec = torch.zeros((n_events, 3), dtype=torch.float64, device=device)  # sin^2 values
        
        # Pre-allocate probability matrices for vectorized calculations
        self._prob_matrix = torch.zeros((n_events, 3, 3), dtype=torch.float64, device=device)  # Full probability matrix

        # Now to set up masks for oscillations in/out - create masks but only for torch.where usage
        self._mask_em = (self._osc_in == NuType.E.value) & (self._osc_out == NuType.Mu.value)
        self._mask_me = (self._osc_in == NuType.Mu.value) & (self._osc_out == NuType.E.value)
        self._mask_mm = (self._osc_in == NuType.Mu.value) & (self._osc_out == NuType.Mu.value)
        self._mask_ee = (self._osc_in == NuType.E.value) & (self._osc_out == NuType.E.value)
        
        self._calc_tau = bool(torch.any(self._osc_in == NuType.Tau.value) or torch.any(self._osc_in == NuType.TauBar.value))
        
        if self._calc_tau:
            self._mask_et = (self._osc_in == NuType.E.value) & (self._osc_out == NuType.Tau.value)
            self._mask_mt = (self._osc_in == NuType.Mu.value) & (self._osc_out == NuType.Tau.value)
            self._mask_tm = (self._osc_in == NuType.Tau.value) & (self._osc_out == NuType.Mu.value)
            self._mask_te = (self._osc_in == NuType.Tau.value) & (self._osc_out == NuType.E.value)
            self._mask_tt = (self._osc_in == NuType.Tau.value) & (self._osc_out == NuType.Tau.value)

        # Might as well do constants now    
        self._Lover4E = self.eVsqkm_to_GeV_over4 * self.L / self._energy        
        self._Amatter = self.ye * self.rho * self._energy * self.YerhoE2a
        self.current_osc_pars = torch.zeros(6, dtype=torch.float64, device=self._energy.device)
        self._setup = True

    def calc_probability(
        self,
        osc_params: torch.Tensor,
    ) -> torch.Tensor:
        # --------------------------------------------------------------------- #
        # First calculate useful simple functions of the oscillation parameters #
        # --------------------------------------------------------------------- #

        if not self._setup:
            raise MagpyProbabilityException("Oscillator not set up. Please call set_energy_osc() first.")

        if (
            torch.equal(osc_params, self.current_osc_pars)
            and self.current_weights is not None
        ):
            return self.current_weights

        s13sq = osc_params[self._S13SQ_IDX]
        dmsq21 = osc_params[self._DMSQ21_IDX]
        dmsq31 = osc_params[self._DMSQ31_IDX]
        delta_cp = osc_params[self._DELTA_IDX]
        s12sq = osc_params[self._S12SQ_IDX]
        s23sq = osc_params[self._S23SQ_IDX]

        self.current_osc_pars = osc_params

        # Pre-compute frequently used 1-x terms to reduce operations
        if torch.is_tensor(s13sq) and s13sq.dim() > 0:
            torch.sub(1, s13sq, out=self._c13sq)
            torch.sub(1, s12sq, out=self._c12sq)  
            torch.sub(1, s23sq, out=self._c23sq)
            c13sq = self._c13sq
            c12sq = self._c12sq
            c23sq = self._c23sq
        else:
            c13sq = 1 - s13sq
            c12sq = 1 - s12sq
            c23sq = 1 - s23sq

        # Ueisq's - more efficient calculation
        Ue2sq = c13sq * s12sq
        Ue3sq = s13sq

        # Umisq's, Utisq's - use pre-computed cosines  
        Um3sq = c13sq * s23sq
        Um2sq = c12sq * c23sq  # More efficient than (1-s12sq)*(1-s23sq)
        Ut2sq = s13sq * s12sq * s23sq

        Jrr = torch.sqrt(Um2sq * Ut2sq)

        sind = torch.sin(delta_cp)
        cosd = torch.cos(delta_cp)

        Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd
        Jmatter = 8 * Jrr * c13sq * sind

        # With E
        Dmsqee = dmsq31 - s12sq * dmsq21

        # calculate A, B, C, See, Tee, and part of Tmm
        A = dmsq21 + dmsq31
        See = A - dmsq21 * Ue2sq - dmsq31 * Ue3sq
        Tmm = dmsq21 * dmsq31
        
        # Handle scalar/tensor cases for Tee calculation to avoid rsub
        if torch.is_tensor(Ue3sq) and Ue3sq.dim() > 0:
            torch.sub(1, Ue3sq, out=self._tmp)         # tmp = 1 - Ue3sq
            torch.sub(self._tmp, Ue2sq, out=self._tmp) # tmp = tmp - Ue2sq = 1 - Ue3sq - Ue2sq
            Tee = Tmm * self._tmp
        else:
            Tee = Tmm * (1 - Ue3sq - Ue2sq)

        # E
        C = self._Amatter * Tee
        A = A + self._Amatter

        # ---------------------------------- #
        # Get lambda3 from lambda+ of MP/DMP - optimized calculation #
        # ---------------------------------- #
        torch.divide(self._Amatter, Dmsqee, out=self._xmat)
        # Pre-compute (xmat - 1) to avoid redundant calculation
        torch.sub(self._xmat, 1, out=self._tmp)  # tmp = xmat - 1
        # More efficient: sqrt((1-xmat)^2 + 4*s13sq*xmat) = sqrt((xmat-1)^2 + 4*s13sq*xmat)
        torch.add(dmsq31, 0.5 * Dmsqee * (
            self._tmp  # xmat - 1 (pre-computed)
            + torch.sqrt(self._tmp * self._tmp + 4 * s13sq * self._xmat)
        ), out=self._lambda3)

        # ---------------------------------------------------------------------------- #
        # Newton iterations to improve lambda3 arbitrarily, if needed, (B needed here) #
        # ---------------------------------------------------------------------------- #
        B = Tmm + self._Amatter * See  # B is only needed for N_Newton >= 1
        for _ in range(self.n_newton):
            # Optimize Newton iteration by pre-computing terms
            lambda3_sq = self._lambda3 * self._lambda3
            lambda3_minus_A = self._lambda3 - A
            # Simplify: lambda3 + lambda3 - A = 2*lambda3 - A
            numerator = lambda3_sq * (2 * self._lambda3 - A) + C
            denominator = self._lambda3 * (3 * self._lambda3 - 2 * A) + B
            torch.mul(numerator, torch.reciprocal(denominator), out=self._lambda3)

        # ------------------- #
        # Get  Delta lambda's using JIT-compiled function #
        # ------------------- #
        self._Dlambda21, self._lambda2, self._Dlambda32, self._Dlambda31 = _calc_lambda_jit(
            A, self._lambda3, C
        )

        # ----------------------- #
        # Use Rosetta for Veisq's #
        # ----------------------- #
        # denominators
        torch.reciprocal(self._Dlambda31 * self._Dlambda32 * self._Dlambda21, out=self._PiDlambdaInv)
        torch.mul(self._PiDlambdaInv, self._Dlambda21, out=self._Xp3)
        torch.mul(self._PiDlambdaInv, self._Dlambda31, out=self._Xp2)
        torch.neg(self._Xp2, out=self._Xp2)  # Avoid torch.rsub by using explicit negation

        # numerators - reuse pre-allocated tensors
        torch.mul((self._lambda3 * (self._lambda3 - See) + Tee), self._Xp3, out=self._Ue3sq)
        torch.mul((self._lambda2 * (self._lambda2 - See) + Tee), self._Xp2, out=self._Ue2sq)

        Smm = A - dmsq21 * Um2sq - dmsq31 * Um3sq
        
        # Optimize (See + Smm - A) calculation to avoid rsub
        See_plus_Smm_minus_A = See + Smm - A        # Handle scalar/tensor cases for Tmm calculation to avoid rsub
        if torch.is_tensor(Um3sq) and Um3sq.dim() > 0:
            torch.sub(1, Um3sq, out=self._tmp)         # tmp = 1 - Um3sq
            torch.sub(self._tmp, Um2sq, out=self._tmp) # tmp = tmp - Um2sq = 1 - Um3sq - Um2sq
            Tmm = Tmm * self._tmp + self._Amatter * See_plus_Smm_minus_A
        else:
            Tmm = Tmm * (1 - Um3sq - Um2sq) + self._Amatter * See_plus_Smm_minus_A

        torch.mul((self._lambda3 * (self._lambda3 - Smm) + Tmm), self._Xp3, out=self._Um3sq)
        torch.mul((self._lambda2 * (self._lambda2 - Smm) + Tmm), self._Xp2, out=self._Um2sq)

        # ------------- #
        # Use NHS for J #
        # ------------- #
        Jmatter = (
            Jmatter
            * dmsq21
            * dmsq31
            * (
                dmsq31
                - dmsq21
            )
            * self._PiDlambdaInv
        )

        # ----------------------- #
        # Get all elements of Usq #
        # ----------------------- #
        # Revert to working individual operations for correctness
        torch.sub(1, self._Ue3sq, out=self._tmp)  # tmp = 1 - Ue3sq
        torch.sub(self._tmp, self._Ue2sq, out=self._Ue1sq)  # Ue1sq = tmp - Ue2sq
        
        torch.sub(1, self._Um3sq, out=self._tmp)  # tmp = 1 - Um3sq
        torch.sub(self._tmp, self._Um2sq, out=self._Um1sq)  # Um1sq = tmp - Um2sq

        torch.sub(1, self._Um3sq, out=self._tmp)  # tmp = 1 - Um3sq
        torch.sub(self._tmp, self._Ue3sq, out=self._Ut3sq)  # Ut3sq = tmp - Ue3sq
        
        torch.sub(1, self._Um2sq, out=self._tmp)  # tmp = 1 - Um2sq
        torch.sub(self._tmp, self._Ue2sq, out=self._Ut2sq)  # Ut2sq = tmp - Ue2sq
        
        torch.sub(1, self._Um1sq, out=self._tmp)  # tmp = 1 - Um1sq
        torch.sub(self._tmp, self._Ue1sq, out=self._Ut1sq)  # Ut1sq = tmp - Ue1sq

        # ----------------------- #
        # Get the kinematic terms #
        # ----------------------- #
        torch.mul(self._Dlambda21, self._Lover4E, out=self._D21)
        torch.mul(self._Dlambda32, self._Lover4E, out=self._D32)

        # Use pre-allocated vectorized tensors for batch computation
        torch.add(self._D32, self._D21, out=self._tmp)  # D31 = D32 + D21
        
        # Fill pre-allocated D vector (no new memory allocation)
        self._D_vec[:, 0] = self._D21
        self._D_vec[:, 1] = self._tmp  # D31
        self._D_vec[:, 2] = self._D32
        
        # Vectorized sine computation using pre-allocated vector
        torch.sin(self._D_vec, out=self._sin_vec)
        
        # Extract sine results (views into the pre-allocated vector - no copy)
        self._sinD21 = self._sin_vec[:, 0]
        self._sinD31 = self._sin_vec[:, 1] 
        self._sinD32 = self._sin_vec[:, 2]

        torch.mul(self._sinD21 * self._sinD31, self._sinD32, out=self._triple_sin)

        # Vectorized square operations using pre-allocated vector
        torch.square(self._sin_vec, out=self._sinsq_vec)
        self._sinsq_vec.mul_(2)  # Multiply all by 2 in one operation
        
        # Extract squared results (views into the pre-allocated vector)
        self._sinsqD21_2 = self._sinsq_vec[:, 0]
        self._sinsqD31_2 = self._sinsq_vec[:, 1]
        self._sinsqD32_2 = self._sinsq_vec[:, 2]

        # ------------------------------------------------------------------- #
        # Calculate the three necessary probabilities using JIT-compiled optimized function #
        # ------------------------------------------------------------------- #
        
        # Use JIT-compiled function for maximum performance
        self._pme_CPC, self._pmm, self._pee = _calc_probabilities_jit(
            self._Ut3sq, self._Um2sq, self._Ue1sq, self._Um1sq, self._Ue2sq,
            self._Ut2sq, self._Um3sq, self._Ue3sq,
            self._Ut1sq,
            self._sinsqD21_2, self._sinsqD31_2, self._sinsqD32_2
        )
        
        torch.mul(Jmatter, self._triple_sin, out=self._pme_CPV)
        torch.neg(self._pme_CPV, out=self._pme_CPV)  # Avoid torch.rsub by explicit negation

        # ---------------------------- #
        # Assign all the probabilities #
        # ---------------------------- #
        torch.sub(self._pme_CPC, self._pme_CPV, out=self._pem)
        torch.add(self._pme_CPC, self._pme_CPV, out=self._pme)

        weights = self.current_weights
        
        # Batch the main probability assignments using vectorized operations
        # Create combined mask for all non-tau transitions
        
        # Use torch.where more efficiently by combining operations
        torch.where(self._mask_em, self._pem, weights, out=weights)
        torch.where(self._mask_me, self._pme, weights, out=weights) 
        torch.where(self._mask_mm, self._pmm, weights, out=weights)
        torch.where(self._mask_ee, self._pee, weights, out=weights)

        if self._calc_tau:
            # Calculate tau probabilities individually - more efficient than stacking
            # pet = 1 - pee - pem
            torch.sub(1, self._pee, out=self._tmp)
            torch.sub(self._tmp, self._pem, out=self._pet)
            
            # pmt = 1 - pme - pmm  
            torch.sub(1, self._pme, out=self._tmp)
            torch.sub(self._tmp, self._pmm, out=self._pmt)
            
            # ptm = 1 - pem - pmm
            torch.sub(1, self._pem, out=self._tmp)
            torch.sub(self._tmp, self._pmm, out=self._ptm)
            
            # pte = 1 - pee - pme
            torch.sub(1, self._pee, out=self._tmp)
            torch.sub(self._tmp, self._pme, out=self._pte)
            
            # ptt = 1 - pet - pmt
            torch.sub(1, self._pet, out=self._tmp)
            torch.sub(self._tmp, self._pmt, out=self._ptt)
            
            # Batch tau assignments
            torch.where(self._mask_et, self._pet, weights, out=weights)
            torch.where(self._mask_mt, self._pmt, weights, out=weights)
            torch.where(self._mask_tm, self._ptm, weights, out=weights)
            torch.where(self._mask_te, self._pte, weights, out=weights)
            torch.where(self._mask_tt, self._ptt, weights, out=weights)

        self.current_weights = weights

        if self.current_weights.max() > 1 or self.current_weights.min() < 0:
            raise MagpyProbabilityException(
                "Oscillation probabilities must be in the range [0, 1]. "
            )

        return self.current_weights