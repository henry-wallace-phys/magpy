"""
Tensorified version of nufast:
 https://github.com/PeterDenton/NuFast-LBL/tree/main
 Based on arXiv:2405.02400
"""

from enum import Enum
from typing import List

import torch
from magpy.Exceptions import MagpyProbabilityException

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
        self._xmat = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._PiDlambdaInv = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Xp2 = torch.zeros(n_events, dtype=torch.float64, device=device)
        self._Xp3 = torch.zeros(n_events, dtype=torch.float64, device=device)

        # Pre-allocate vectorized stack tensors - these will be filled during calc_probability
        self._D_stack = torch.zeros((n_events, 3), dtype=torch.float64, device=device)
        self._sin_stack = torch.zeros((n_events, 3), dtype=torch.float64, device=device)
        self._sinsq_stack = torch.zeros((n_events, 3), dtype=torch.float64, device=device)
        
        # Pre-allocate probability calculation stacks
        self._Ut_terms = torch.zeros((n_events, 3), dtype=torch.float64, device=device)
        self._Um_terms = torch.zeros((n_events, 3), dtype=torch.float64, device=device)
        self._Ue_terms = torch.zeros((n_events, 3), dtype=torch.float64, device=device)

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

        c13sq = 1 - s13sq

        # Ueisq's
        Ue2sq = c13sq * s12sq
        Ue3sq = s13sq

        # Umisq's, Utisq's and Jvac
        Um3sq = c13sq * s23sq
        # Um2sq and Ut2sq are used here as temporary variables, will be properly defined later
        Ut2sq = (
            s13sq
            * s12sq
            * s23sq
        )
        Um2sq = (1 - s12sq) * (
            1 - s23sq
        )

        Jrr = torch.sqrt(Um2sq * Ut2sq)

        sind = torch.sin(delta_cp)
        cosd = torch.cos(delta_cp)

        Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd
        Jmatter = 8 * Jrr * c13sq * sind

        # With E
        Dmsqee = (
            dmsq31
            - s12sq * dmsq21
        )  # Dmsq21

        # calculate A, B, C, See, Tee, and part of Tmm
        A = (
            dmsq21 + dmsq31
        )  # temporary variable
        See = (
            A
            - dmsq21 * Ue2sq
            - dmsq31 * Ue3sq
        )
        Tmm = (
            dmsq21 * dmsq31
        )  # using Tmm as a temporary variable
        Tee = Tmm * (1 - Ue3sq - Ue2sq)

        # E
        C = self._Amatter * Tee
        A = A + self._Amatter

        # ---------------------------------- #
        # Get lambda3 from lambda+ of MP/DMP #
        # ---------------------------------- #
        torch.divide(self._Amatter, Dmsqee, out=self._xmat)
        torch.sub(1, self._xmat, out=self._tmp)
        torch.add(dmsq31, 0.5 * Dmsqee * (
            self._xmat
            - 1
            + torch.sqrt(self._tmp * self._tmp + 4 * s13sq * self._xmat)
        ), out=self._lambda3)

        # ---------------------------------------------------------------------------- #
        # Newton iterations to improve lambda3 arbitrarily, if needed, (B needed here) #
        # ---------------------------------------------------------------------------- #
        B = Tmm + self._Amatter * See  # B is only needed for N_Newton >= 1
        for _ in range(self.n_newton):
            torch.mul(
                (self._lambda3 * self._lambda3 * (self._lambda3 + self._lambda3 - A) + C),
                torch.reciprocal(self._lambda3 * (2 * (self._lambda3 - A) + self._lambda3) + B),
                out=self._lambda3
            )

        # ------------------- #
        # Get  Delta lambda's #
        # ------------------- #
        torch.sub(A, self._lambda3, out=self._tmp)
        torch.sqrt(self._tmp * self._tmp - 4 * C * torch.reciprocal(self._lambda3), out=self._Dlambda21)
        torch.mul(0.5, (A - self._lambda3 + self._Dlambda21), out=self._lambda2)
        torch.sub(self._lambda3, self._lambda2, out=self._Dlambda32)
        torch.add(self._Dlambda32, self._Dlambda21, out=self._Dlambda31)

        # ----------------------- #
        # Use Rosetta for Veisq's #
        # ----------------------- #
        # denominators
        torch.reciprocal(self._Dlambda31 * self._Dlambda32 * self._Dlambda21, out=self._PiDlambdaInv)
        torch.mul(self._PiDlambdaInv, self._Dlambda21, out=self._Xp3)
        torch.mul(-self._PiDlambdaInv, self._Dlambda31, out=self._Xp2)

        # numerators - reuse pre-allocated tensors
        torch.mul((self._lambda3 * (self._lambda3 - See) + Tee), self._Xp3, out=self._Ue3sq)
        torch.mul((self._lambda2 * (self._lambda2 - See) + Tee), self._Xp2, out=self._Ue2sq)

        Smm = (
            A
            - dmsq21 * Um2sq
            - dmsq31 * Um3sq
        )
        Tmm = Tmm * (1 - Um3sq - Um2sq) + self._Amatter * (See + Smm - A)

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
        torch.sub(1 - self._Ue3sq, self._Ue2sq, out=self._Ue1sq)
        torch.sub(1 - self._Um3sq, self._Um2sq, out=self._Um1sq)

        torch.sub(1 - self._Um3sq, self._Ue3sq, out=self._Ut3sq)
        torch.sub(1 - self._Um2sq, self._Ue2sq, out=self._Ut2sq)
        torch.sub(1 - self._Um1sq, self._Ue1sq, out=self._Ut1sq)

        # ----------------------- #
        # Get the kinematic terms #
        # ----------------------- #
        torch.mul(self._Dlambda21, self._Lover4E, out=self._D21)
        torch.mul(self._Dlambda32, self._Lover4E, out=self._D32)

        # Use pre-allocated stacks for vectorized computation
        torch.add(self._D32, self._D21, out=self._tmp)  # D31 = D32 + D21
        
        # Fill pre-allocated D stack (no new memory allocation)
        self._D_stack[:, 0] = self._D21
        self._D_stack[:, 1] = self._tmp  # D31
        self._D_stack[:, 2] = self._D32
        
        # Vectorized sine computation using pre-allocated stack
        torch.sin(self._D_stack, out=self._sin_stack)
        
        # Extract sine results (views into the pre-allocated stack - no copy)
        self._sinD21 = self._sin_stack[:, 0]
        self._sinD31 = self._sin_stack[:, 1] 
        self._sinD32 = self._sin_stack[:, 2]

        torch.mul(self._sinD21 * self._sinD31, self._sinD32, out=self._triple_sin)

        # Vectorized square operations using pre-allocated stack
        torch.square(self._sin_stack, out=self._sinsq_stack)
        self._sinsq_stack.mul_(2)  # Multiply all by 2 in one operation
        
        # Extract squared results (views into the pre-allocated stack)
        self._sinsqD21_2 = self._sinsq_stack[:, 0]
        self._sinsqD31_2 = self._sinsq_stack[:, 1]
        self._sinsqD32_2 = self._sinsq_stack[:, 2]

        # ------------------------------------------------------------------- #
        # Calculate the three necessary probabilities, separating CPC and CPV #
        # ------------------------------------------------------------------- #
        
        # Vectorize the pme_CPC calculation - compute all three terms at once
        Ut_terms = torch.stack([
            self._Ut3sq - self._Um2sq * self._Ue1sq - self._Um1sq * self._Ue2sq,
            self._Ut2sq - self._Um3sq * self._Ue1sq - self._Um1sq * self._Ue3sq,
            self._Ut1sq - self._Um3sq * self._Ue2sq - self._Um2sq * self._Ue3sq
        ], dim=-1)
        
        sinsq_stack = torch.stack([self._sinsqD21_2, self._sinsqD31_2, self._sinsqD32_2], dim=-1)
        
        # Single vectorized multiplication and sum
        torch.sum(Ut_terms * sinsq_stack, dim=-1, out=self._pme_CPC)
        
        torch.mul(-Jmatter, self._triple_sin, out=self._pme_CPV)

        # Vectorize pmm calculation
        Um_terms = torch.stack([
            self._Um2sq * self._Um1sq,
            self._Um3sq * self._Um1sq, 
            self._Um3sq * self._Um2sq
        ], dim=-1)
        
        torch.sub(1, 2 * torch.sum(Um_terms * sinsq_stack, dim=-1), out=self._pmm)

        # Vectorize pee calculation  
        Ue_terms = torch.stack([
            self._Ue2sq * self._Ue1sq,
            self._Ue3sq * self._Ue1sq,
            self._Ue3sq * self._Ue2sq
        ], dim=-1)
        
        torch.sub(1, 2 * torch.sum(Ue_terms * sinsq_stack, dim=-1), out=self._pee)

        # ---------------------------- #
        # Assign all the probabilities #
        # ---------------------------- #
        torch.sub(self._pme_CPC, self._pme_CPV, out=self._pem)
        torch.add(self._pme_CPC, self._pme_CPV, out=self._pme)

        # Vectorized probability assignment - batch all torch.where operations
        weights = self.current_weights
        
        # Create a stacked mask tensor and probability tensor for vectorized assignment
        if self._calc_tau:
            # Pre-compute tau probabilities to avoid rsub in torch.where
            torch.sub(1, self._pee, out=self._pet)
            torch.sub(self._pet, self._pem, out=self._pet)  # pet = 1 - pee - pem
            
            torch.sub(1, self._pme, out=self._pmt) 
            torch.sub(self._pmt, self._pmm, out=self._pmt)  # pmt = 1 - pme - pmm
            
            torch.sub(1, self._pem, out=self._ptm)
            torch.sub(self._ptm, self._pmm, out=self._ptm)  # ptm = 1 - pem - pmm
            
            torch.sub(1, self._pee, out=self._pte)
            torch.sub(self._pte, self._pme, out=self._pte)  # pte = 1 - pee - pme
            
            torch.sub(1, self._pet, out=self._ptt)
            torch.sub(self._ptt, self._pmt, out=self._ptt)  # ptt = 1 - pet - pmt
            
            # Stack all masks and probabilities for vectorized assignment
            mask_stack = torch.stack([
                self._mask_em, self._mask_me, self._mask_mm, self._mask_ee,
                self._mask_et, self._mask_mt, self._mask_tm, self._mask_te, self._mask_tt
            ], dim=0)
            
            prob_stack = torch.stack([
                self._pem, self._pme, self._pmm, self._pee,
                self._pet, self._pmt, self._ptm, self._pte, self._ptt
            ], dim=0)
            
            # Vectorized assignment - apply all masks at once
            for i in range(mask_stack.shape[0]):
                torch.where(mask_stack[i], prob_stack[i], weights, out=weights)
        else:
            # Non-tau case - simpler vectorization
            mask_stack = torch.stack([
                self._mask_em, self._mask_me, self._mask_mm, self._mask_ee
            ], dim=0)
            
            prob_stack = torch.stack([
                self._pem, self._pme, self._pmm, self._pee
            ], dim=0)
            
            for i in range(4):
                torch.where(mask_stack[i], prob_stack[i], weights, out=weights)

        self.current_weights = weights

        if self.current_weights.max() > 1 or self.current_weights.min() < 0:
            raise MagpyProbabilityException(
                "Oscillation probabilities must be in the range [0, 1]. "
            )

        return self.current_weights