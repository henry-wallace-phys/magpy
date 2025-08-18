'''
Tensorified version of nufast:
 https://github.com/PeterDenton/NuFast-LBL/tree/main
 Based on arXiv:2405.02400
'''
from enum import Enum

import torch
import numpy as np

from magpy.utils.device_manager import DeviceManager
from magpy.Exceptions import MagpyProbabilityException

class NuType(Enum):
    E = 1
    Mu = 2
    Tau = 3

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


@torch.jit.script
class Oscillator:
    '''
    Oscillation through matter
    '''
    def __init__(self, L: float, ye: float, rho: float, n_newton: int):
        self.L = L
        self.rho = rho
        self.ye = ye
        self.n_newton = n_newton

        self.current_osc_pars = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        self.current_weights = torch.tensor([], dtype=torch.float64)
        
        self.eVsqkm_to_GeV_over4 = 1e-9 / 1.97327e-7 * 1e3 / 4
        self.YerhoE2a = 1.52588e-4


    def calc_probability(self, osc_params: torch.Tensor, E: torch.Tensor, osc_in: torch.Tensor, osc_out: torch.Tensor)->torch.Tensor:
        # --------------------------------------------------------------------- #
        # First calculate useful simple functions of the oscillation parameters #
        # --------------------------------------------------------------------- #

        if osc_params.isnan().any():
            raise MagpyProbabilityException("Oscillation parameters must not be NaN.")

        if torch.equal(osc_params, self.current_osc_pars) and self.current_weights is not None:
            return self.current_weights

        
        else:
            self.current_osc_pars = osc_params

        self.current_osc_pars = osc_params



        c13sq = 1 - osc_params[OscParIndex.S13SQ.value]

        # Ueisq's
        Ue2sq = c13sq * osc_params[OscParIndex.S12SQ.value]
        Ue3sq = osc_params[OscParIndex.S13SQ.value]

        # Umisq's, Utisq's and Jvac
        Um3sq = c13sq * osc_params[OscParIndex.S23SQ.value]
        # Um2sq and Ut2sq are used here as temporary variables, will be properly defined later
        Ut2sq = osc_params[OscParIndex.S13SQ.value] * osc_params[OscParIndex.S12SQ.value] * osc_params[OscParIndex.S23SQ.value]
        Um2sq = (1 - osc_params[OscParIndex.S12SQ.value]) * (1 - osc_params[OscParIndex.S23SQ.value])

        Jrr = torch.sqrt(Um2sq * Ut2sq)
        
        if (torch.isnan(Jrr)):
            raise MagpyProbabilityException(f"Jrr is NaN, check oscillation parameters Um2sq: {Um2sq}, {Ut2sq}.")

        sind = torch.sin(osc_params[OscParIndex.DELTA.value])
        cosd = torch.cos(osc_params[OscParIndex.DELTA.value])

        Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd
        Jmatter = 8 * Jrr * c13sq * sind
        
        # With E
        Amatter = self.ye * self.rho * E * self.YerhoE2a
        Dmsqee =  osc_params[OscParIndex.DMSQ31.value] - osc_params[OscParIndex.S12SQ.value] * osc_params[OscParIndex.DMSQ21.value] # Dmsq21

        # calculate A, B, C, See, Tee, and part of Tmm
        A = osc_params[OscParIndex.DMSQ21.value] + osc_params[OscParIndex.DMSQ31.value] # temporary variable
        See = A - osc_params[OscParIndex.DMSQ21.value] * Ue2sq - osc_params[OscParIndex.DMSQ31.value] * Ue3sq
        Tmm = osc_params[OscParIndex.DMSQ21.value] * osc_params[OscParIndex.DMSQ31.value] # using Tmm as a temporary variable
        Tee = Tmm * (1 - Ue3sq - Ue2sq)
        
        # E
        C = Amatter * Tee
        A = A + Amatter

        # ---------------------------------- #
        # Get lambda3 from lambda+ of MP/DMP #
        # ---------------------------------- #
        xmat = Amatter / Dmsqee
        tmp = 1 - xmat
        lambda3 = osc_params[OscParIndex.DMSQ31.value] + 0.5 * Dmsqee * (xmat - 1 + torch.sqrt(tmp * tmp + 4 * osc_params[OscParIndex.S13SQ.value] * xmat))

        # ---------------------------------------------------------------------------- #
        # Newton iterations to improve lambda3 arbitrarily, if needed, (B needed here) #
        # ---------------------------------------------------------------------------- #
        B = Tmm + Amatter * See # B is only needed for N_Newton >= 1
        for _ in range(self.n_newton):
            lambda3 = (lambda3 * lambda3 * (lambda3 + lambda3 - A) + C) / (lambda3 * (2 * (lambda3 - A) + lambda3) + B) # this strange form prefers additions to multiplications

        # ------------------- #
        # Get  Delta lambda's #
        # ------------------- #
        tmp = A - lambda3
        Dlambda21 = torch.sqrt(tmp * tmp - 4 * C / lambda3)
        lambda2 = 0.5 * (A - lambda3 + Dlambda21)
        Dlambda32 = lambda3 - lambda2
        Dlambda31 = Dlambda32 + Dlambda21

        # ----------------------- #
        # Use Rosetta for Veisq's #
        # ----------------------- #
        # denominators	  
        PiDlambdaInv = 1 / (Dlambda31 * Dlambda32 * Dlambda21)
        Xp3 = PiDlambdaInv * Dlambda21
        Xp2 = -PiDlambdaInv * Dlambda31

        # numerators
        Ue3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3
        Ue2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2

        Smm = A - osc_params[OscParIndex.DMSQ21.value] * Um2sq - osc_params[OscParIndex.DMSQ31.value] * Um3sq
        Tmm = Tmm * (1 - Um3sq - Um2sq) + Amatter * (See + Smm - A)

        Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3
        Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2

        # ------------- #
        # Use NHS for J #
        # ------------- #
        Jmatter = Jmatter * osc_params[OscParIndex.DMSQ21.value] * osc_params[OscParIndex.DMSQ31.value] * (osc_params[OscParIndex.DMSQ31.value] - osc_params[OscParIndex.DMSQ21.value]) * PiDlambdaInv

        # ----------------------- #
        # Get all elements of Usq #
        # ----------------------- #
        Ue1sq = 1 - Ue3sq - Ue2sq
        Um1sq = 1 - Um3sq - Um2sq

        Ut3sq = 1 - Um3sq - Ue3sq
        Ut2sq = 1 - Um2sq - Ue2sq
        Ut1sq = 1 - Um1sq - Ue1sq

        # ----------------------- #
        # Get the kinematic terms #
        # ----------------------- #
        Lover4E = self.eVsqkm_to_GeV_over4 * self.L / E

        D21 = Dlambda21 * Lover4E
        D32 = Dlambda32 * Lover4E
        
        sinD21 = torch.sin(D21)
        sinD31 = torch.sin(D32 + D21)
        sinD32 = torch.sin(D32)

        triple_sin = sinD21 * sinD31 * sinD32

        sinsqD21_2 = 2 * torch.pow(sinD21, 2)
        sinsqD31_2 = 2 * torch.pow(sinD31, 2)
        sinsqD32_2 = 2 * torch.pow(sinD32, 2)

        # ------------------------------------------------------------------- #
        # Calculate the three necessary probabilities, separating CPC and CPV #
        # ------------------------------------------------------------------- #
        Pme_CPC = (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2 \
                    + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2 \
                    + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2
        Pme_CPV = -Jmatter * triple_sin

        Pmm = 1 - 2 * (Um2sq * Um1sq * sinsqD21_2 \
                    + Um3sq * Um1sq * sinsqD31_2 \
                    + Um3sq * Um2sq * sinsqD32_2)

        Pee = 1 - 2 * (Ue2sq * Ue1sq * sinsqD21_2 \
                    + Ue3sq * Ue1sq * sinsqD31_2 \
                    + Ue3sq * Ue2sq * sinsqD32_2)

        # ---------------------------- #
        # Assign all the probabilities #
        # ---------------------------- #
        # probs_returned = torch.zeros((len(E), 3, 3), device=E.device, dtype=E.dtype)
        out_prob = torch.zeros(len(E), dtype=torch.float64, device=E.device)
        # Direct assignment without redundant indexing
        Pem = Pme_CPC - Pme_CPV
        Pet = 1 - Pee - Pem
        Pme = Pme_CPC + Pme_CPV
        Pmt = 1 - Pme - Pmm
        Pte = 1 - Pee - Pme
        Ptm = 1 - Pem - Pmm
        Ptt = 1 - Pet - Pmt
        
        # Use masking to simplify
        
        out_prob[(osc_in == NuType.E.value) & (osc_out == NuType.E.value)] = Pee[(osc_in == NuType.E.value) & (osc_out == NuType.E.value)]
        out_prob[(osc_in == NuType.E.value) & (osc_out==NuType.Mu.value)] = Pem[(osc_in == NuType.E.value) & (osc_out==NuType.Mu.value)]
        out_prob[(osc_in == NuType.E.value) & (osc_out==NuType.Tau.value)] = Pet[(osc_in == NuType.E.value) & (osc_out==NuType.Tau.value)]
        out_prob[(osc_in == NuType.Mu.value) & (osc_out==NuType.E.value)] = Pme[(osc_in == NuType.Mu.value) & (osc_out==NuType.E.value)]
        out_prob[(osc_in == NuType.Mu.value) & (osc_out==NuType.Mu.value)] = Pmm[(osc_in == NuType.Mu.value) & (osc_out==NuType.Mu.value)]
        out_prob[(osc_in == NuType.Mu.value) & (osc_out==NuType.Tau.value)] = Pmt[(osc_in == NuType.Mu.value) & (osc_out==NuType.Tau.value)]
        out_prob[(osc_in == NuType.Tau.value) & (osc_out==NuType.E.value)] = Pte[(osc_in == NuType.Tau.value) & (osc_out==NuType.E.value)]
        out_prob[(osc_in == NuType.Tau.value) & (osc_out==NuType.Mu.value)] = Ptm[(osc_in == NuType.Tau.value) & (osc_out==NuType.Mu.value)]
        out_prob[(osc_in == NuType.Tau.value) & (osc_out==NuType.Tau.value)] = Ptt[(osc_in == NuType.Tau.value) & (osc_out==NuType.Tau.value)]

        if torch.any(out_prob<0) or torch.any(out_prob>1):
            raise MagpyProbabilityException("Oscillation probabilities must be in the range [0, 1]. ")
        
        self.current_weights = out_prob.clone()

        return out_prob