"""
Tensorified version of nufast:
 https://github.com/PeterDenton/NuFast-LBL/tree/main
 Based on arXiv:2405.02400
"""

from enum import Enum
from typing import List

import torch
from magpy.Exceptions import MagpyProbabilityException
from magpy.objects.mc_event import MCEventMonolith, MCEventIndices

from magpy.utils.device_manager import DeviceManager

class NuType(Enum):
    E = 12
    EBar = -12
    Mu = 14
    MuBar = -14
    Tau = 16
    TauBar = -16

    def __str__(self):
        return self.name

class InternalNuType(Enum):
    E = 0
    Mu = 1
    Tau =2

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
    """
    Oscillation through matter
    """

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

        # Now to set up masks for oscillations in/out
        self._masks = [[] for _ in range(3)]
        for idx, i in enumerate([NuType.E, NuType.Mu, NuType.Tau]):
            for j in [NuType.E, NuType.Mu, NuType.Tau]:
                mask = (self._osc_in == i.value) & (self._osc_out == j.value)
                self._masks[idx].append(mask)

        self._calc_tau = torch.any(self._osc_in == NuType.Tau.value) or torch.any(self._osc_in == NuType.TauBar.value)

        # Might as well do constants now    
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

        s13sq = osc_params[OscParIndex.S13SQ.value]
        dmsq21 = osc_params[OscParIndex.DMSQ21.value]
        dmsq31 = osc_params[OscParIndex.DMSQ31.value]
        delta_cp = osc_params[OscParIndex.DELTA.value]
        s12sq = osc_params[OscParIndex.S12SQ.value]
        s23sq = osc_params[OscParIndex.S23SQ.value]

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

        # if torch.isnan(Jrr):
        #     raise MagpyProbabilityException(
        #         f"Jrr is NaN, check oscillation parameters Um2sq: {Um2sq}, {Ut2sq}."
        #     )

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
        xmat = self._Amatter / Dmsqee
        tmp = 1 - xmat
        lambda3 = dmsq31 + 0.5 * Dmsqee * (
            xmat
            - 1
            + torch.sqrt(tmp * tmp + 4 * s13sq * xmat)
        )

        # ---------------------------------------------------------------------------- #
        # Newton iterations to improve lambda3 arbitrarily, if needed, (B needed here) #
        # ---------------------------------------------------------------------------- #
        B = Tmm + self._Amatter * See  # B is only needed for N_Newton >= 1
        for _ in range(self.n_newton):
            lambda3 = (lambda3 * lambda3 * (lambda3 + lambda3 - A) + C) * torch.reciprocal(
                lambda3 * (2 * (lambda3 - A) + lambda3) + B
            )  # this strange form prefers additions to multiplications

        # ------------------- #
        # Get  Delta lambda's #
        # ------------------- #
        tmp = A - lambda3
        Dlambda21 = torch.sqrt(tmp * tmp - 4 * C * torch.reciprocal(lambda3))
        lambda2 = 0.5 * (A - lambda3 + Dlambda21)
        Dlambda32 = lambda3 - lambda2
        Dlambda31 = Dlambda32 + Dlambda21

        # ----------------------- #
        # Use Rosetta for Veisq's #
        # ----------------------- #
        # denominators
        PiDlambdaInv = torch.reciprocal(Dlambda31 * Dlambda32 * Dlambda21)
        Xp3 = PiDlambdaInv * Dlambda21
        Xp2 = -PiDlambdaInv * Dlambda31

        # numerators
        Ue3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3
        Ue2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2

        Smm = (
            A
            - dmsq21 * Um2sq
            - dmsq31 * Um3sq
        )
        Tmm = Tmm * (1 - Um3sq - Um2sq) + self._Amatter * (See + Smm - A)

        Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3
        Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2

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
            * PiDlambdaInv
        )

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

        D21 = Dlambda21 * self._Lover4E
        D32 = Dlambda32 * self._Lover4E

        sinD21 = torch.sin(D21)
        sinD31 = torch.sin(D32 + D21)
        sinD32 = torch.sin(D32)

        triple_sin = sinD21 * sinD31 * sinD32

        sinsqD21_2 = 2 * torch.square(sinD21)
        sinsqD31_2 = 2 * torch.square(sinD31)
        sinsqD32_2 = 2 * torch.square(sinD32)

        # ------------------------------------------------------------------- #
        # Calculate the three necessary probabilities, separating CPC and CPV #
        # ------------------------------------------------------------------- #
        pme_CPC = (
            (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2
            + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2
            + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2
        )
        pme_CPV = -Jmatter * triple_sin


        # p m->m
        pmm = 1 - 2 * (
            Um2sq * Um1sq * sinsqD21_2
            + Um3sq * Um1sq * sinsqD31_2
            + Um3sq * Um2sq * sinsqD32_2
        )

        pee= 1 - 2 * (
            Ue2sq * Ue1sq * sinsqD21_2
            + Ue3sq * Ue1sq * sinsqD31_2
            + Ue3sq * Ue2sq * sinsqD32_2
        )

        # ---------------------------- #
        # Assign all the probabilities #
        # ---------------------------- #
        # probs_returned = torch.zeros((len(E), 3, 3), device=E.device, dtype=E.dtype)
        # Direct assignment without redundant indexing

        pem = pme_CPC - pme_CPV
        pme = pme_CPC + pme_CPV

        self.current_weights[self._masks[InternalNuType.E.value][InternalNuType.Mu.value]] = pem[self._masks[InternalNuType.E.value][InternalNuType.Mu.value]]
        self.current_weights[self._masks[InternalNuType.Mu.value][InternalNuType.E.value]] = pme[self._masks[InternalNuType.Mu.value][InternalNuType.E.value]]
        self.current_weights[self._masks[InternalNuType.Mu.value][InternalNuType.Mu.value]] = pmm[self._masks[InternalNuType.Mu.value][InternalNuType.Mu.value]]
        self.current_weights[self._masks[InternalNuType.E.value][InternalNuType.E.value]] = pee[self._masks[InternalNuType.E.value][InternalNuType.E.value]]


        if self._calc_tau:
            pet = 1 - pee - pem
            pmt = 1 - pme - pmm
            pte = 1 - pee - pme
            ptm = 1 - pem - pmm
            ptt = 1 - pet - pmt
            self.current_weights[self._masks[InternalNuType.Mu.value][InternalNuType.Tau.value]] = pmt[self._masks[InternalNuType.Mu.value][InternalNuType.Tau.value]]
            self.current_weights[self._masks[InternalNuType.E.value][InternalNuType.Tau.value]] = pet[self._masks[InternalNuType.E.value][InternalNuType.Tau.value]]
            self.current_weights[self._masks[InternalNuType.Tau.value][InternalNuType.Mu.value]] = ptm[self._masks[InternalNuType.Tau.value][InternalNuType.Mu.value]]
            self.current_weights[self._masks[InternalNuType.Tau.value][InternalNuType.Tau.value]] = ptt[self._masks[InternalNuType.Tau.value][InternalNuType.Tau.value]]
            self.current_weights[self._masks[InternalNuType.E.value][InternalNuType.Tau.value]] = pet[self._masks[InternalNuType.E.value][InternalNuType.Tau.value]]
            self.current_weights[self._masks[InternalNuType.Tau.value][InternalNuType.E.value]] = pte[self._masks[InternalNuType.Tau.value][InternalNuType.E.value]]


        if self.current_weights.max() > 1 or self.current_weights.min() < 0:
            raise MagpyProbabilityException(
                "Oscillation probabilities must be in the range [0, 1]. "
            )

        return self.current_weights