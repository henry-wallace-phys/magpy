# Central model for handling everything...
from typing import List, Optional

import torch
from cProfile import Profile
from pstats import SortKey, Stats


from magpy.file_io.mc_file import MCFile
from magpy.file_io.spline_file import SplineFile
from magpy.file_io.systematic_file import SystematicFile
from magpy.objects.mc_event import MCEventIndices
from magpy.models.spline_syst_model import SplineSystematicModel
from magpy.oscillator.oscillator import Oscillator
from magpy.Exceptions import MagpyInvalidObjectError

class SampleModel:
    def __init__(self, mc_file: MCFile, spline_file: SplineFile, systematic_file: SystematicFile, oscillator: Oscillator, bin_variables: Optional[List[MCEventIndices]] = None):
        """
        Handle the full event model, including MC events, splines and systematics.
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
        self.oscillator.set_energy_osc(self.mc_monolith[:, MCEventIndices.TRUE_NEUTRINO_ENERGY.value],
                                       self.mc_monolith[:, MCEventIndices.START_NU.value],
                                       self.mc_monolith[:, MCEventIndices.END_NU.value])

    def set_bin_variables(self, events: List[MCEventIndices] | torch.Tensor):
        if isinstance(events, torch.Tensor):
            self._bin_variables = events
        else:
            self._bin_variables = torch.tensor([e.value for e in events])

    def initialise_mc_indices(self):
        if self._bin_variables is None:
            raise MagpyInvalidObjectError("Bin variables not set. Please set bin variables before initialising MC indices.")

        self._mc_indices = self.spline_syst_handler.get_monolith_splines(self.mc_file.monolith, self._bin_variables)
        
    @property
    def mc_indices(self) -> List[torch.Tensor]:
        return self._mc_indices
    
    @property
    def mc_monolith(self)-> torch.Tensor:
        return self.mc_file.monolith.monolith
    
    def reweight(self, osc_pars, syst_pars) -> torch.Tensor:
        """
        Reweight the MC events based on the spline systematics.
        """
        if self._mc_indices is None:
            raise MagpyInvalidObjectError("MC indices not initialised. Please initialise MC indices before reweighting.")
        
        self.mc_monolith[:, MCEventIndices.WEIGHT.value].fill_(1.0)  # Reset weights to 1.0
        
        self.mc_monolith[:, MCEventIndices.WEIGHT.value]  = self.oscillator.calc_probability(osc_params=osc_pars)
        self.spline_syst_handler.reweight(syst_pars, self.mc_monolith)
 
        return self.mc_monolith
    
    def profile(self, osc_pars, syst_pars, n_iter: int = 1000):
        """
        Profile the reweighting process.
        """
        
        
        with Profile() as pr:
            for _ in  tqdm(range(n_iter)):
                pr.disable()
                osc_mod = osc_pars.clone() * (
                        1.0001 + (np.random.uniform(0, 1) / (n_iter * 100))
                )    
                pr.enable()
                self.oscillator.calc_probability(osc_params=osc_mod)

            stats = Stats(pr)
            stats.strip_dirs()
            stats.sort_stats(SortKey.CUMULATIVE)
            stats.print_stats()

if __name__=="__main__":
    from tqdm import tqdm
    import time
    import numpy as np
    
    SPLINE_FILE_PATH = "/Users/henrywallace/software/magpy/src/magpy/tests/data/converted_splines.root"
    MC_FILE_PATH = "/Users/henrywallace/software/magpy/src/magpy/tests/data/NuWro_FlatTree.root"
    SYST_FILE_PATH='/Users/henrywallace/software/magpy/src/magpy/tests/data/syst_file.yml'

    systematic_file = SystematicFile(SYST_FILE_PATH)
    mc_file = MCFile(MC_FILE_PATH, "FlatTree_VARS")
    
    mc_file.set_mc_branch(MCEventIndices.TRUE_NEUTRINO_ENERGY, "Enu_true")
    mc_file.set_mc_branch(MCEventIndices.TRUE_Q2, "Q2")
    mc_file.set_mc_const(MCEventIndices.RECO_NEUTRINO_ENERGY, 0)
    mc_file.set_mc_branch(MCEventIndices.INTERACTION_MODE, "Mode")
    mc_file.set_mc_branch(MCEventIndices.TARGET, "tgt")
    # consts
    mc_file.set_mc_const(MCEventIndices.START_NU, 12)
    mc_file.set_mc_const(MCEventIndices.END_NU, 14)
    mc_file.set_mc_const(MCEventIndices.WEIGHT, 0)
    mc_file.fill_monolith()
    
    spline_file = SplineFile(SPLINE_FILE_PATH)
    oscillator = Oscillator(1300, 0.5, 3, 0)

    handler = SampleModel(mc_file, spline_file, systematic_file, oscillator=oscillator)

    bin_vars = torch.tensor([MCEventIndices.TRUE_NEUTRINO_ENERGY.value,
                             MCEventIndices.TRUE_Q2.value,
                             MCEventIndices.DUMMY.value])

    handler.set_bin_variables(bin_vars)
    print("Setting up indices...")
    handler.initialise_mc_indices()
    print("Setting up tensors")
    osc_reweight = torch.tensor([0.31,0.02, 0.55, 0.7 * torch.pi, 7.5e-5, 2.5e-3])
    syst_vals = torch.tensor([1.1, 1.1, 1.1, 1.1, 1.1])

    print("Reweighting")
    n_iter = 1000
    times = np.zeros(n_iter)
    
    handler.profile(osc_reweight, syst_vals, n_iter=n_iter)
    
    

    for i in (pbar:= tqdm(range(n_iter), total=n_iter, desc="Reweighting iterations")):
        # Randomize osc
        osc_mod = osc_reweight.clone() * (
                1.0001 + (np.random.uniform(0, 1) / (n_iter * 100))
            )
    
        syst_mod = syst_vals.clone() * (
                1.0001 + (np.random.uniform(0, 1) / (n_iter * 100))
            )
        
        start = time.time()
        handler.reweight(osc_mod, syst_vals)
        stop = time.time()

        times[i] = stop - start

        if i%100 ==0 and i>0:
            pbar.set_postfix({"Avg Time": f"{np.mean(times[i-100:i]*1000):.8f}±{np.std(times[i-100:i]*1000):.8f}ms"})

    print(f"Average time/reweight: {np.mean(times)*1000:.8f}±{np.std(times)*1000:.8f}ms")