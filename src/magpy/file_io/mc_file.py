from typing import Optional

import pandas as pd
from tqdm import tqdm
import uproot

from magpy.file_io.root_file import RootFile
from magpy.objects.mc_event import MCEvent, MCEventMonolith, MCEventIndices
from magpy.Exceptions import MagpyInvalidObjectError, MagpyModeNotFoundException
from magpy.utils.modes import SplineModes
import numpy as np


class MCFile(RootFile):
    CONST_BRANCH="__CONST__"
    
    def __init__(self, file_name: str, tree_name: str = "mc_tree") -> None:
        super().__init__(file_name)
        self.mc_tree = self.get_root_object(tree_name)

        if not isinstance(self.mc_tree, uproot.TTree):
            raise MagpyInvalidObjectError(f"ROOT object '{tree_name}' is not a TTree.")
        
        self._event_branches_names = [""]*MCEventIndices.NENTRIES.value

        self._const_vals = [None]*MCEventIndices.NENTRIES.value

        self._monolith = None

    # Now we need to set branch information
    def set_mc_const(self, mc_event_idx: MCEventIndices, value: Optional[float]) -> None:
        self._event_branches_names[mc_event_idx.value] = self.CONST_BRANCH
        self._const_vals[mc_event_idx.value] = value
        
    def set_mc_branch(self, mc_event_idx: MCEventIndices, branch_name: str) -> None:
        self._event_branches_names[mc_event_idx.value] = branch_name
        
    def fill_monolith(self):
        if "" in self._event_branches_names:
            raise MagpyInvalidObjectError("Not all branches are set for MCEvent.")
        
        branch_list = [i for i in self._event_branches_names if i != self.CONST_BRANCH]
        
        if branch_list:
            self._mc_event_monolith = self.mc_tree.arrays(branch_list, library="pd")
        else:
            raise MagpyInvalidObjectError("No branches set for MCEvent, cannot fill monolith.")
            
        # Now we fill the monolith
        monolith_list = []

        print("Filling MC Event monolith...")
        for i, event in tqdm(self._mc_event_monolith.iterrows(), total=len(self._mc_event_monolith)):
            
            # Get the info
            try:
                true_energy = self.__const_or_branch(event, MCEventIndices.TRUE_NEUTRINO_ENERGY)
                true_q2 = self.__const_or_branch(event, MCEventIndices.TRUE_Q2)
                reco_energy = self.__const_or_branch(event, MCEventIndices.RECO_NEUTRINO_ENERGY)
                interaction_mode = SplineModes.from_generator_mode(self.__const_or_branch(event, MCEventIndices.INTERACTION_MODE)).value
                start_nu = self.__const_or_branch(event, MCEventIndices.START_NU)
                end_nu = self.__const_or_branch(event, MCEventIndices.END_NU)
                target = self.__const_or_branch(event, MCEventIndices.TARGET)
                weight = self.__const_or_branch(event, MCEventIndices.WEIGHT)

                mono_entry = MCEvent(
                    true_neutrino_energy=true_energy,
                    true_q2=true_q2,
                    reco_neutrino_energy=reco_energy,
                    interaction_mode=interaction_mode,
                    start_nu=start_nu,
                    end_nu=end_nu,
                    target=target,
                    weight=weight
                )
                
            except MagpyModeNotFoundException:
                continue
            except Exception as e:
                raise MagpyInvalidObjectError(f"Error processing event: {e}") from e
            monolith_list.append(mono_entry) 

        self._monolith = MCEventMonolith(monolith_list)

    def __const_or_branch(self, event: pd.DataFrame, event_idx: MCEventIndices):
        if self._event_branches_names[event_idx.value] == self.CONST_BRANCH:
            return self._const_vals[event_idx.value]
        
        return event.loc[self._event_branches_names[event_idx.value]]
    
    @property
    def monolith(self)->MCEventMonolith:
        """Get the monolith of MC events"""
        if self._monolith is None:
            raise MagpyInvalidObjectError("Monolith has not been filled yet.")
        return self._monolith