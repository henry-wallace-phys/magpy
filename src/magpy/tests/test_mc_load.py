from pathlib import Path

from magpy.file_io.mc_file import MCFile
from magpy.objects.mc_event import MCEventIndices, MCEvent, MCEventMonolith

class MCTest:
    mc_file_path = Path(__file__).parent / "data" / "NuWro_FlatTree.root"
    tree_name = "FlatTree_VARS"
    
    N_MONO_EVENTS = 186191

    def test_open_mc_file(self):
        '''Can we open?'''
        self.mc_file = MCFile(str(self.mc_file_path), self.tree_name)
        assert self.mc_file is not None
        
    def test_check_branches(self):
        '''Are the branches being set?'''
        if self.mc_file is None:
            raise AssertionError("MCFile not opened correctly.")
        
        self.mc_file = MCFile(str(self.mc_file_path), self.tree_name)
        
        # Set branches to read
        self.mc_file.set_mc_branch(MCEventIndices.TRUE_NEUTRINO_ENERGY, "Enu_true")
        self.mc_file.set_mc_branch(MCEventIndices.TRUE_Q2, "Q2")
        self.mc_file.set_mc_const(MCEventIndices.RECO_NEUTRINO_ENERGY, 0)
        self.mc_file.set_mc_branch(MCEventIndices.INTERACTION_MODE, "Mode")
        self.mc_file.set_mc_branch(MCEventIndices.TARGET, "tgt")

        # Get consts
        self.mc_file.set_mc_const(MCEventIndices.START_NU, 12)
        self.mc_file.set_mc_const(MCEventIndices.END_NU, 14)
        self.mc_file.set_mc_const(MCEventIndices.WEIGHT, 0)

        expected_out = ["Enu_true", "Q2", "__CONST__", "Mode", "__CONST__","__CONST__", "tgt", "__CONST__"]

        assert self.mc_file._event_branches_names == expected_out, f"Expected {expected_out} but got {self.mc_file._event_branches_names}"
        
        
    def test_fill_monolith(self):
        '''Has the monolith filled?'''
        if self.mc_file is None:
            raise AssertionError("MCFile not opened correctly.")
        self.mc_file.fill_monolith()
        assert self.mc_file.monolith is not None, "Monolith should not be None"
        assert len(self.mc_file.monolith) == self.N_MONO_EVENTS, f"Expected {self.N_MONO_EVENTS} events but got {len(self.mc_file.monolith)}"
        
def test_monolith():
    test = MCTest()
    test.test_open_mc_file()
    test.test_check_branches()
    test.test_fill_monolith()