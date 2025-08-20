from pathlib import Path
import pytest
import jax.numpy as jnp
from typing import Optional

# Enable 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)

from magpy.file_io.mc_file import MCFile
from magpy.objects.mc_event import MCEventIndices, MCEvent, MCEventMonolith

class MCTest:
    mc_file_path = Path(__file__).parent / "data" / "NuWro_FlatTree.root"
    tree_name = "FlatTree_VARS"
    
    N_MONO_EVENTS = 186191
    mc_file: Optional[MCFile] = None

    def test_open_mc_file(self):
        '''Can we open JAX MC file?'''
        self.mc_file = MCFile(str(self.mc_file_path), self.tree_name)
        assert self.mc_file is not None
        
    def test_check_branches(self):
        '''Are the branches being set correctly?'''
        if not hasattr(self, 'mc_file') or self.mc_file is None:
            self.mc_file = MCFile(str(self.mc_file_path), self.tree_name)
        
        # Set branches to read
        self.mc_file.set_mc_branch(MCEventIndices.TRUE_NEUTRINO_ENERGY, "Enu_true")
        self.mc_file.set_mc_branch(MCEventIndices.TRUE_Q2, "Q2")
        self.mc_file.set_mc_const(MCEventIndices.RECO_NEUTRINO_ENERGY, 0)
        self.mc_file.set_mc_branch(MCEventIndices.INTERACTION_MODE, "Mode")
        self.mc_file.set_mc_branch(MCEventIndices.TARGET, "tgt")
        self.mc_file.set_mc_const(MCEventIndices.START_NU, 14)  # Use constant for muon neutrino
        self.mc_file.set_mc_const(MCEventIndices.END_NU, 14)    # Use constant for muon neutrino
        self.mc_file.set_mc_const(MCEventIndices.WEIGHT, 1)
        
        # Check that branches are set
        for i, branch_name in enumerate(self.mc_file._event_branches_names):
            assert branch_name != "", f"Branch {i} not set"

    def test_monolith_creation(self):
        '''Can we create a monolith?'''
        if self.mc_file is None:
            self.test_check_branches()  # This sets up the file
        
        assert self.mc_file is not None, "MC file should be initialized"
        
        # Fill the monolith
        self.mc_file.fill_monolith()
        
        # Check that monolith was created
        assert self.mc_file._monolith is not None
        
        # Check monolith properties
        monolith = self.mc_file.monolith
        assert isinstance(monolith, MCEventMonolith)
        assert monolith.n_events > 0
        
        # Check that the monolith array is a JAX array
        assert isinstance(monolith.monolith, jnp.ndarray)
        
        # Check dimensions
        assert monolith.monolith.ndim == 2
        assert monolith.monolith.shape[1] == MCEventIndices.NENTRIES.value

    def test_monolith_data_integrity(self):
        '''Test that the monolith contains reasonable data'''
        if self.mc_file is None or self.mc_file._monolith is None:
            self.test_monolith_creation()
        
        assert self.mc_file is not None, "MC file should be initialized"
        monolith = self.mc_file.monolith
        
        # Check that energies are reasonable
        energies = monolith.monolith[:, MCEventIndices.TRUE_NEUTRINO_ENERGY.value]
        assert jnp.all(energies > 0), "All energies should be positive"
        assert jnp.all(energies < 1000), "Energies should be reasonable (< 1000 GeV)"
        
        # Check Q2 values
        q2_values = monolith.monolith[:, MCEventIndices.TRUE_Q2.value]
        assert jnp.all(q2_values >= 0), "Q2 values should be non-negative"
        
        # Check weights
        weights = monolith.monolith[:, MCEventIndices.WEIGHT.value]
        assert jnp.all(weights == 1.0), "All weights should be 1.0 (as set by constant)"

    def test_individual_event_creation(self):
        '''Test creating individual JAX MC events'''
        event = MCEvent(
            true_neutrino_energy=1.5,
            true_q2=0.5,
            reco_neutrino_energy=1.4,
            interaction_mode=1,
            start_nu=14,
            end_nu=14,
            target=1000000001,
            weight=1.0
        )
        
        # Convert to array
        event_array = event.to_array()
        assert isinstance(event_array, jnp.ndarray)
        assert len(event_array) == MCEventIndices.NENTRIES.value
        
        # Check values
        assert event_array[MCEventIndices.TRUE_NEUTRINO_ENERGY.value] == 1.5
        assert event_array[MCEventIndices.TRUE_Q2.value] == 0.5
        assert event_array[MCEventIndices.WEIGHT.value] == 1.0

    def test_monolith_from_events(self):
        '''Test creating monolith from individual events'''
        events = []
        for i in range(10):
            event = MCEvent(
                true_neutrino_energy=1.0 + i * 0.1,
                true_q2=0.1 + i * 0.01,
                reco_neutrino_energy=1.0 + i * 0.1,
                interaction_mode=1,
                start_nu=14,
                end_nu=14,
                target=1000000001,
                weight=1.0
            )
            events.append(event)
        
        monolith = MCEventMonolith(events)
        
        assert monolith.n_events == 10
        assert isinstance(monolith.monolith, jnp.ndarray)
        assert monolith.monolith.shape == (10, MCEventIndices.NENTRIES.value)
        
        # Check that energies increase as expected
        energies = monolith.monolith[:, MCEventIndices.TRUE_NEUTRINO_ENERGY.value]
        expected_energies = jnp.array([1.0 + i * 0.1 for i in range(10)])
        assert jnp.allclose(energies, expected_energies)


# Standalone test functions for pytest
def test_mc_event_creation():
    '''Test JAX MC event creation'''
    test = MCTest()
    test.test_individual_event_creation()

def test_mc_monolith_creation():
    '''Test JAX MC monolith creation'''
    test = MCTest()
    test.test_monolith_from_events()

@pytest.mark.skipif(not Path(__file__).parent.joinpath("data", "NuWro_FlatTree.root").exists(), 
                   reason="Test data file not available")
def test_mc_file_loading():
    '''Test JAX MC file loading (requires test data)'''
    test = MCTest()
    test.test_open_mc_file()
    test.test_check_branches()

@pytest.mark.skipif(not Path(__file__).parent.joinpath("data", "NuWro_FlatTree.root").exists(), 
                   reason="Test data file not available")
def test_mc_file_monolith():
    '''Test JAX MC file monolith creation (requires test data)'''
    test = MCTest()
    test.test_monolith_creation()
    test.test_monolith_data_integrity()

def test_performance_comparison():
    '''Test performance of JAX implementation'''
    import time
    
    # Create a large number of events
    n_events = 10000
    events = []
    for i in range(n_events):
        event = MCEvent(
            true_neutrino_energy=1.0 + i * 0.0001,
            true_q2=0.1 + i * 0.00001,
            reco_neutrino_energy=1.0 + i * 0.0001,
            interaction_mode=1,
            start_nu=14,
            end_nu=14,
            target=1000000001,
            weight=1.0
        )
        events.append(event)
    
    # Time monolith creation
    start_time = time.perf_counter()
    monolith = MCEventMonolith(events)
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    print(f"JAX MCEventMonolith creation: {elapsed*1000:.3f}ms for {n_events} events")
    
    # Verify result
    assert monolith.n_events == n_events
    assert isinstance(monolith.monolith, jnp.ndarray)


if __name__ == "__main__":
    # Run basic tests
    test = MCTest()
    test.test_individual_event_creation()
    test.test_monolith_from_events()
    
    # Run performance test
    test_performance_comparison()
    
    # Run pytest for comprehensive testing
    pytest.main([__file__])
