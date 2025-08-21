"""
A series of tests for JAX spline code
"""
from pathlib import Path

import pytest
import jax.numpy as jnp

# Enable 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)

from magpy.models.spline_syst_model import SplineSystematicModel
from magpy.file_io.spline_file import SplineFile
from magpy.file_io.systematic_file import SystematicFile
from magpy.objects.mc_event import MCEventMonolith, MCEvent, MCEventIndices


class TestSplineSyst:
    '''
    Series of tests for spline-syst model
    '''
    spline_file_path = Path(__file__).parent / "data" / "converted_splines.root"
    syst_file_path = Path(__file__).parent / "data" / "syst_file.yml"

        
    spline_file = SplineFile(str(spline_file_path))
    assert spline_file is not None, "Spline file should be loaded successfully"
    syst_file = SystematicFile(str(syst_file_path))
    
    handler = syst_file.systematic_handler
    model = SplineSystematicModel(spline_file, handler)


    def test_spline_mc_monolith(self):
        """
        Check if we can retrieve the correct spline indices for a monolith event
        """
        assert self.handler is not None, "Handler should be initialized"
        assert self.model is not None, "Model should be initialized"

        mc = [
            MCEvent(
                true_neutrino_energy=0.1,
                true_q2=0.1,
                reco_neutrino_energy=0.5,
                interaction_mode=0,
                start_nu=12,
                end_nu=14,
                target=0,
            ),
            MCEvent(
                true_neutrino_energy=3.5,
                true_q2=2.0,
                reco_neutrino_energy=2.5,
                interaction_mode=2,
                start_nu=12,
                end_nu=14,
                target=0,
            ),
        ]

        mc_mono = MCEventMonolith(mc_event_list=mc)
        expected_rows = [0, 82]

        result = self.model.get_monolith_splines(
            mc_mono,
            jnp.array(
                [
                    MCEventIndices.TRUE_NEUTRINO_ENERGY.value,
                    MCEventIndices.RECO_NEUTRINO_ENERGY.value,
                    MCEventIndices.DUMMY.value,
                ],
            )
        )
        
        
        assert (
            expected_rows == list(result)
        ), f"Spline indices are {result} but expected {expected_rows}"


if __name__ == "__main__":
    test = TestSplineSyst()
    test.test_file_opening()
    pytest.main([__file__])