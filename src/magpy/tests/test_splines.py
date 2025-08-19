"""
A series of tests for spline code
"""

from pathlib import Path

import torch
import pytest

from magpy.objects.spline_handler import SplineMonolith, Spline
from magpy.file_io.spline_file import SplineFile
from magpy.models.spline_syst_model import SplineSystematicModel
from magpy.objects.systematic_handler import SystematicHandler, Systematic
from magpy.objects.mc_event import MCEventMonolith, MCEvent, MCEventIndices


# Test file loading
class SplineTest:
    flat_spline = Spline(
        torch.tensor([0, 1, 2, 3], dtype=torch.float64),
        torch.tensor([0, 0, 0, 0], dtype=torch.float64),
    )

    flat_response = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float64)

    non_flat_spline = Spline(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))
    non_flat_response = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 1.0, 2.0],
        ],
        dtype=torch.float64,
    )

    def test_spline(self, spline: Spline, expected: torch.Tensor):
        assert torch.isclose(
            spline.spline, expected, rtol=1e-8, atol=1e-8
        ).all(), f"Expected {expected}, but got {spline.spline}"

    def test_flat_spline(self):
        """Check behaviour for flat splines"""
        assert self.flat_spline.is_flat
        self.test_spline(self.flat_spline, self.flat_response)

    def test_non_flat_spline(self):
        """Check behaviour for non-flat splines"""

        assert not self.non_flat_spline.is_flat
        self.test_spline(self.non_flat_spline, self.non_flat_response)

    def test_monolith(self):
        """Check behaviour for spline monoliths"""

        spline_monolith = SplineMonolith([self.flat_spline, self.non_flat_spline])
        # Check indexing is done
        assert torch.isclose(
            spline_monolith[0], self.flat_spline.spline, rtol=1e-8, atol=1e-8
        ).all(), f"Expected {self.flat_spline.spline}, but got {spline_monolith[0]}"
        assert torch.isclose(
            spline_monolith[1], self.non_flat_spline.spline, rtol=1e-8, atol=1e-8
        ).all(), f"Expected {self.non_flat_spline.spline}, but got {spline_monolith[1]}"

        x_vals = torch.tensor([10, 1.5])
        assert torch.isclose(
            spline_monolith(x_vals),
            torch.tensor([1, 1.5], dtype=torch.float64),
            rtol=1e-8,
            atol=1e-8,
        ).all(), f"Expected {[1, 1.5]}, but got {spline_monolith(x_vals)}"


class SplineModelTest:
    spline_file_path = Path(__file__).parent / "data" / "converted_splines.root"
    systs = [
        Systematic(
            syst_name="mysyst1",
            spline_name="mysyst1",
            modes=[0],
            nominal=1.0,
            error=0.1,
            syst_type="spline",
            range=(0.0, 1.0),
        ),
        Systematic(
            syst_name="mysyst2",
            spline_name="mysyst2",
            modes=[2],
            nominal=1.0,
            error=0.1,
            syst_type="spline",
            range=(0.0, 1.0),
        ),
        Systematic(
            syst_name="mysyst3",
            spline_name="mysyst3",
            modes=[1],
            nominal=1.0,
            error=0.1,
            syst_type="spline",
            range=(0.0, 1.0),
        ),
        Systematic(
            syst_name="mysyst4",
            spline_name="mysyst4",
            modes=[3],
            nominal=1.0,
            error=0.1,
            syst_type="spline",
            range=(0.0, 1.0),
        ),
        Systematic(
            syst_name="mysyst5",
            spline_name="mysyst5",
            modes=[4],
            nominal=1.0,
            error=0.1,
            syst_type="spline",
            range=(0.0, 1.0),
        ),
    ]

    def test_file_opening(self):
        spline_file = SplineFile(str(self.spline_file_path))
        assert spline_file is not None, "Spline file should be loaded successfully"
        handler = SystematicHandler(self.systs)
        model = SplineSystematicModel(spline_file, handler)

        assert model is not None, "Model should be created successfully"
        assert (
            len(model.index_tensor) == 250
        ), f"Model index tensor should have 250 elements, instead got {len(model.index_tensor)}"
        assert (
            model.index_tensor.shape[1] == 6
        ), f"Model index tensor should have 6 columns, instead got {model.index_tensor.shape[1]}"
        assert torch.equal(
            model.index_tensor[0], torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int64)
        ), f"First row of index tensor should be [0, 0, 0, 0, 0, 0], instead got {model.index_tensor[0]}"

    def test_spline_mc_monolith(self):
        """
        Check if we can retrieve the correct spline indices for a monolith event
        """
        spline_file = SplineFile(str(self.spline_file_path))

        handler = SystematicHandler(self.systs)
        model = SplineSystematicModel(spline_file, handler)

        mc = MCEvent(
            true_neutrino_energy=1.0,
            true_q2=2.0,
            reco_neutrino_energy=1.5,
            interaction_mode=0,
            start_nu=12,
            end_nu=14,
            target=0,
        )

        mc = [
            MCEvent(
                true_neutrino_energy=1.0,
                true_q2=2.0,
                reco_neutrino_energy=1.5,
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
        expected_rows = [torch.tensor(1), torch.tensor(82)]

        result = model.get_monolith_splines(
            mc_mono,
            torch.tensor(
                [
                    MCEventIndices.TRUE_NEUTRINO_ENERGY.value,
                    MCEventIndices.RECO_NEUTRINO_ENERGY.value,
                    MCEventIndices.DUMMY.value,
                ]
            ),
        )
        
        
        assert (
            expected_rows == list(result)
        ), f"Spline indices are {result} but expected {expected_rows}"


def test_spline_handler():
    model = SplineTest()
    model.test_flat_spline()
    model.test_non_flat_spline()
    model.test_monolith()


def test_spline_model():
    model = SplineModelTest()
    model.test_file_opening()
    model.test_spline_mc_monolith()
