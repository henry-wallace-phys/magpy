"""
A series of tests for spline code
"""
from pathlib import Path

import torch

from magpy.objects.spline_handler import SplineMonolith, Spline
from magpy.file_io.spline_file import SplineFile
from magpy.models.spline_syst_model import SplineSystematicModel
from magpy.objects.systematic_handler import SystematicHandler, Systematic

# Test file loading
def test_flat_spline():
    """Check behaviour for flat splines"""
    spline = Spline(
        torch.tensor([0, 1, 2, 3], dtype=torch.float64),
        torch.tensor([0, 0, 0, 0], dtype=torch.float64),
    )
    expected = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float64)

    assert spline.is_flat
    assert torch.isclose(spline.spline, expected, rtol=1e-8, atol=1e-8).all(), f"Expected {expected}, but got {spline.spline}"


def test_non_flat_spline():
    """Check behaviour for non-flat splines"""
    spline = Spline(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 1.0, 2.0],
        ],
        dtype=torch.float64,
    )

    assert not spline.is_flat
    assert torch.isclose(spline.spline, expected, rtol=1e-8, atol=1e-8).all(), f"Expected {expected}, but got {spline.spline}"


def test_monolith():
    """Check behaviour for spline monoliths"""
    flat_example = Spline(
        torch.tensor([0, 1, 2, 3], dtype=torch.float64),
        torch.tensor([0, 0, 0, 0], dtype=torch.float64),
    )
    spline = Spline(
        torch.tensor([0, 1, 2, 3], dtype=torch.float64),
        torch.tensor([0, 1, 2, 3], dtype=torch.float64),
    )

    spline_monolith = SplineMonolith([flat_example, spline])
    # Check indexing is done

    print(spline_monolith[1])
    print(spline.spline)

    assert torch.isclose(
        spline_monolith[0], flat_example.spline, rtol=1e-8, atol=1e-8
    ).all(), f"Expected {flat_example.spline}, but got {spline_monolith[0]}"
    assert torch.isclose(spline_monolith[1], spline.spline, rtol=1e-8, atol=1e-8).all(), f"Expected {spline.spline}, but got {spline_monolith[1]}"

    x_vals = torch.tensor([10, 1.5])
    assert torch.isclose(
        spline_monolith(x_vals),
        torch.tensor([1, 1.5], dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    ).all(), f"Expected {[1, 1.5]}, but got {spline_monolith(x_vals)}"


def test_spline_file_open():
    # Get spline file in tests/data
    spline_file_path = Path(__file__).parent / "data" / "converted_splines.root"

    spline_file = SplineFile(spline_file_path)

    assert spline_file is not None, "Spline file should be loaded successfully"
    assert len(spline_file.spline_names) == 250, f"Spline file should contain 250 spline names, instead got {len(spline_file.spline_names)}"
    assert len(spline_file.monolith) == 250, f"Spline monolith should contain 250 splines, instead got {len(spline_file.monolith)}"

    assert spline_file.get_bin_handler() is not None, "Bin handler should be available"
    
def test_spline_model():
    spline_file_path = Path(__file__).parent / "data" / "converted_splines.root"
    spline_file = SplineFile(spline_file_path)

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
        )
    ]

    handler = SystematicHandler(systs)
    model = SplineSystematicModel(spline_file, handler)

    assert model is not None, "Model should be created successfully"
    assert len(model.index_tensor) == 250, f"Model index tensor should have 250 elements, instead got {len(model.index_tensor)}"
    assert model.index_tensor.shape[1] == 6, f"Model index tensor should have 6 columns, instead got {model.index_tensor.shape[1]}"
    assert torch.equal(model.index_tensor[0], torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int64)), f"First row of index tensor should be [0, 0, 0, 0, 0, 0], instead got {model.index_tensor[0]}"