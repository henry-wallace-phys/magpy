"""
A series of tests for spline code
"""

import torch
from magpy.objects.spline_handler import SplineMonolith, Spline


# Test file loading
def test_flat_spline():
    """Check behaviour for flat splines"""
    spline = Spline(
        torch.tensor([0, 1, 2, 3], dtype=torch.float64),
        torch.tensor([0, 0, 0, 0], dtype=torch.float64),
    )
    expected = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float64)

    assert spline.is_flat
    assert torch.isclose(spline.spline, expected, rtol=1e-8, atol=1e-8).all()


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
    assert torch.isclose(spline.spline, expected, rtol=1e-8, atol=1e-8).all()


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
    ).all()
    assert torch.isclose(spline_monolith[1], spline.spline, rtol=1e-8, atol=1e-8).all()

    x_vals = torch.tensor([10, 1.5])
    assert torch.isclose(
        spline_monolith(x_vals),
        torch.tensor([1, 1.5], dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    ).all()
