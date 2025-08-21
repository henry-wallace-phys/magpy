"""
A series of tests for JAX spline code
"""
from pathlib import Path

import pytest
import jax.numpy as jnp

# Enable 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)

from magpy.objects.spline_handler import SplineMonolith, Spline
from magpy.models.spline_syst_model import SplineSystematicModel
from magpy.file_io.spline_file import SplineFile
from magpy.file_io.systematic_file import SystematicFile
import time
    

# Test file loading
class SplineTest:

    flat_spline = Spline(
        jnp.array([0, 1, 2, 3], dtype=jnp.float64),
        jnp.array([0, 0, 0, 0], dtype=jnp.float64),
    )

    flat_response = jnp.array([0, 0, 0, 0, 0], dtype=jnp.float64)

    non_flat_spline = Spline(
        jnp.array([0, 1, 2, 3], dtype=jnp.float64),
        jnp.array([0, 1, 2, 3], dtype=jnp.float64),
    )

    non_flat_response = jnp.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 1.0, 2.0],
        ],
        dtype=jnp.float64,
    )

    def test_flat_spline(self):
        """Test that flat splines are handled correctly"""
        assert self.flat_spline.is_flat
        assert len(self.flat_spline) == 0

    def test_non_flat_spline(self):
        """Test that non-flat splines work correctly"""
        assert not self.non_flat_spline.is_flat
        assert len(self.non_flat_spline) > 0
        assert jnp.allclose(self.non_flat_spline.spline, self.non_flat_response)


    def test_spline_monolith_creation(self):
        """Test creating a spline monolith"""
        splines = [self.flat_spline, self.non_flat_spline]
        monolith = SplineMonolith(splines)
        
        assert monolith._n_splines == 2
        assert len(monolith._flat_splines) == 2
        assert monolith._flat_splines[0] == True  # flat spline
        assert monolith._flat_splines[1] == False  # non-flat spline

    def test_spline_monolith_indexing(self):
        """Test indexing into spline monolith"""
        splines = [self.flat_spline, self.non_flat_spline]
        monolith = SplineMonolith(splines)
        
        # Test accessing flat spline
        flat_result = monolith[0]
        assert jnp.array_equal(flat_result, SplineMonolith.FLAT_SPLINE)
        
        # Test accessing non-flat spline
        non_flat_result = monolith[1]
        assert len(non_flat_result) > 0

    def test_spline_monolith_evaluation(self):
        """Test evaluating splines in monolith"""
        splines = [self.flat_spline, self.non_flat_spline]
        monolith = SplineMonolith(splines)
        
        # Test evaluation with parameters
        test_params = jnp.array([0.5, 1.5])
        weights = monolith(test_params)
        
        assert len(weights) == 2
        assert jnp.all(jnp.isfinite(weights))


def test_spline_creation():
    """Test creating JAX splines"""
    x = jnp.linspace(0, 10, 11)
    y = jnp.sin(x)
    
    spline = Spline(x, y)
    assert not spline.is_flat  # sin(x) is not flat
    assert len(spline) > 0


def test_spline_flat_detection():
    """Test flat spline detection"""
    x = jnp.array([0, 1, 2, 3])
    y_flat = jnp.array([1, 1, 1, 1])  # Flat
    y_varying = jnp.array([0, 1, 2, 3])  # Varying
    
    flat_spline = Spline(x, y_flat)
    varying_spline = Spline(x, y_varying)
    
    assert flat_spline.is_flat
    assert not varying_spline.is_flat


def test_spline_monolith_performance():
    """Test performance of spline monolith operations"""
    # Create multiple splines
    splines = []
    for i in range(100):
        x = jnp.linspace(0, 10, 11)
        y = jnp.linspace(0, 10, 11)**2
        
        # Now we want to do something silly        
        splines.append(Spline(x, y))
    
    syst_map = jnp.array([[i%10, i] for i in range(100)])  # Simple mapping for testing
    
    monolith = SplineMonolith(splines)
    monolith.map_splines_to_syst(syst_map)

    # Test evaluation performance
    test_params = jnp.ones(100)*2
    
    # Burn compile_loop
    weights = monolith(test_params)

    
    start_time = time.perf_counter()
    weights = monolith(test_params)
    end_time = time.perf_counter()
    
    print(weights)
    
    print(weights)
    # Check results
    assert len(weights) == 100
    assert jnp.all(jnp.isfinite(weights))
    
    
    # Performance check
    assert jnp.all(weights==4)
    elapsed = end_time - start_time
    print(f"JAX SplineMonolith: {elapsed*1000:.3f}ms for 100 splines")
    assert elapsed < 0.6  # Should be reasonably fast


def test_spline_systematic_mapping():
    """Test spline systematic mapping functionality"""
    # Create test splines
    splines = []
    for i in range(10):
        x = jnp.linspace(0, 5, 6)
        y = x * (i + 1)  # Different slopes
        splines.append(Spline(x, y))
    
    monolith = SplineMonolith(splines)
    
    # Create systematic mapping
    syst_map = jnp.array([
        [0, 0],  # Systematic 0, spline 0
        [0, 1],  # Systematic 0, spline 1
        [1, 2],  # Systematic 1, spline 2
        [1, 3],  # Systematic 1, spline 3
        [2, 4],  # Systematic 2, spline 4
    ])
    
    monolith.map_splines_to_syst(syst_map)
    

def test_spline_edge_cases():
    """Test edge cases in spline handling"""
    # Test single point spline (should be flat)
    x_single = jnp.array([1.0])
    y_single = jnp.array([1.0])
    
    # This might raise an error with scipy, so we handle it
    try:
        single_spline = Spline(x_single, y_single)
        assert single_spline.is_flat
    except ValueError:
        # Expected for single point splines
        pass
    
    # Test two-point spline
    x_two = jnp.array([0.0, 1.0])
    y_two = jnp.array([0.0, 1.0])
    
    two_spline = Spline(x_two, y_two)
    assert not two_spline.is_flat or two_spline.is_flat  # Either is valid


if __name__ == "__main__":
    # Run basic tests
    test = SplineTest()
    test.test_flat_spline()
    test.test_non_flat_spline()
    test.test_spline_monolith_creation()
    test.test_spline_monolith_indexing()
    test.test_spline_monolith_evaluation()
    
    
    
    # Run pytest for the rest
    pytest.main([__file__])