import jax.numpy as jnp
import pytest
from magpy.utils.bin_handler import BinHandler

# Enable 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)

TEST_KINEMATIC_INDICES = jnp.array([0.1, 2, 100, 1], dtype=jnp.float32)
EXPECTED_BIN_INDICES = jnp.array([0, 2, 5, 1], dtype=jnp.int32)

EXPECTED_BINS = jnp.array(
    [
        [[0.0000, 1.5000], [2.5000, 3.5000], [0.0000, 1.5000]],
        [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]],
        [[0.0000, 1.5000], [1.5000, 2.5000], [0.0000, 1.5000]],
    ],
    dtype=jnp.float64,
)


def test_bin_index_conversion():
    """
    Test JAX bin handler's ability to test indices
    """
    bin_edges = [[0, 1.5, 2.5, 3.5], [0, 1.5, 2.5, 3.5, 4.5, 5.5], [0, 1.5]]

    indices = jnp.array([0, 1, 2])

    expected_bins = jnp.array(
        [
            [[0.0000, 1.5000], [0.0000, 1.5000], [0.0000, 1.5000]],
            [[0.0000, 1.5000], [1.5000, 2.5000], [0.0000, 1.5000]],
            [[0.0000, 1.5000], [2.5000, 3.5000], [0.0000, 1.5000]],
        ],
        dtype=jnp.float64,
    )

    bin_handler = BinHandler(bin_edges)
    bins = bin_handler.get_bin_from_int(indices)


    assert jnp.allclose(bins, expected_bins, rtol=1e-5)


def test_kinematic_bin_finding():
    """
    Test that the kinematic bin finding works correctly
    """
    bin_edges = [[0, 1.5, 2.5, 3.5], [0, 1.5, 2.5, 3.5, 4.5, 5.5], [0, 1.5]]

    bin_handler = BinHandler(bin_edges)

    kinematic = jnp.array([0.1, 2.0, 1.0], dtype=jnp.float64)
    expected_indices = jnp.array([[0, 1, 0]], dtype=jnp.int32)

    bin_indices = bin_handler.find_bin(kinematic.reshape(1, -1))
    assert jnp.array_equal(bin_indices, expected_indices)


def test_kinematic_bin_finding_multiple():
    """
    Test kinematic bin finding with multiple events
    """
    bin_edges = [[0, 1.5, 2.5, 3.5], [0, 1.5, 2.5, 3.5, 4.5, 5.5], [0, 1.5]]

    bin_handler = BinHandler(bin_edges)

    # Test multiple events
    test_kinematics = jnp.array([
        [0.1, 2.0, 1.0],
        [2.0, 4.0, 0.5],
        [3.0, 5.0, 1.2]
    ], dtype=jnp.float64)
    
    expected_indices = jnp.array([
        [0, 1, 0],
        [1, 3, 0], 
        [2, 4, 0]
    ], dtype=jnp.int32)

    bin_indices = bin_handler.find_bin(test_kinematics)
    assert jnp.array_equal(bin_indices, expected_indices)


def test_bin_handler_properties():
    """
    Test that the bin handler properties work correctly
    """
    bin_edges = [[0, 1.5, 2.5, 3.5], [0, 1.5, 2.5, 3.5, 4.5, 5.5], [0, 1.5]]
    bin_handler = BinHandler(bin_edges)

    # Test that properties exist and have correct types
    assert bin_handler.bin_indices is not None
    assert bin_handler.bin_edge_tensor is not None
    assert isinstance(bin_handler.bin_indices, jnp.ndarray)
    assert isinstance(bin_handler.bin_edge_tensor, jnp.ndarray)


def test_out_of_bounds_handling():
    """
    Test handling of out-of-bounds values
    """
    bin_edges = [[0, 1.5, 2.5, 3.5], [0, 1.5, 2.5, 3.5], [0, 1.5]]
    bin_handler = BinHandler(bin_edges)

    # Test with out of bounds index
    out_of_bounds_index = jnp.array([999])  # Way out of bounds
    
    # This should handle gracefully and return [-1, -1] arrays
    with pytest.warns(UserWarning):
        bins = bin_handler.get_bin_from_int(out_of_bounds_index)
        
    # Check that invalid bins are marked with -1
    assert jnp.all(bins[0] == -1.0)


def test_performance_large_arrays():
    """
    Test performance with large arrays
    """
    bin_edges = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]
    bin_handler = BinHandler(bin_edges)

    # Large array test
    n_events = 10000
    import numpy as np
    np.random.seed(42)  # For reproducibility
    test_kinematics = jnp.array([
        [np.random.uniform(0, 3.5), np.random.uniform(0, 5.5), np.random.uniform(0, 1.5)]
        for _ in range(n_events)
    ], dtype=jnp.float64)

    import time
    start_time = time.perf_counter()
    bin_indices = bin_handler.find_bin(test_kinematics)
    end_time = time.perf_counter()

    # Check results
    assert bin_indices.shape == (n_events, 3)
    
    # Performance check  
    elapsed = end_time - start_time
    print(f"JAX BinHandler: {elapsed*1000:.3f}ms for {n_events} events")
    assert elapsed < 0.5  # Should be reasonably fast with JAX


if __name__ == "__main__":
    pytest.main([__file__])
