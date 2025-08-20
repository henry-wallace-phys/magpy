"""
Series of tests of JAX nu oscillator
"""

import jax.numpy as jnp
import pytest
import numpy as np

# Enable 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)

from magpy.oscillator.oscillator import Oscillator

from magpy.oscillator.nu_types import NuType

# Energies
# Now this is done can run tests

# Expected outputs:

TEST_DATA = [
    (
        0.31,
        0.02,
        0.55,
        0.7 * np.pi,
        7.5e-5,
        2.5e-3,
        jnp.array(
            [0.9598862328001443, 0.025676407527673743, 
             0.014437359672181967, 0.9268887954768323,
             0.01206345199427121, 0.06104775252889645,
             0.0024712562438703543, 0.4027676101665143,
             0.5947611335896154, 0.04578768284281947, 
             0.4062516001460377, 0.5479607170111429,
             0.03764251095598536, 0.571555982305812,
             0.3908015067382027, 0.02732352168034819,
             0.5816849478596912, 0.3909915304599606,
             0.8831064067109113, 0.08220067407098837,
             0.03469291921810029, 0.9714637292473465,
             0.011164541494939489, 0.017371729257714037,
             0.055281434094292134, 0.1818123615901669,
             0.762906204315541, 0.024013341497295105,
             0.18766387659674955, 0.7883227819059554,
             0.061612159194796526, 0.7359869643388447,
             0.20240087646635874, 0.00452292925535842,
             0.801171581908311, 0.1943054888363306],
            dtype=jnp.float64,
        ),
    ),
    (
        0.30,
        0.021,
        0.45,
        0,
        7.51e-5,
        2.51e-3,
        jnp.array(
                [0.9570817334818662, 0.032293704371769205,
                 0.010624562146364591, 0.9234525763484688,
                 0.0528154188947389, 0.023732004756792324,
                 0.032293704371769205, 0.3725708125831583,
                 0.5951354830450725, 0.0528154188947389,
                 0.36992934436092506, 0.577255236744336,
                 0.010624562146364591, 0.5951354830450725,
                 0.394239954808563, 0.023732004756792324,
                 0.577255236744336, 0.39901275849887163,
                 0.8781966185017407, 0.04678615594002696,
                 0.07501722555823234, 0.9706989055596977,
                 0.009658351023792847, 0.01964274341650942,
                 0.04678615594002696, 0.21198425682958466,
                 0.7412295872303883, 0.009658351023792847,
                 0.20139548139420016, 0.788946167582007,
                 0.07501722555823234, 0.7412295872303883,
                 0.18375318721137934, 0.01964274341650942,
                 0.788946167582007, 0.19141108900148363],
            dtype=jnp.float64,
        ),
    ),
    (
        0.30,
        0.021,
        0.50,
        np.pi,
        7.51e-5,
        2.51e-3,
        jnp.array(
            [0.9570817334818662, 0.009625281286105123,
             0.03329298523202867, 0.9234525763484688,
             0.02100248325692104, 0.05554494039461018,
             0.00962528128610512, 0.38697837582755235,
             0.6033963428863426, 0.021002483256921046,
             0.39532843577121535, 0.5836690809718637,
             0.03329298523202868, 0.6033963428863426,
             0.3633106718816287, 0.05554494039461018,
             0.5836690809718637, 0.3607859786335261,
             0.8781966185017407, 0.06926815623789784,
             0.05253522526036147, 0.9706989055596977,
             0.018497624584948653, 0.010803469855353614,
             0.06926815623789784, 0.18233650020008507,
             0.7483953435620171, 0.018497624584948653,
             0.18539537941104356, 0.7961069960040078,
             0.05253522526036147, 0.7483953435620171,
             0.19906943117762144, 0.010803469855353614,
             0.7961069960040078, 0.19308953414063856],
            dtype=jnp.float64,
        ),
    ),
]



def test_oscillator_creation():
    """Test that the JAX oscillator can be created"""
    osc = Oscillator(L=1300, ye=0.5, rho=3.0, n_newton=1000)
    assert osc.L == 1300
    assert osc.ye == 0.5
    assert osc.rho == 3.0
    assert osc.n_newton == 1000


def test_oscillator_energy_setting():
    """Test setting energies in the oscillator"""
    osc = Oscillator(L=1300, ye=0.5, rho=3.0, n_newton=1000)
    
    energies = jnp.linspace(0.1, 10.0, 100)
    start_nu = jnp.full(100, 14)  # muon neutrino
    end_nu = jnp.full(100, 14)    # muon neutrino
    
    osc.set_energy_osc(energies, start_nu, end_nu)
    
    # Check that oscillator is properly initialized
    # We can't access private attributes, so just test that calc_probability works
    params = jnp.array([0.31, 0.025, 0.56, 0.0, 7.5e-5, 2.5e-3])
    probs = osc.calc_probability(params)
    assert len(probs) == len(energies)


@pytest.mark.parametrize("s12, s13, s23, delta_cp, dmsq21, dmsq31, expected", TEST_DATA)
def test_oscillator_probabilities(s12, s13, s23, delta_cp, dmsq21, dmsq31, expected):
    """Test oscillation probability calculations against expected values"""
    osc = Oscillator(L=1300, ye=0.5, rho=3.0, n_newton=1000)
    
    # Set up test energies - use same range as original tests
    energies = [1.0, 2.0]

    osc_channels = [NuType.ELECTRON.value, -1*NuType.ELECTRON.value,
                    NuType.MUON.value, -1* NuType.MUON.value,
                    NuType.TAU.value, -1* NuType.TAU.value]

    final_osc_chans = []
    final_energies = []
    
    osc_combs =  [(in_, out) for in_ in osc_channels for out in osc_channels if np.sign(in_) == np.sign(out)]
    
    for E_ENERGY in energies:
        final_osc_chans.extend(osc_combs)
        final_energies.extend([E_ENERGY] * len(osc_combs))

    final_osc_chans = jnp.array(final_osc_chans, dtype=jnp.int64)
    osc_in = final_osc_chans[:, 0]
    osc_out = final_osc_chans[:, 1]
    
    energies = jnp.array(final_energies, dtype=jnp.float64)
    
    osc.set_energy_osc(energies, osc_in, osc_out)
    
    # Calculate probabilities
    params = jnp.array([s12, s13, s23, delta_cp, dmsq21, dmsq31])
    probs = osc.calc_probability(params)
    
    # Check that we get reasonable probability values
    assert jnp.all(probs >= 0.0)
    assert jnp.all(probs <= 1.0)
    
    
    # Check length matches number of energies
    assert len(probs) == len(energies)    
    assert jnp.allclose(probs, expected, rtol=1e-5, atol=1e-8)


def test_oscillator_neutrino_types():
    """Test different neutrino type combinations"""
    osc = Oscillator(L=1300, ye=0.5, rho=3.0, n_newton=1000)
    
    energies = jnp.array([1.0, 2.0, 5.0])
    
    # Test mu -> e oscillation
    start_nu = jnp.full(3, 14)  # muon neutrino
    end_nu = jnp.full(3, 12)    # electron neutrino
    
    osc.set_energy_osc(energies, start_nu, end_nu)
    
    params = jnp.array([0.31, 0.025, 0.56, 0.0, 7.5e-5, 2.5e-3])
    probs = osc.calc_probability(params)
    
    assert jnp.all(probs >= 0.0)
    assert jnp.all(probs <= 1.0)


def test_oscillator_consistency():
    """Test that multiple calculations with same parameters give same results"""
    osc = Oscillator(L=1300, ye=0.5, rho=3.0, n_newton=1000)
    
    energies = jnp.linspace(0.5, 10.0, 50)
    start_nu = jnp.full(50, 14)
    end_nu = jnp.full(50, 14)
    
    osc.set_energy_osc(energies, start_nu, end_nu)
    
    params = jnp.array([0.31, 0.025, 0.56, 0.0, 7.5e-5, 2.5e-3])
    
    # Calculate twice
    probs1 = osc.calc_probability(params)
    probs2 = osc.calc_probability(params)
    
    # Should be identical
    assert jnp.allclose(probs1, probs2, rtol=1e-15)


def test_oscillator_performance():
    """Test that oscillator performs well with large arrays"""
    osc = Oscillator(L=1300, ye=0.5, rho=3.0, n_newton=1000)
    
    # Large array test
    n_events = 10000
    energies = jnp.linspace(0.1, 100.0, n_events)
    start_nu = jnp.full(n_events, 14)
    end_nu = jnp.full(n_events, 14)
    
    osc.set_energy_osc(energies, start_nu, end_nu)
    
    params = jnp.array([0.31, 0.025, 0.56, 0.0, 7.5e-5, 2.5e-3])
    
    import time
    start_time = time.perf_counter()
    probs = osc.calc_probability(params)
    end_time = time.perf_counter()
    
    # Check results
    assert len(probs) == n_events
    assert jnp.all(probs >= 0.0)
    assert jnp.all(probs <= 1.0)
    
    # Performance check - should be much faster than 1ms for 10k events
    elapsed = end_time - start_time
    print(f"JAX Oscillator: {elapsed*1000:.3f}ms for {n_events} events")
    assert elapsed < 0.1  # Should be much faster than 100ms


if __name__ == "__main__":
    pytest.main([__file__])
