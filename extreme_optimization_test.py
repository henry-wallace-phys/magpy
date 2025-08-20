#!/usr/bin/env python3
"""
Extreme optimization test for sub-millisecond reweight performance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import jax.numpy as jnp
from magpy.benchmarking.benchmark_reweight import benchmark_reweight
from magpy.models.spline_syst_model import SplineSystematicModel
from magpy.file_io.spline_file import SplineFile
from magpy.file_io.systematic_file import SystematicFile
from magpy.file_io.mc_file import MCFile
from magpy.objects.mc_event import MCEventIndices

def enable_extreme_optimizations():
    """Enable all extreme optimizations for sub-millisecond performance"""
    print("🚀 Enabling extreme optimizations...")
    
    # Load the model components
    spline_file_path = "src/magpy/tests/data/converted_splines.root"
    systematic_file_path = "src/magpy/tests/data/syst_file.yml"
    mc_file_path = "src/magpy/tests/data/test_mc_file.root"
    
    spline_file = SplineFile(spline_file_path)
    systematic_file = SystematicFile(systematic_file_path)
    mc_file = MCFile(mc_file_path)
    
    # Get bin handler
    bin_variables = [MCEventIndices.TRUE_NEUTRINO_ENERGY.value, MCEventIndices.RECO_NEUTRINO_ENERGY.value, MCEventIndices.DUMMY.value]
    
    # Create the model
    model = SplineSystematicModel(spline_file, systematic_file)
    model.setup_bins(mc_file, bin_variables)
    
    # Enable extreme fusion optimizations
    model.enable_extreme_fusion()
    print("✅ Extreme fusion enabled in SplineSystematicModel")
    
    return model

def test_extreme_performance():
    """Test extreme optimization performance"""
    print("⚡ Testing extreme optimization performance...")
    
    # Quick benchmark with extreme optimizations
    times = benchmark_reweight(25)  # Shorter test first
    avg_time_ms = np.mean(times[2:]) * 1000
    
    print(f"Average time with extreme optimizations: {avg_time_ms:.3f}ms per reweight")
    print(f"Target: <1ms per reweight")
    
    if avg_time_ms < 1.0:
        print("🎉 SUCCESS: Sub-millisecond performance achieved!")
        return True
    else:
        improvement_needed = avg_time_ms / 1.0
        print(f"⚠️  Still {improvement_needed:.1f}x slower than target")
        return False

def run_extreme_benchmark():
    """Run the full benchmark with extreme optimizations"""
    print("🔥 Running full extreme optimization benchmark...")
    
    # Enable optimizations first
    model = enable_extreme_optimizations()
    
    # Test performance
    success = test_extreme_performance()
    
    if success:
        print("Running full 100-iteration benchmark...")
        times = benchmark_reweight(100)
        avg_time_ms = np.mean(times[2:]) * 1000
        print(f"Full benchmark average: {avg_time_ms:.3f}ms per reweight")
    
    return success

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 EXTREME OPTIMIZATION TEST FOR SUB-MS PERFORMANCE 🚀")
    print("=" * 60)
    
    try:
        success = run_extreme_benchmark()
        
        if success:
            print("🎉 EXTREME OPTIMIZATION SUCCESS! 🎉")
        else:
            print("⚠️  More optimization needed for sub-ms target")
            
    except Exception as e:
        print(f"❌ Error during extreme optimization test: {e}")
        import traceback
        traceback.print_exc()
