# MAGPY: JAX-Accelerated Neutrino Physics Framework
"""
High-performance neutrino oscillation framework powered by JAX.

This module provides JAX-accelerated implementations for:
- Neutrino oscillation probability calculations
- MC event handling and reweighting  
- Spline-based systematic uncertainties
- File I/O for ROOT-based neutrino data

All implementations provide high-performance JAX-accelerated computations
while maintaining clean, intuitive APIs.
"""

# Core JAX implementations (default)
from magpy.oscillator.oscillator import Oscillator
from magpy.objects.mc_event import MCEvent, MCEventMonolith, MCEventIndices
from magpy.objects.spline_handler import Spline, SplineMonolith
from magpy.file_io.mc_file import MCFile
from magpy.file_io.spline_file import SplineFile
from magpy.file_io.systematic_file import SystematicFile, SystematicHandler
from magpy.models.sample_model import SampleModel
from magpy.models.spline_syst_model import SplineSystematicModel
from magpy.utils.bin_handler import BinHandler
from magpy.Exceptions import MagpyException, MagpySplineException
import numpy as np
pi = np.pi

# Default aliases for compatibility - no longer needed since classes have original names
# Oscillator = SimpleJAXOscillator
# MCEvent = JAXMCEvent
# MCEventMonolith = JAXMCEventMonolith
# Spline = JAXSpline
# SplineMonolith = JAXSplineMonolith
# MCFile = JAXMCFile
# SplineFile = JAXSplineFile
# SampleModel = JAXSampleModel
# SplineSystematicModel = JAXSplineSystematicModel
# BinHandler = JAXBinHandler

# Version and metadata

# Performance note
import jax
jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision by default

__all__ = [
    # Core neutrino physics classes
    'MCEvent', 'MCEventMonolith', 'MCEventIndices',
    'Spline', 'SplineMonolith',
    'MCFile', 'SplineFile',
    'SampleModel', 'SplineSystematicModel',
    'BinHandler',
    # Oscillator classes
    'Oscillator',
    # Utilities and file handling
    'SystematicFile', 'SystematicHandler',
    # Constants and exceptions
    'pi', 'MagpyException', 'MagpySplineException',
]
