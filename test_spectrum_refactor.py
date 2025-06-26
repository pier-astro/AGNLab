#!/usr/bin/env python3
"""
Quick test to verify the refactored Spectrum class works correctly.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import agnlab
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agnlab.spectrum import Spectrum

def test_spectrum_creation():
    """Test different ways to create Spectrum objects."""
    
    # Create some test data
    wave = np.linspace(4000, 7000, 100)
    flux = np.random.random(100) + 10
    fluxerr = np.random.random(100) * 0.1
    
    print("Testing Spectrum creation methods...")
    
    # Test 1: Direct initialization
    spec1 = Spectrum(wave=wave, flux=flux, fluxerr=fluxerr, name="test_direct")
    print(f"✓ Direct init: {spec1.name}, wave range: {spec1.wave[0]:.1f}-{spec1.wave[-1]:.1f}")
    
    # Test 2: Class method from_arrays (equivalent to old make_spectrum)
    spec2 = Spectrum.from_arrays(wave, flux, fluxerr, name="test_from_arrays")
    print(f"✓ from_arrays: {spec2.name}, wave range: {spec2.wave[0]:.1f}-{spec2.wave[-1]:.1f}")
    
    # Test 3: Create test file and read it (equivalent to old read_txt)
    test_file = "test_spectrum.txt"
    data = np.column_stack([wave, flux, fluxerr])
    np.savetxt(test_file, data)
    
    spec3 = Spectrum.from_txt(test_file, name="test_from_file")
    print(f"✓ from_txt: {spec3.name}, wave range: {spec3.wave[0]:.1f}-{spec3.wave[-1]:.1f}")
    
    # Test 4: Empty initialization (for backward compatibility)
    spec4 = Spectrum()
    print(f"✓ Empty init: {spec4.name}")
    
    # Cleanup
    os.remove(test_file)
    
    print("\nAll tests passed! ✅")
    print("\nMigration guide:")
    print("Old: spec = make_spectrum(wave, flux, fluxerr)")
    print("New: spec = Spectrum(wave, flux, fluxerr)")
    print("  or: spec = Spectrum.from_arrays(wave, flux, fluxerr)")
    print()
    print("Old: spec = read_txt('file.txt')")
    print("New: spec = Spectrum.from_txt('file.txt')")

if __name__ == "__main__":
    test_spectrum_creation()
