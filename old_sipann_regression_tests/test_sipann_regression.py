import pytest
import numpy as np
import os
import sys

# Attempt to import SiPANN modules
try:
    from SiPANN import scee, nn, comp
except ImportError:
    # Fallback if running from a non-installed environment
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from SiPANN import scee, nn, comp

# ==============================================================================
# REFERENCE VALUES
# ==============================================================================
# These were all calculated using SiPANN 2.0.1, running on python 3.8.10 
EXPECTED_VALUES = {
    'straight_wg': np.array([[[[1.18345623e+12]]], [[[1.25738308e+12]]], [[[1.33256200e+12]]]]),
    'coupler_te0': np.array([[[[[-5.67750029e+13]]]], [[[[-6.32611102e+13]]]], [[[[-7.02837425e+13]]]]]),
    'coupler_te1': np.array([[[[[-5.96289576e+13]]]], [[[[-6.60933089e+13]]]], [[[[-7.30440309e+13]]]]]), 
    'gap_func_sym': np.array([ 0.55100644+0.07212481j, 0.55724723-0.33364082j, -0.00173838-0.75385275j]),
    'half_ring': np.array([-0.06544974+0.05906779j, -0.02405997-0.10990032j, 0.11702874-0.08420384j])
}

class TestSiPANNRegression:
    """
    Regression tests to ensure SiPANN v2 (GDSTK/Vectorized) matches 
    SiPANN v1 (GDSPY/Quad).
    """

    # Shared Test Parameters
    WAVELENGTHS = np.array([1500.0, 1550.0, 1600.0])
    WIDTH = 500.0
    THICKNESS = 220.0
    SW_ANGLE = 90.0

    def _assert_match(self, actual, key, name):
        """
        Helper method to handle regression comparison.
        If expected is None, it fails intentionally to provide the value.
        """
        expected = EXPECTED_VALUES.get(key)

        if expected is None:
            # Format the numpy array for easy copy-pasting
            formatted_val = np.array2string(actual, separator=', ', precision=8, suppress_small=True)
            msg = (
                f"\n\n[MISSING REFERENCE VALUE] for: {name}\n"
                f"--------------------------------------------------------\n"
                f"Copy this value into EXPECTED_VALUES['{key}']:\n\n"
                f"np.array({formatted_val})\n"
                f"--------------------------------------------------------\n"
            )
            pytest.fail(msg)
        else:
            # Verify values match within tolerance
            np.testing.assert_allclose(
                actual, 
                expected, 
                rtol=1e-5, 
                atol=1e-7, 
                err_msg=f"REGRESSION FAILURE on {name}: Values deviated from old model."
            )

    def test_nn_straight_waveguide(self):
        """Checks Neural Network output for straight waveguides."""
        actual = nn.straightWaveguide(self.WAVELENGTHS, self.WIDTH, self.THICKNESS, self.SW_ANGLE)
        self._assert_match(actual, 'straight_wg', "NN Straight Waveguide")

    def test_nn_evanescent_coupler_te0(self):
        """Checks TE0 output for evanescent couplers."""
        gap = 200.0
        te0, _ = nn.evWGcoupler(self.WAVELENGTHS, self.WIDTH, self.THICKNESS, gap, self.SW_ANGLE)
        self._assert_match(te0, 'coupler_te0', "NN Coupler TE0")

    def test_nn_evanescent_coupler_te1(self):
        """Checks TE1 output for evanescent couplers (Separated so it runs even if TE0 fails)."""
        gap = 200.0
        _, te1 = nn.evWGcoupler(self.WAVELENGTHS, self.WIDTH, self.THICKNESS, gap, self.SW_ANGLE)
        self._assert_match(te1, 'coupler_te1', "NN Coupler TE1")

    def test_scee_integration_logic(self):
        """CRITICAL TEST: Verifies integration logic."""
        gap_fn = lambda z: 100.0 + 0.01 * z
        dgap_fn = lambda z: 0.01 + 0.0 * z
        zmin, zmax = 0.0, 10000.0

        dev = scee.GapFuncSymmetric(self.WIDTH, self.THICKNESS, gap_fn, dgap_fn, zmin, zmax, self.SW_ANGLE)
        actual = dev.predict((1, 4), self.WAVELENGTHS)
        self._assert_match(actual, 'gap_func_sym', "SCEE Integration (GapFuncSymmetric)")

    def test_scee_analytic_half_ring(self):
        """Checks analytic solutions."""
        radius = 10000.0
        gap = 200.0
        
        dev = scee.HalfRing(self.WIDTH, self.THICKNESS, radius, gap, self.SW_ANGLE)
        actual = dev.predict((1, 4), self.WAVELENGTHS)
        self._assert_match(actual, 'half_ring', "SCEE HalfRing Analytic")

    #@pytest.mark.skip(reason="Smoke test for GDS generation only.")
    def test_gdstk_generation_smoke(self, tmp_path):
        filename = str(tmp_path / "test_output.gds")
        length = 10000.0
        gap = 200.0
        dev = scee.StraightCoupler(self.WIDTH, self.THICKNESS, gap, length, self.SW_ANGLE)
        try:
            dev.gds(filename)
            assert os.path.exists(filename)
        except Exception as e:
            pytest.fail(f"GDS generation failed: {e}")