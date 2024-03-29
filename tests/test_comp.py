import pytest

from SiPANN.comp import racetrack_sb_rr


class Test_racetrack_sb_rr:
    def test_warnings(self):
        with pytest.warns(UserWarning):
            racetrack_sb_rr(400, 240, 5000, 90, 5000, sw_angle=90, loss=[0.99])
