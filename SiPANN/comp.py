import warnings
from abc import ABC, abstractmethod

import gdstk
import numpy as np
import pkg_resources
import scipy.integrate as integrate
import scipy.special as special

from SiPANN.import_nn import ImportLR
from SiPANN.nn import bentWaveguide, straightWaveguide
from SiPANN.scee import HalfRacetrack, clean_inputs, get_coeffs

"""
Similarly to before, we initialize all ANN's and regressions as global objects to speed things up.
"""
cross_file = pkg_resources.resource_filename("SiPANN", "LR/DC_coeffs.pkl")
DC_coeffs = ImportLR(cross_file)


class racetrack_sb_rr:
    """Racetrack waveguide arc used to connect to a racetrack directional coupler."""

    def __init__(self, width, thickness, radius, gap, length, sw_angle=90, loss=[0.99]):
        self.width = width
        self.thickness = thickness
        self.radius = radius
        self.gap = gap
        self.length = length
        self.sw_angle = sw_angle
        self.loss = loss
        if np.any(self.width < 400):
            warnings.warn("Width is less than 400nm, may produce invalid results", UserWarning)
        if np.any(self.width > 600):
            warnings.warn("Width is greater than 600nm, may produce invalid results", UserWarning)
        if np.any(self.thickness < 180):
            warnings.warn("Thickness is less than 180nm, may produce invalid results", UserWarning)
        if np.any(self.thickness > 240):
            warnings.warn("Thickness is greater than 240nm, may produce invalid results", UserWarning)
        if np.any(self.sw_angle < 80):
            warnings.warn("Sidewall Angle is less than 80 degrees, may produce invalid results", UserWarning)
        if np.any(self.sw_angle < 90):
            warnings.warn("Sidewall Angle is greater than 90 degrees, may produce invalid results", UserWarning)
        if np.any(self.gap < 100):
            warnings.warn("Gap is less than 100nm, may produce invalid results", UserWarning)

    def update(self, **kwargs):
        self.width = kwargs.get("width", self.width)
        self.thickness = kwargs.get("thickness", self.thickness)
        self.radius = kwargs.get("radius", self.radius)
        self.gap = kwargs.get("gap", self.gap)
        self.length = kwargs.get("length", self.length)
        self.sw_angle = kwargs.get("sw_angle", self.sw_angle)
        # Re-trigger warnings (omitted for brevity)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs(
                (
                    self.width,
                    self.thickness,
                    self.sw_angle,
                    self.radius,
                    self.gap,
                    self.length,
                )
            )
        else:
            return clean_inputs(
                (
                    wavelength,
                    self.width,
                    self.thickness,
                    self.sw_angle,
                    self.radius,
                    self.gap,
                    self.length,
                )
            )

    def predict(self, wavelength):
        if np.any(wavelength < 1450):
            warnings.warn("Wavelength is less than 1450nm", UserWarning)
        if np.any(wavelength > 1650):
            warnings.warn("Wavelength is greater than 1650nm", UserWarning)

        wavelength, width, thickness, sw_angle, radius, gap, length = self._clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        rr = HalfRacetrack(self.width, self.thickness, self.radius, self.gap, self.length)
        t = rr.predict((1, 3), wavelength)

        # pull phase from coupler section
        phi_t = -np.unwrap(np.angle(t))
        t_mag = np.abs(t)

        # pull phase from bent sections
        TE0_B = np.squeeze(
            bentWaveguide(
                wavelength=wavelength * 1e-3,
                width=self.width * 1e-3,
                thickness=self.thickness * 1e-3,
                sw_angle=self.sw_angle,
                radius=self.radius * 1e-3,
            )
        )
        L_b = np.pi * radius
        phi_b = np.unwrap(2 * np.pi * np.real(TE0_B) / wavelength) * (L_b)

        # pull phase from straight sections
        TE0 = np.squeeze(
            straightWaveguide(
                wavelength=wavelength * 1e-3,
                width=self.width * 1e-3,
                thickness=self.thickness * 1e-3,
                sw_angle=self.sw_angle,
            )
        )
        L_s = length
        phi_s = np.unwrap(2 * np.pi * np.real(TE0) / wavelength) * L_s

        phi = phi_t + phi_b + phi_s
        lossPoly = np.poly1d(self.loss)
        alpha = lossPoly(wavelength)

        E = (
            (t_mag - alpha * np.exp(1j * phi))
            / (1 - alpha * t_mag * np.exp(1j * phi))
            * np.exp(-1j * phi)
        )

        return E, alpha, t, phi

    def sparams(self, wavelength):
        n = 1 if np.isscalar(wavelength) else len(wavelength)
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, getting sparams doesn't make sense")
        s_matrix = np.zeros((2, 2, n), dtype="complex")
        s_matrix[0, 1] = self.predict(wavelength)[0]
        s_matrix += np.transpose(s_matrix, (1, 0, 2))
        s_matrix = np.transpose(s_matrix, (2, 0, 1))
        return s_matrix

    def gds(self, filename=None, view=False, extra=0, units="nms"):
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == "nms":
            scale = 1e-3
        elif units == "microns":
            scale = 1
        else:
            raise ValueError("Invalid units")

        sc_radius = self.radius * scale
        sc_gap = self.gap * scale
        sc_width = self.width * scale
        sc_length = self.length * scale

        # --- GDSTK IMPLEMENTATION ---
        pathTop = gdstk.FlexPath(
            (-sc_length / 2, 2 * sc_radius + sc_width / 2 + sc_gap / 2), sc_width
        )
        pathTop.horizontal(sc_length, relative=True)
        pathTop.turn(sc_radius, -np.pi)
        pathTop.horizontal(-sc_length, relative=True)
        pathTop.turn(sc_radius, -np.pi)

        pathBottom = gdstk.FlexPath(
            (-sc_radius - sc_width / 2 - sc_length / 2, -sc_gap / 2 - sc_width / 2), sc_width
        )
        pathBottom.horizontal(2 * (sc_radius + sc_width / 2) + sc_length, relative=True)

        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        path_cell = lib.new_cell("C0")
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if filename is not None:
            lib.write_gds(filename)
        elif view:
             print("View not supported in Gdstk. Please save to GDS file.")