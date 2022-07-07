import warnings
from abc import ABC, abstractmethod

import gdspy
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
    """Racetrack waveguide arc used to connect to a racetrack directional
    coupler. Ports labeled as::

        |           -------         |
        |         /         \       |
        |         \         /       |
        |           -------         |
        |   1 ----------------- 2   |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm
        thickness : float or ndarray
            Thickness of waveguide in nm
        radius : float or ndarray
            Distance from center of ring to middle of waveguide in nm.
        gap : float or ndarray
            Minimum distance from ring waveguide edge to straight waveguide edge in nm.
        length : float or ndarray
            Length of straight portion of ring waveguide in nm.
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """

    def __init__(self, width, thickness, radius, gap, length, sw_angle=90, loss=[0.99]):
        self.width = width
        self.thickness = thickness
        self.radius = radius
        self.gap = gap
        self.length = length
        self.sw_angle = sw_angle
        self.loss = loss
        if np.any(self.width < 400):
            warnings.warn(
                "Width is less than 400nm, may produce invalid results", UserWarning
            )
        if np.any(self.width > 600):
            warnings.warn(
                "Width is greater than 600nm, may produce invalid results", UserWarning
            )
        if np.any(self.thickness < 180):
            warnings.warn(
                "Thickness is less than 180nm, may produce invalid results", UserWarning
            )
        if np.any(self.thickness > 240):
            warnings.warn(
                "Thickness is greater than 240nm, may produce invalid results",
                UserWarning,
            )
        if np.any(self.sw_angle < 80):
            warnings.warn(
                "Sidewall Angle is less than 80 degrees, may produce invalid results",
                UserWarning,
            )
        if np.any(self.sw_angle < 90):
            warnings.warn(
                "Sidewall Angle is greater than 90 degrees, may produce invalid results",
                UserWarning,
            )
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def update(self, **kwargs):
        """Takes in any parameter defined by __init__ and changes it.

        Parameters
        ----------
        attribute : float or ndarray
            Included if any device needs to have an attribute changed.
        """
        self.width = kwargs.get("width", self.width)
        self.thickness = kwargs.get("thickness", self.thickness)
        self.radius = kwargs.get("radius", self.radius)
        self.gap = kwargs.get("gap", self.gap)
        self.length = kwargs.get("length", self.length)
        self.sw_angle = kwargs.get("sw_angle", self.sw_angle)
        if np.any(self.width < 400):
            warnings.warn(
                "Width is less than 400nm, may produce invalid results", UserWarning
            )
        if np.any(self.width > 600):
            warnings.warn(
                "Width is greater than 600nm, may produce invalid results", UserWarning
            )
        if np.any(self.thickness < 180):
            warnings.warn(
                "Thickness is less than 180nm, may produce invalid results", UserWarning
            )
        if np.any(self.thickness > 240):
            warnings.warn(
                "Thickness is greater than 240nm, may produce invalid results",
                UserWarning,
            )
        if np.any(self.sw_angle < 80):
            warnings.warn(
                "Sidewall Angle is less than 80 degrees, may produce invalid results",
                UserWarning,
            )
        if np.any(self.sw_angle < 90):
            warnings.warn(
                "Sidewall Angle is greater than 90 degrees, may produce invalid results",
                UserWarning,
            )
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def _clean_args(self, wavelength):
        """Makes sure all attributes are the same size.

        Parses through all self attributes to make sure they're all the same size for
        simulations. Must be reimplemented for all child classes if they have unique attributes.
        Also takes in wavelength parameter to clean as is needed occasionally.

        Parameters
        ----------
        wavelength : float or ndarray
            Wavelength

        Returns
        ----------
        inputs : (tuple)
            Cleaned array of all devices attributes (and wavelength if included.)
        """
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
        """Predicts the output when light is put in port 1 and out port 2.

        Parameters
        ----------
        wavelength : float or ndarray
            Wavelength(s) to predict at

        Returns
        ----------
        k/t : complex ndarray
            The value of the light coming through
        """
        # check to make sure wavelength is in valid range
        if np.any(wavelength < 1450):
            warnings.warn(
                "Wavelength is less than 1450nm, may produce invalid results",
                UserWarning,
            )
        if np.any(wavelength > 1650):
            warnings.warn(
                "Wavelength is greater than 1650nm, may produce invalid results",
                UserWarning,
            )

        wavelength, width, thickness, sw_angle, radius, gap, length = self._clean_args(
            wavelength
        )
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        rr = HalfRacetrack(
            self.width, self.thickness, self.radius, self.gap, self.length
        )
        # k = rr.predict((1, 4), wavelength)
        t = rr.predict((1, 3), wavelength)

        # pull phase from coupler section
        phi_t = -np.unwrap(np.angle(t))

        # pull coupling from coupler section
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
        L_b = np.pi * radius  # length of two bent waveguides (half the circle)
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
        L_s = length  # length of the coupler regiod
        phi_s = np.unwrap(2 * np.pi * np.real(TE0) / wavelength) * L_s

        # get total phase
        phi = phi_t + phi_b + phi_s

        # calculate loss
        # lossTemp = self.loss.copy()
        # lossTemp[-1] = loss[-1] # assume uniform loss
        lossPoly = np.poly1d(self.loss)
        alpha = lossPoly(wavelength)

        # transfer function of resonator
        E = (
            (t_mag - alpha * np.exp(1j * phi))
            / (1 - alpha * t_mag * np.exp(1j * phi))
            * np.exp(-1j * phi)
        )

        return E, alpha, t, phi

    def sparams(self, wavelength):
        """Returns scattering parameters.

        Runs SCEE to get scattering parameters at wavelength input.

        Parameters
        ----------
        wavelength:    float or ndarray
            wavelengths to get sparams at

        Returns
        ----------
        freq : ndarray
            frequency for s_matrix in Hz, size n (number of wavelength points)
        s_matrix : ndarray
            size (2,2,n) complex matrix of scattering parameters
        """
        # get number of points to evaluate at
        n = 1 if np.isscalar(wavelength) else len(wavelength)
        # check to make sure the geometry isn't an array
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError(
                "You have changing geometries, getting sparams doesn't make sense"
            )
        s_matrix = np.zeros((2, 2, n), dtype="complex")

        # calculate upper half of matrix (diagonal is 0)
        s_matrix[0, 1] = self.predict(wavelength)[0]

        # apply symmetry (note diagonal is 0, no need to subtract it)
        s_matrix += np.transpose(s_matrix, (1, 0, 2))

        # transpose so depth comes first
        s_matrix = np.transpose(s_matrix, (2, 0, 1))
        return s_matrix

    def gds(self, filename=None, view=False, extra=0, units="nms"):
        """Writes the geometry to the gds file.

        Parameters
        ----------
        filename : str, optional
            Location to save file to. Defaults to None.
        extra : int, optional
            Extra straight portion to add to ends of waveguides to make room in simulation
            (units same as units parameter). Defaults to 0.
        units : {'microns' or 'nms'}, optional
            Units to save gds file in. Defaults to microns.
        view : bool, optional
            Whether to visually show gds file. Defaults to False.
        """
        # check to make sure the geometry isn't an array
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError(
                "You have changing geometries, making gds doesn't make sense"
            )

        if units == "nms":
            scale = 1e-3
        elif units == "microns":
            scale = 1
        else:
            raise ValueError("Invalid units")

        # scale to proper units
        sc_radius = self.radius * scale
        sc_gap = self.gap * scale
        sc_width = self.width * scale
        sc_length = self.length * scale

        # write to GDS
        pathTop = gdspy.Path(
            sc_width, (-sc_length / 2, 2 * sc_radius + sc_width / 2 + sc_gap / 2)
        )
        pathTop.segment(sc_length, "+x")
        pathTop.turn(sc_radius, "rr")
        pathTop.segment(sc_length, "-x")
        pathTop.turn(sc_radius, "rr")

        pathBottom = gdspy.Path(
            sc_width,
            (-sc_radius - sc_width / 2 - sc_length / 2, -sc_gap / 2 - sc_width / 2),
        )
        pathBottom.segment(2 * (sc_radius + sc_width / 2) + sc_length, "+x")

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell("C0")
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if view:
            gdspy.LayoutViewer(cells="C0")

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()
