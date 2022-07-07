import warnings
from abc import ABC, abstractmethod

import gdspy
import numpy as np
import pkg_resources
from scipy import special
from scipy.integrate import quad

from SiPANN.import_nn import ImportLR

##########################################################################################
####  We initialize all ANN's and regressions as global objects to speed things up.  #####
##########################################################################################
cross_file = pkg_resources.resource_filename("SiPANN", "LR/DC_coeffs.pkl")
DC_coeffs = ImportLR(cross_file)


#########################################################################################
######################  Helper Functions used throughout classes  #######################
#########################################################################################
def get_neff(wavelength, width, thickness, sw_angle=90):
    """Return neff for a given waveguide profile.

    Leverages Multivariate Linear Regression that maps wavelength, width, thickness and
    sidewall angle to effective index with silicon core and silicon dioxide cladding

    Parameters
    ----------
        wavelength:  float or ndarray
            wavelength (Valid for 1450nm-1650nm)
        width:    float or ndarray
            width (Valid for 400nm-600nm)
        thickness:    float or ndarray
            thickness (Valid for 180nm-240nm)
        sw_angle:    float or ndarray
            sw_angle (Valid for 80-90 degrees)

    Returns
    ----------
        neff:    float or ndarray
            effective index of waveguide
    """

    # clean everything
    wavelength, width, thickness, sw_angle = clean_inputs(
        (wavelength, width, thickness, sw_angle)
    )
    # get coefficients
    _, _, _, _, neff = get_coeffs(wavelength, width, thickness, sw_angle)

    return neff


def get_coeffs(wavelength, width, thickness, sw_angle):
    """Return coefficients and neff for a given waveguide profile as used in
    SCEE.

    Leverages Multivariate Linear Regression that maps wavelength, width, thickness and
    sidewall angle to effective index and coefficients used in estimate of even and odd
    effective indices with silicon core and silicon dioxide cladding.

    Parameters
    ----------
        wavelength:    float or ndarray
            wavelength (Valid for 1450nm-1650nm)
        width:    float or ndarray
            width (Valid for 400nm-600nm)
        thickness:    float or ndarray
            thickness (Valid for 180nm-240nm)
        sw_angle:    float or ndarray
            sw_angle (Valid for 80-90 degrees)

    Returns
    ----------
        ae:    float or ndarray
            used in even mode estimation in neff + ae exp(ge * g)
        ao:    float or ndarray
            used in odd mode estimation in neff + ao exp(go * g)
        ge:    float or ndarray
            used in even mode estimation in neff + ae exp(ge * g)
        go:    float or ndarray
            used in odd mode estimation in neff + ao exp(go * g)
        neff:    float or ndarray
            effective index of waveguide
    """
    inputs = np.column_stack((wavelength, width, thickness, sw_angle))
    coeffs = DC_coeffs.predict(inputs)
    ae = coeffs[:, 0]
    ao = coeffs[:, 1]
    ge = coeffs[:, 2]
    go = coeffs[:, 3]
    neff = coeffs[:, 4]

    return (ae, ao, ge, go, neff)


def get_closed_ans(
    ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist
):
    """Return coupling as found in Columbia paper.

    Uses general form of closed form solutions as found in M. Bahadori et al.,
    "Design Space Exploration of Microring Resonators in Silicon Photonic Interconnects: Impact of the Ring Curvature,"
    in Journal of Lightwave Technology, vol. 36, no. 13, pp. 2767-2782, 1 July1, 2018..

    Parameters
    ----------
        ae:    float or ndarray
            used in even mode estimation in neff + ae exp(ge * g)
        ao:    float or ndarray
            used in odd mode estimation in neff + ao exp(go * g)
        ge:    float or ndarray
            used in even mode estimation in neff + ae exp(ge * g)
        go:    float or ndarray
            used in odd mode estimation in neff + ao exp(go * g)
        neff:    float or ndarray
            effective index of waveguide
        wavelength:    float or ndarray (Valid for 1450nm-1650nm)
            wavelength
        gap:    float or ndarray
            gap distance
        B   (function): B function as found in paper
        xe:    float or ndarray
            as found in paper
        xo:    float or ndarray
            as found in paper
        offset:    float or ndarray
            0 or pi/2 depending on through/cross coupling
        trig:    float or ndarray
            sin or cos depending on through/cross coupling
        z_dist:    float or ndarray
            distance light will travel

    Returns
    ----------
        k/t   (complex ndarray): coupling coefficient
    """
    even_part = ae * np.exp(-ge * gap) * B(xe) / ge
    odd_part = ao * np.exp(-go * gap) * B(xo) / go
    phase_part = 2 * z_dist * neff

    mag = trig((even_part + odd_part) * np.pi / wavelength)
    phase = (even_part - odd_part + phase_part) * np.pi / wavelength + offset

    return mag * np.exp(-1j * phase)


def clean_inputs(inputs):
    """Makes all inputs as the same shape to allow passing arrays through.

    Used to make sure all inputs have the same length - ie that it's trying
    to run a specific number of simulations, not a varying amount

    Parameters
    ----------
        inputs : tuple
            can be an arbitrary mixture of floats/ndarray

    Returns
    ----------
        inputs : tuple
            returns all inputs as same size ndarrays
    """

    inputs = list(inputs)
    # make all scalars into numpy arrays
    for i in range(len(inputs)):
        if np.isscalar(inputs[i]):
            inputs[i] = np.array([inputs[i]])

    # take largest size of numpy arrays, or set value (if it's not 0)
    n = max(len(i) for i in inputs)

    # if it's smaller than largest, make array full of same value
    for i in range(len(inputs)):
        if len(inputs[i]) != n:
            if len(inputs[i]) != 1:
                raise ValueError("Mismatched Input Array Size")
            inputs[i] = np.full((n), inputs[i][0])

    return inputs


class DC(ABC):
    """Abstract Class that all directional couplers inherit from. Each DC will
    inherit from it.

    Base Class for DC. All other DC classes should be based on this one, including same functions (so
    documentation should be the same/similar with exception of device specific attributes). Ports are numbered as::

        |       2---\      /---4       |
        |            ------            |
        |            ------            |
        |       1---/      \---3       |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """

    def __init__(self, width, thickness, sw_angle=90):
        self.width = width
        self.thickness = thickness
        self.sw_angle = sw_angle
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
        if np.any(self.sw_angle > 90):
            warnings.warn(
                "Sidewall Angle is greater than 90 degrees, may produce invalid results",
                UserWarning,
            )

    def _clean_args(self, wavelength):
        """Makes sure all attributes are the same size.

        Parses through all self attributes to make sure they're all the same size for
        simulations. Must be reimplemented for all child classes if they have unique attributes.
        Also takes in wavelength parameter to clean as is needed occasionally.

        Parameters
        ----------
        wavelength : float or ndarray
            Wavelength (Valid for 1450nm-1650nm)

        Returns
        ----------
        inputs : (tuple)
            Cleaned array of all devices attributes (and wavelength if included.)
        """
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle))

    def update(self, **kwargs):
        """Takes in any parameter defined by __init__ and changes it.

        Parameters
        ----------
        attribute : float or ndarray
            Included if any device needs to have an attribute changed.
        """
        self.width = kwargs.get("width", self.width)
        self.thickness = kwargs.get("thickness", self.thickness)
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
        if np.any(self.sw_angle > 90):
            warnings.warn(
                "Sidewall Angle is greater than 90 degrees, may produce invalid results",
                UserWarning,
            )

    def sparams(self, wavelength):
        """Returns scattering parameters.

        Runs SCEE to get scattering parameters at wavelength input.

        Parameters
        ----------
        wavelength:    float or ndarray
            wavelengths to get sparams at (Valid for 1450nm-1650nm)

        Returns
        ----------
        freq : ndarray
            frequency for s_matrix in Hz, size n (number of wavelength points)
        s_matrix : ndarray
            size (n,4,4) complex matrix of scattering parameters, in order of passed in wavelengths
        """
        # get number of points to evaluate at
        n = 1 if np.isscalar(wavelength) else len(wavelength)
        # check to make sure the geometry isn't an array
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError(
                "You have changing geometries, getting sparams doesn't make sense"
            )
        s_matrix = np.zeros((4, 4, n), dtype="complex")

        # calculate upper half of matrix (diagonal is 0)
        for i in range(1, 5):
            for j in range(i, 5):
                s_matrix[i - 1, j - 1] = self.predict((i, j), wavelength)

        # apply symmetry (note diagonal is 0, no need to subtract it)
        s_matrix += np.transpose(s_matrix, (1, 0, 2))

        # transpose so depth comes first
        s_matrix = np.transpose(s_matrix, (2, 0, 1))
        return s_matrix

    @abstractmethod
    def predict(self, ports, wavelength):
        """Predicts the output when light is put in the specified port (see
        diagram above)

        Parameters
        ----------
        ports : 2-tuple
            Specifies the port coming in and coming out
        wavelength : float or ndarray
            Wavelength(s) to predict at (Valid for 1450nm-1650nm)
        extra_arc : float, optional
            Adds phase to compensate for waveguides getting to gap function. Defaults to 0.
        part : {"both", "mag", "ph"}, optional
            To speed up calculation, can calculate only magnitude (mag), phase (ph), or both. Defaults to both.

        Returns
        ----------
        k/t : complex ndarray
            The value of the light coming through
        """
        pass

    @abstractmethod
    def gds(
        self, filename=None, extra=0, units="microns", view=False, sbend_h=0, sbend_v=0
    ):
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
        sbend_h : int, optional
            How high to horizontally make additional sbends to move ports farther away.
            Sbends insert after extra. Only available in couplers with all horizontal
            ports (units same as units parameters). Defaults to 0
        sbend_v : int, optional
            Same as sbend_h, but vertical distance. Defaults to 0.
        """
        pass


#########################################################################################
# Integral Estimators. Make any coupling device as desired.
#########################################################################################
class GapFuncSymmetric(DC):
    """This class will create arbitrarily shaped SYMMETRIC (ie both waveguides
    are same shape) directional couplers.

    It takes in a gap function that describes the gap as one progreses through the device. Note that the shape fo the waveguide
    will simply be half of gap function. Also requires the derivative of the gap function for this purpose. Ports are numbered as::

        |       2---\      /---4       |
        |            ------            |
        |            ------            |
        |       1---/      \---3       |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        gap : function
            Gap function as one progresses along the waveguide (Must always be > 100nm)
        dgap : function
            Derivative of the gap function
        zmin : float
            Where to begin integration in the gap function
        zmax : float
            Where to end integration in the gap function
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """

    def __init__(self, width, thickness, gap, dgap, zmin, zmax, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.gap = gap
        self.dgap = dgap
        self.zmin = zmin
        self.zmax = zmax

    def update(self, **kwargs):
        super().update(**kwargs)
        self.gap = kwargs.get("gap", self.gap)
        self.dgap = kwargs.get("dgap", self.dgap)
        self.zmin = kwargs.get("zmin", self.zmin)
        self.zmax = kwargs.get("zmax", self.zmax)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle))

    def predict(self, ports, wavelength, extra_arc=0, part="both"):
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

        # clean data and get coefficients
        wavelength, width, thickness, sw_angle = self._clean_args(wavelength)
        n = len(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        # make sure ports are valid
        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        # if it's coming to itself, or to adjacent port
        if (
            (ports[0] == ports[1])
            or (ports[0] + ports[1] == 3)
            or (ports[0] + ports[1] == 7)
        ):
            return np.zeros(len(wavelength))

        # determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        # determine z distance
        arcFomula = lambda x: np.sqrt(1 + (self.dgap(x) / 2) ** 2)
        z_dist = quad(arcFomula, self.zmin, self.zmax)[0] + extra_arc

        # calculate everything
        mag = np.ones(n)
        phase = np.zeros(n)
        for i in range(n):
            if part in ["both", "mag"]:
                f_mag = lambda z: float(
                    ae[i] * np.exp(-ge[i] * self.gap(z))
                    + ao[i] * np.exp(-go[i] * self.gap(z))
                )
                mag[i] = trig(
                    np.pi * quad(f_mag, self.zmin, self.zmax)[0] / wavelength[i]
                )
            if part in ["both", "ph"]:
                f_phase = lambda z: float(
                    ae[i] * np.exp(-ge[i] * self.gap(z))
                    - ao[i] * np.exp(-go[i] * self.gap(z))
                )
                phase[i] = (
                    np.pi * quad(f_phase, self.zmin, self.zmax)[0] / wavelength[i]
                    + 2 * np.pi * neff[i] * z_dist / wavelength[i]
                    + offset
                )

        return mag * np.exp(-1j * phase)

    def gds(
        self, filename=None, extra=0, units="microns", view=False, sbend_h=0, sbend_v=0
    ):
        # check to make sure the geometry isn't an array
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError(
                "You have changing geometries, making gds doesn't make sense"
            )

        if units == "nms":
            scale = 1
        elif units == "microns":
            scale = 10 ** -3
        else:
            raise ValueError("Invalid units")

        # scale to proper units
        sc_zmin = self.zmin * scale
        sc_zmax = self.zmax * scale
        sc_width = self.width * scale
        cL = sc_zmax - sc_zmin
        cH = self.gap(self.zmin) * scale / 2

        # make parametric functions
        paraTop = lambda x: (
            x * cL + sc_zmin,
            scale * self.gap(x * (self.zmax - self.zmin) + self.zmin) / 2
            + sc_width / 2,
        )
        paraBottom = lambda x: (
            x * cL + sc_zmin,
            -scale * self.gap(x * (self.zmax - self.zmin) + self.zmin) / 2
            - sc_width / 2,
        )
        sbend = sbend_h != 0 and sbend_v != 0
        sbendDown = lambda x: (sbend_h * x, -sbend_v / 2 * (1 - np.cos(np.pi * x)))
        sbendUp = lambda x: (sbend_h * x, sbend_v / 2 * (1 - np.cos(np.pi * x)))
        dsbendDown = lambda x: (sbend_h, -np.pi * sbend_v / 2 * np.sin(np.pi * x))
        dsbendUp = lambda x: (sbend_h, np.pi * sbend_v / 2 * np.sin(np.pi * x))

        # write to GDS
        pathTop = gdspy.Path(
            sc_width, (sc_zmin - extra - sbend_h, cH + sc_width / 2 + sbend_v)
        )
        pathTop.segment(extra, "+x")
        if sbend:
            pathTop.parametric(sbendDown, dsbendDown)
        pathTop.parametric(paraTop, relative=False)
        if sbend:
            pathTop.parametric(sbendUp, dsbendUp)
        pathTop.segment(extra, "+x")

        pathBottom = gdspy.Path(
            sc_width, (sc_zmin - extra - sbend_h, -cH - sc_width / 2 - sbend_v)
        )
        pathBottom.segment(extra, "+x")
        if sbend:
            pathBottom.parametric(sbendUp, dsbendUp)
        pathBottom.parametric(paraBottom, relative=False)
        if sbend:
            pathBottom.parametric(sbendDown, dsbendDown)
        pathBottom.segment(extra, "+x")

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell("C0")
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if view:
            gdspy.LayoutViewer(cells=path_cell)

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()


class GapFuncAntiSymmetric(DC):
    """This class will create arbitrarily shaped ANTISYMMETRIC (ie waveguides
    are different shapes) directional couplers.

    It takes in a gap function that describes the gap as one progreses through the device. Also takes in arc length
    of each port up till coupling point.
    Ports are numbered as:
    |       2---\      /---4       |
    |            ------            |
    |            ------            |
    |       1---/      \---3       |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        gap : function
            Gap function as one progresses along the waveguide (Must always be > 100nm)
        zmin : float
            Where to begin integration in the gap function
        zmax : float
            Where to end integration in the gap function
        arc1, arc2, arc3, arc4 : float
            Arclength from entrance of each port till minimum coupling point
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """

    def __init__(
        self, width, thickness, gap, zmin, zmax, arc1, arc2, arc3, arc4, sw_angle=90
    ):
        super().__init__(width, thickness, sw_angle)
        self.gap = gap
        self.zmin = zmin
        self.zmax = zmax
        self.arc1 = arc1
        self.arc2 = arc2
        self.arc3 = arc3
        self.arc4 = arc4

    def update(self, **kwargs):
        super().update(**kwargs)
        self.gap = kwargs.get("gap", self.gap)
        self.arc_l = kwargs.get("arc_l", self.arc_l)
        self.arc_u = kwargs.get("arc_u", self.arc_u)
        self.zmin = kwargs.get("zmin", self.zmin)
        self.zmax = kwargs.get("zmax", self.zmax)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle))

    def predict(self, ports, wavelength, extra_arc=0, part="both"):
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

        # clean data and get coefficients
        wavelength, width, thickness, sw_angle = self._clean_args(wavelength)
        n = len(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        # make sure ports are valid
        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        # determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        # determine z distance
        if 1 in ports and 3 in ports:
            z_dist = self.arc1 + self.arc3 + extra_arc
        elif 1 in ports and 4 in ports:
            z_dist = self.arc1 + self.arc4 + extra_arc
        elif 2 in ports and 4 in ports:
            z_dist = self.arc2 + self.arc4 + extra_arc
        elif 2 in ports and 3 in ports:
            z_dist = self.arc2 + self.arc3 + extra_arc
        # if it's coming to itself, or to adjacent port
        else:
            return np.zeros(len(wavelength))

        # calculate everything
        mag = np.ones(n)
        phase = np.zeros(n)
        for i in range(n):
            if part in ["both", "mag"]:
                f_mag = lambda z: float(
                    ae[i] * np.exp(-ge[i] * self.gap(z))
                    + ao[i] * np.exp(-go[i] * self.gap(z))
                )
                mag[i] = trig(
                    np.pi * quad(f_mag, self.zmin, self.zmax)[0] / wavelength[i]
                )
            if part in ["both", "ph"]:
                f_phase = lambda z: float(
                    ae[i] * np.exp(-ge[i] * self.gap(z))
                    - ao[i] * np.exp(-go[i] * self.gap(z))
                )
                phase[i] = (
                    np.pi * quad(f_phase, self.zmin, self.zmax)[0] / wavelength[i]
                    + 2 * np.pi * neff[i] * z_dist / wavelength[i]
                    + offset
                )

        return mag * np.exp(-1j * phase)

    def gds(
        self, filename=None, extra=0, units="microns", view=False, sbend_h=0, sbend_v=0
    ):
        """Still needs to be implemented for this class."""
        raise NotImplementedError(
            "Generating GDS file of nonsymmetric coupler not supported yet."
        )


#########################################################################################
# All the Different types of DC's with closed form solutions. These will be faster than defining it manually in the function form above.
#########################################################################################
class HalfRing(DC):
    """This class will create half of a ring resonator.

    It takes in a radius and gap along with usual waveguide parameters. Ports are numbered as::

        |         2 \     / 4          |
        |            \   /             |
        |             ---              |
        |         1---------3          |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        radius : float or ndarray
            Distance from center of ring to middle of waveguide in nm.
        gap : float or ndarray
            Minimum distance from ring waveguide edge to straight waveguide edge in nm. (Must be > 100nm)
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """

    def __init__(self, width, thickness, radius, gap, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.radius = radius
        self.gap = gap
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def update(self, **kwargs):
        super().update(**kwargs)
        self.radius = kwargs.get("radius", self.radius)
        self.gap = kwargs.get("gap", self.gap)
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs(
                (self.width, self.thickness, self.sw_angle, self.radius, self.gap)
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
                )
            )

    def predict(self, ports, wavelength):
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

        # clean data and get coefficients
        wavelength, width, thickness, sw_angle, radius, gap = self._clean_args(
            wavelength
        )
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        # make sure ports are valid
        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        # determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        # determine z distance
        if 1 in ports and 3 in ports:
            z_dist = 2 * (radius + width / 2)
        elif (
            1 in ports
            and 4 in ports
            or (2 not in ports or 4 not in ports)
            and 2 in ports
            and 3 in ports
        ):
            z_dist = np.pi * radius / 2 + radius + width / 2
        elif 2 in ports and 4 in ports:
            z_dist = np.pi * radius
        else:
            return np.zeros(len(wavelength))

        # calculate everything
        B = (
            lambda x: np.pi
            * x
            * np.exp(-x)
            * (special.iv(1, x) + special.modstruve(-1, x))
        )
        xe = ge * (radius + width / 2)
        xo = go * (radius + width / 2)
        return get_closed_ans(
            ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist
        )

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
            scale = 1
        elif units == "microns":
            scale = 10 ** -3
        else:
            raise ValueError("Invalid units")

        # scale to proper units
        sc_radius = self.radius * scale
        sc_gap = self.gap * scale
        sc_width = self.width * scale

        # write to GDS
        pathTop = gdspy.Path(
            sc_width, (sc_radius, sc_radius + sc_width / 2 + sc_gap / 2 + extra)
        )
        pathTop.segment(extra, "-y")
        pathTop.arc(sc_radius, 0, -np.pi)
        pathTop.segment(extra, "+y")

        pathBottom = gdspy.Path(
            sc_width, (-sc_radius - sc_width / 2 - extra, -sc_gap / 2 - sc_width / 2)
        )
        pathBottom.segment(2 * (sc_radius + sc_width / 2 + extra), "+x")

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell("C0")
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if view:
            gdspy.LayoutViewer(cells=path_cell)

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()


class HalfRacetrack(DC):
    """This class will create half of a ring resonator.

    It takes in a radius and gap along with usual waveguide parameters. Ports are numbered as::

        |      2 \           / 4       |
        |         \         /          |
        |          ---------           |
        |      1---------------3       |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        radius : float or ndarray
            Distance from center of ring to middle of waveguide in nm.
        gap : float or ndarray
            Minimum distance from ring waveguide edge to straight waveguide edge in nm. (Must be > 100nm)
        length : float or ndarray
            Length of straight portion of ring waveguide in nm.
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """

    def __init__(self, width, thickness, radius, gap, length, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.radius = radius
        self.gap = gap
        self.length = length
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def update(self, **kwargs):
        super().update(**kwargs)
        self.radius = kwargs.get("radius", self.radius)
        self.gap = kwargs.get("gap", self.gap)
        self.length = kwargs.get("length", self.length)
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

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

    def predict(self, ports, wavelength):
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

        # clean data and get coefficients
        wavelength, width, thickness, sw_angle, radius, gap, length = self._clean_args(
            wavelength
        )
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        # make sure ports are valid
        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        # determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        # determine z distance
        if 1 in ports and 3 in ports:
            z_dist = 2 * (radius + width / 2) + length
        elif (
            1 in ports
            and 4 in ports
            or (2 not in ports or 4 not in ports)
            and 2 in ports
            and 3 in ports
        ):
            z_dist = np.pi * radius / 2 + radius + width / 2 + length
        elif 2 in ports and 4 in ports:
            z_dist = np.pi * radius + length
        else:
            return np.zeros(len(wavelength))

        # calculate everything
        B = lambda x: length * x / (radius + width / 2) + np.pi * x * np.exp(-x) * (
            special.iv(1, x) + special.modstruve(-1, x)
        )
        xe = ge * (radius + width / 2)
        xo = go * (radius + width / 2)
        return get_closed_ans(
            ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist
        )

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
            scale = 1
        elif units == "microns":
            scale = 10 ** -3
        else:
            raise ValueError("Invalid units")

        # scale to proper units
        sc_radius = self.radius * scale
        sc_gap = self.gap * scale
        sc_width = self.width * scale
        sc_length = self.length * scale

        # write to GDS
        pathTop = gdspy.Path(
            sc_width,
            (sc_radius + sc_length / 2, sc_radius + sc_width / 2 + sc_gap / 2 + extra),
        )
        pathTop.segment(extra, "-y")
        pathTop.arc(sc_radius, 0, -np.pi / 2)
        pathTop.segment(sc_length, "-x")
        pathTop.arc(sc_radius, -np.pi / 2, -np.pi)
        pathTop.segment(extra, "+y")

        pathBottom = gdspy.Path(
            sc_width,
            (
                -sc_radius - sc_width / 2 - sc_length / 2 - extra,
                -sc_gap / 2 - sc_width / 2,
            ),
        )
        pathBottom.segment(2 * (sc_radius + sc_width / 2) + sc_length + 2 * extra, "+x")

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell("C0")
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if view:
            gdspy.LayoutViewer(cells=path_cell)

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()


class StraightCoupler(DC):
    """This class will create half of a ring resonator.

    It takes in a radius and gap along with usual waveguide parameters. Ports are numbered as::

        |      2---------------4       |
        |      1---------------3       |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        gap : float or ndarray
           Distance between the two waveguides edge in nm. (Must be > 100nm)
        length : float or ndarray
            Length of both waveguides in nm.
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """

    def __init__(self, width, thickness, gap, length, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.gap = gap
        self.length = length
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def update(self, **kwargs):
        super().update(**kwargs)
        self.gap = kwargs.get("gap", self.gap)
        self.length = kwargs.get("length", self.length)
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs(
                (self.width, self.thickness, self.sw_angle, self.gap, self.length)
            )
        else:
            return clean_inputs(
                (
                    wavelength,
                    self.width,
                    self.thickness,
                    self.sw_angle,
                    self.gap,
                    self.length,
                )
            )

    def predict(self, ports, wavelength):
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

        # clean data and get coefficients
        wavelength, width, thickness, sw_angle, gap, length = self._clean_args(
            wavelength
        )
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        # make sure ports are valid
        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        # determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        # determine z distance
        if (
            1 in ports
            and 3 in ports
            or 1 in ports
            and 4 in ports
            or 2 in ports
            and 4 in ports
            or 2 in ports
            and 3 in ports
        ):
            z_dist = length
        else:
            return np.zeros(len(wavelength))

        # calculate everything
        B = lambda x: x
        xe = ge * length
        xo = go * length
        return get_closed_ans(
            ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist
        )

    def gds(
        self, filename=None, view=False, extra=0, units="nms", sbend_h=0, sbend_v=0
    ):
        # check to make sure the geometry isn't an array
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError(
                "You have changing geometries, making gds doesn't make sense"
            )

        if units == "nms":
            scale = 1
        elif units == "microns":
            scale = 10 ** -3
        else:
            raise ValueError("Invalid units")

        # scale to proper units
        sc_width = self.width * scale
        sc_gap = self.gap * scale
        sc_length = self.length * scale

        sbend = sbend_h != 0 and sbend_v != 0
        sbendDown = lambda x: (sbend_h * x, -sbend_v / 2 * (1 - np.cos(np.pi * x)))
        sbendUp = lambda x: (sbend_h * x, sbend_v / 2 * (1 - np.cos(np.pi * x)))
        dsbendDown = lambda x: (sbend_h, -np.pi * sbend_v / 2 * np.sin(np.pi * x))
        dsbendUp = lambda x: (sbend_h, np.pi * sbend_v / 2 * np.sin(np.pi * x))

        # write to GDS
        pathTop = gdspy.Path(
            sc_width,
            (-sc_length / 2 - sbend_h - extra, sbend_v + sc_width / 2 + sc_gap / 2),
        )
        pathTop.segment(extra, "+x")
        if sbend:
            pathTop.parametric(sbendDown, dsbendDown)
        pathTop.segment(sc_length, "+x")
        if sbend:
            pathTop.parametric(sbendUp, dsbendUp)
        pathTop.segment(extra, "+x")

        pathBottom = gdspy.Path(
            sc_width,
            (-sc_length / 2 - sbend_h - extra, -sbend_v - sc_width / 2 - sc_gap / 2),
        )
        pathBottom.segment(extra, "+x")
        if sbend:
            pathBottom.parametric(sbendUp, dsbendUp)
        pathBottom.segment(sc_length, "+x")
        if sbend:
            pathBottom.parametric(sbendDown, dsbendDown)
        pathBottom.segment(extra, "+x")

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell("C0")
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if view:
            gdspy.LayoutViewer(cells=path_cell)

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()


class Standard(DC):
    """Normal/Standard Shaped Directional Coupler.

    This is what most people think of when they think directional coupler. Ports are numbered as::

        |       2---\      /---4       |
        |            ------            |
        |            ------            |
        |       1---/      \---3       |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        gap : float or ndarray
           Minimum distance between the two waveguides edge in nm. (Must be > 100nm)
        length : float or ndarray
            Length of the straight portion of both waveguides in nm.
        H : float or ndarray
            Horizontal distance between end of coupler until straight portion in nm.
        H : float or ndarray
            Vertical distance between end of coupler until straight portion in nm.
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """

    def __init__(self, width, thickness, gap, length, H, V, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.gap = gap
        self.length = length
        self.H = H
        self.V = V
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def update(self, **kwargs):
        super().update(**kwargs)
        self.gap = kwargs.get("gap", self.gap)
        self.length = kwargs.get("length", self.length)
        self.H = kwargs.get("H", self.H)
        self.V = kwargs.get("V", self.V)
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs(
                (
                    self.width,
                    self.thickness,
                    self.sw_angle,
                    self.gap,
                    self.length,
                    self.H,
                    self.V,
                )
            )
        else:
            return clean_inputs(
                (
                    wavelength,
                    self.width,
                    self.thickness,
                    self.sw_angle,
                    self.gap,
                    self.length,
                    self.H,
                    self.V,
                )
            )

    def predict(self, ports, wavelength):
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

        # clean data and get coefficients
        wavelength, width, thickness, sw_angle, gap, length, H, V = self._clean_args(
            wavelength
        )
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        # make sure ports are valid
        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        # determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        # determine z distance - length + 2*sbend length
        m = (V * np.pi / 2) ** 2 / (H ** 2 + (V * np.pi / 2) ** 2)
        z_dist = length + 2 * np.sqrt(
            H ** 2 + (V * np.pi / 2) ** 2
        ) / np.pi * special.ellipeinc(np.pi, m)
        if (
            1 in ports
            and 3 in ports
            or 1 in ports
            and 4 in ports
            or 2 in ports
            and 4 in ports
            or 2 in ports
            and 3 in ports
        ):
            z_dist = z_dist
        else:
            return np.zeros(len(wavelength))

        # calculate everything
        B = lambda x: x * (
            1 + 2 * H * np.exp(-V * x / length) * special.iv(0, V * x / length) / length
        )
        xe = ge * length
        xo = go * length
        return get_closed_ans(
            ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist
        )

    def gds(
        self, filename=None, view=False, extra=0, units="nms", sbend_h=0, sbend_v=0
    ):
        # check to make sure the geometry isn't an array
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError(
                "You have changing geometries, making gds doesn't make sense"
            )

        if units == "nms":
            scale = 1
        elif units == "microns":
            scale = 10 ** -3
        else:
            raise ValueError("Invalid units")

        # scale to proper units
        sc_width = self.width * scale
        sc_gap = self.gap * scale
        sc_length = self.length * scale
        sc_H = self.H * scale
        sc_V = self.V * scale

        # make parametric functions
        sbendDown = lambda x: (sc_H * x, -sc_V / 2 * (1 - np.cos(np.pi * x)))
        sbendUp = lambda x: (sc_H * x, sc_V / 2 * (1 - np.cos(np.pi * x)))
        dsbendDown = lambda x: (sc_H, -np.pi * sc_V / 2 * np.sin(np.pi * x))
        dsbendUp = lambda x: (sc_H, np.pi * sc_V / 2 * np.sin(np.pi * x))

        sbend = sbend_h != 0 and sbend_v != 0
        sbendDownExtra = lambda x: (sbend_h * x, -sbend_v / 2 * (1 - np.cos(np.pi * x)))
        sbendUpExtra = lambda x: (sbend_h * x, sbend_v / 2 * (1 - np.cos(np.pi * x)))
        dsbendDownExtra = lambda x: (sbend_h, -np.pi * sbend_v / 2 * np.sin(np.pi * x))
        dsbendUpExtra = lambda x: (sbend_h, np.pi * sbend_v / 2 * np.sin(np.pi * x))

        # write to GDS
        pathTop = gdspy.Path(
            sc_width,
            (
                -sc_length / 2 - sc_H - sbend_h - extra,
                sc_V + sbend_v + sc_width / 2 + sc_gap / 2,
            ),
        )
        pathTop.segment(extra, "+x")
        if sbend:
            pathTop.parametric(sbendDownExtra, dsbendDownExtra)
        pathTop.parametric(sbendDown, dsbendDown)
        pathTop.segment(sc_length, "+x")
        pathTop.parametric(sbendUp, dsbendUp)
        if sbend:
            pathTop.parametric(sbendUpExtra, dsbendUpExtra)
        pathTop.segment(extra, "+x")

        pathBottom = gdspy.Path(
            sc_width,
            (
                -sc_length / 2 - sc_H - sbend_h - extra,
                -sc_V - sbend_v - sc_width / 2 - sc_gap / 2,
            ),
        )
        pathBottom.segment(extra, "+x")
        if sbend:
            pathBottom.parametric(sbendUpExtra, dsbendUpExtra)
        pathBottom.parametric(sbendUp, dsbendUp)
        pathBottom.segment(sc_length, "+x")
        pathBottom.parametric(sbendDown, dsbendDown)
        if sbend:
            pathBottom.parametric(sbendDownExtra, dsbendDownExtra)
        pathBottom.segment(extra, "+x")

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell("C0")
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if view:
            gdspy.LayoutViewer(cells=path_cell)

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()


class DoubleHalfRing(DC):
    """This class will create two equally sized halfrings coupling.

    It takes in a radius and gap along with usual waveguide parameters. Ports are numbered as::

        |         2 \     / 4          |
        |            \   /             |
        |             ---              |
        |             ---              |
        |            /   \             |
        |         1 /     \ 3          |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        radius : float or ndarray
            Distance from center of ring to middle of waveguide in nm.
        gap : float or ndarray
            Minimum distance from ring waveguide edge to other ring waveguide edge in nm. (Must be > 100nm)
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """

    def __init__(self, width, thickness, radius, gap, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.radius = radius
        self.gap = gap
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def update(self, **kwargs):
        super().update(**kwargs)
        self.radius = kwargs.get("radius", self.radius)
        self.gap = kwargs.get("gap", self.gap)
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs(
                (self.width, self.thickness, self.sw_angle, self.radius, self.gap)
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
                )
            )

    def predict(self, ports, wavelength):
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

        # clean data and get coefficients
        wavelength, width, thickness, sw_angle, radius, gap = self._clean_args(
            wavelength
        )
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        # make sure ports are valid
        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        # determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        # determine z distance
        if (
            1 in ports
            and 3 in ports
            or 1 in ports
            and 4 in ports
            or 2 in ports
            and 4 in ports
            or 2 in ports
            and 3 in ports
        ):
            z_dist = np.pi * radius
        else:
            return np.zeros(len(wavelength))

        # calculate everything
        B = (
            lambda x: 0.5
            * np.pi
            * 2
            * x
            * np.exp(-2 * x)
            * (special.iv(1, 2 * x) + special.modstruve(-1, 2 * x))
        )
        xe = ge * (radius + width / 2)
        xo = go * (radius + width / 2)
        return get_closed_ans(
            ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist
        )

    def gds(self, filename, extra=0, units="nm", view=False):
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
        raise NotImplementedError("TODO: Write to GDS file")


class AngledHalfRing(DC):
    """This class will create a halfring resonator with a pushed side.

    It takes in a radius and gap along with usual waveguide parameters. Ports are numbered as::

        |      2  \        / 4       |
        |          \      /          |
        |      1--- \    / ---3      |
        |          \ \  / /          |
        |           \ -- /           |
        |            ----            |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        radius : float or ndarray
            Distance from center of ring to middle of waveguide in nm.
        gap : float or ndarray
            Minimum distance from ring waveguide edge to straight waveguide edge in nm.  (Must be > 100nm)
        theta : float or ndarray
            Angle that the straight waveguide is curved in radians (???).
        sw_angle : float or ndarray, optional (Valid for 80-90 degrees)
    """

    def __init__(self, width, thickness, radius, gap, theta, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.radius = radius
        self.gap = gap
        self.theta = theta
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def update(self, **kwargs):
        super().update(**kwargs)
        self.radius = kwargs.get("radius", self.radius)
        self.gap = kwargs.get("gap", self.gap)
        self.theta = kwargs.get("theta", self.theta)
        if np.any(self.gap < 100):
            warnings.warn(
                "Gap is less than 100nm, may produce invalid results", UserWarning
            )

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs(
                (
                    self.width,
                    self.thickness,
                    self.sw_angle,
                    self.radius,
                    self.gap,
                    self.theta,
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
                    self.theta,
                )
            )

    def predict(self, ports, wavelength):
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

        # clean data and get coefficients
        wavelength, width, thickness, sw_angle, radius, gap, theta = self._clean_args(
            wavelength
        )
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        # make sure ports are valid
        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        # determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        # determine z distance
        if 1 in ports and 3 in ports:
            z_dist = np.pi * (radius + width + gap)
        elif (
            1 in ports
            and 4 in ports
            or (2 not in ports or 4 not in ports)
            and 2 in ports
            and 3 in ports
        ):
            z_dist = np.pi * (radius + width + gap) / 2 + np.pi * radius / 2
        elif 2 in ports and 4 in ports:
            z_dist = np.pi * radius
        else:
            return np.zeros(len(wavelength))

        # calculate everything
        B = lambda x: x
        xe = ge * theta * (radius + width / 2 + gap / 2)
        xo = go * theta * (radius + width / 2 + gap / 2)
        return get_closed_ans(
            ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist
        )

    def gds(self, filename, extra=0, units="nm", view=False):
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
        raise NotImplementedError("TODO: Write to GDS file")


class Waveguide(ABC):
    """Lossless model for a straight waveguide.

    Simple model that makes sparameters for a straight waveguide. May not be
    the best option, but plays nice with other models in SCEE. Ports are numbered as::

        |  1 ----------- 2   |

    Parameters
    ----------
        width : float or ndarray
            Width of the waveguide in nm (Valid for 400nm-600nm)
        thickness : float or ndarray
            Thickness of waveguide in nm (Valid for 180nm-240nm)
        length : float or ndarray
            Length of waveguide in nm.
        sw_angle : float or ndarray, optional
            Sidewall angle of waveguide from horizontal in degrees (Valid for 80-90 degrees). Defaults to 90.
    """

    def __init__(self, width, thickness, length, sw_angle=90):
        self.width = width
        self.thickness = thickness
        self.length = length
        self.sw_angle = sw_angle
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
        if np.any(self.sw_angle > 90):
            warnings.warn(
                "Sidewall Angle is greater than 90 degrees, may produce invalid results",
                UserWarning,
            )

    def _clean_args(self, wavelength):
        """Makes sure all attributes are the same size.

        Parses through all self attributes to make sure they're all the same size for
        simulations. Must be reimplemented for all child classes if they have unique attributes.
        Also takes in wavelength parameter to clean as is needed occasionally.

        Parameters
        ----------
        wavelength : float or ndarray
            Wavelength (Valid for 1450nm-1650nm)

        Returns
        ----------
        inputs : (tuple)
            Cleaned array of all devices attributes (and wavelength if included.)
        """
        if wavelength is None:
            return clean_inputs(
                (self.width, self.thickness, self.sw_angle, self.length)
            )
        else:
            return clean_inputs(
                (wavelength, self.width, self.thickness, self.sw_angle, self.length)
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
        self.length = kwargs.get("thickness", self.length)
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
        if np.any(self.sw_angle > 90):
            warnings.warn(
                "Sidewall Angle is greater than 90 degrees, may produce invalid results",
                UserWarning,
            )

    def sparams(self, wavelength):
        """Returns scattering parameters.

        Runs SCEE to get scattering parameters at wavelength input.

        Parameters
        ----------
        wavelength:    float or ndarray
            wavelengths to get sparams at (Valid for 1450nm-1650nm)

        Returns
        ----------
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
        s_matrix[0, 1] = self.predict(wavelength)

        # apply symmetry (note diagonal is 0, no need to subtract it)
        s_matrix += np.transpose(s_matrix, (1, 0, 2))

        # transpose so depth comes first
        s_matrix = np.transpose(s_matrix, (2, 0, 1))
        return s_matrix

    def predict(self, wavelength):
        """Predicts the output when light is put in port 1 and out port 2.

        Parameters
        ----------
        wavelength : float or ndarray
            Wavelength(s) to predict at (Valid for 1450nm-1650nm)

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

        # clean data and get coefficients
        wavelength, width, thickness, sw_angle, length = self._clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        # calculate everything
        z_dist = length
        phase = 2 * z_dist * neff * np.pi / wavelength

        return np.exp(-1j * phase)

    def gds(self, filename=None, extra=0, units="microns", view=False):
        """Writes the geometry to the gds file.

        Parameters
        ----------
            filename : str
                location to save file to, or if you don't want to defaults to None
            extra : int
                extra straight portion to add to ends of waveguides to make room in simulation
                                (input with units same as units input)
            units : str
                either 'microns' or 'nms'. Units to save gds file in
        """
        # check to make sure the geometry isn't an array
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError(
                "You have changing geometries, making gds doesn't make sense"
            )

        if units == "nms":
            scale = 1
        elif units == "microns":
            scale = 10 ** -3
        else:
            raise ValueError("Invalid units")

        # scale to proper units
        sc_width = self.width * scale
        sc_length = self.length * scale

        # write to GDS
        path = gdspy.Path(sc_width, (-sc_length / 2 - extra, 0))
        path.segment(2 * extra + sc_length, "+x")

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell("C0")
        path_cell.add(path)

        if view:
            gdspy.LayoutViewer(cells=path_cell)

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()
