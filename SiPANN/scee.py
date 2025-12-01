import warnings
from abc import ABC, abstractmethod

import gdstk
import numpy as np
import pkg_resources
from scipy import special
# We try to import simpson. If the user has an old scipy, we fall back to simps.
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson

from scipy.integrate import quad # Kept for non-symmetric classes

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
    """Return neff for a given waveguide profile."""
    wavelength, width, thickness, sw_angle = clean_inputs(
        (wavelength, width, thickness, sw_angle)
    )
    _, _, _, _, neff = get_coeffs(wavelength, width, thickness, sw_angle)
    return neff


def get_coeffs(wavelength, width, thickness, sw_angle):
    """Return coefficients and neff for a given waveguide profile as used in SCEE."""
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
    """Return coupling as found in Columbia paper."""
    even_part = ae * np.exp(-ge * gap) * B(xe) / ge
    odd_part = ao * np.exp(-go * gap) * B(xo) / go
    phase_part = 2 * z_dist * neff

    mag = trig((even_part + odd_part) * np.pi / wavelength)
    phase = (even_part - odd_part + phase_part) * np.pi / wavelength + offset

    return mag * np.exp(-1j * phase)


def clean_inputs(inputs):
    """Makes all inputs as the same shape to allow passing arrays through."""
    inputs = list(inputs)
    for i in range(len(inputs)):
        if np.isscalar(inputs[i]):
            inputs[i] = np.array([inputs[i]])

    n = max(len(i) for i in inputs)

    for i in range(len(inputs)):
        if len(inputs[i]) != n:
            if len(inputs[i]) != 1:
                if len(inputs[i]) != 0:
                    raise ValueError("Mismatched Input Array Size")
            inputs[i] = np.full((n), inputs[i][0])

    return inputs


class DC(ABC):
    """Abstract Class that all directional couplers inherit from."""

    def __init__(self, width, thickness, sw_angle=90):
        self.width = width
        self.thickness = thickness
        self.sw_angle = sw_angle
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
        if np.any(self.sw_angle > 90):
            warnings.warn("Sidewall Angle is greater than 90 degrees, may produce invalid results", UserWarning)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle))

    def update(self, **kwargs):
        self.width = kwargs.get("width", self.width)
        self.thickness = kwargs.get("thickness", self.thickness)
        self.sw_angle = kwargs.get("sw_angle", self.sw_angle)

    def sparams(self, wavelength):
        n = 1 if np.isscalar(wavelength) else len(wavelength)
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, getting sparams doesn't make sense")
        s_matrix = np.zeros((4, 4, n), dtype="complex")

        for i in range(1, 5):
            for j in range(i, 5):
                s_matrix[i - 1, j - 1] = self.predict((i, j), wavelength)

        s_matrix += np.transpose(s_matrix, (1, 0, 2))
        s_matrix = np.transpose(s_matrix, (2, 0, 1))
        return s_matrix

    @abstractmethod
    def predict(self, ports, wavelength):
        pass

    @abstractmethod
    def gds(self, filename=None, extra=0, units="microns", view=False, sbend_h=0, sbend_v=0):
        pass


class GapFuncSymmetric(DC):
    def __init__(self, width, thickness, gap, dgap, zmin, zmax, sw_angle=90, dx=100):
        """
        Added 'dx' parameter to control integration step size (default 100nm).
        """
        super().__init__(width, thickness, sw_angle)
        self.gap = gap
        self.dgap = dgap
        self.zmin = zmin
        self.zmax = zmax
        self.dx = dx # Step size for numerical integration

    def update(self, **kwargs):
        super().update(**kwargs)
        self.gap = kwargs.get("gap", self.gap)
        self.dgap = kwargs.get("dgap", self.dgap)
        self.zmin = kwargs.get("zmin", self.zmin)
        self.zmax = kwargs.get("zmax", self.zmax)
        self.dx = kwargs.get("dx", self.dx)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle))

    def predict(self, ports, wavelength, extra_arc=0, part="both"):
        if np.any(wavelength < 1450):
            warnings.warn("Wavelength is less than 1450nm, may produce invalid results", UserWarning)
        if np.any(wavelength > 1650):
            warnings.warn("Wavelength is greater than 1650nm, may produce invalid results", UserWarning)

        wavelength, width, thickness, sw_angle = self._clean_args(wavelength)
        
        # Get coefficients for all wavelengths at once
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        if (ports[0] == ports[1]) or (ports[0] + ports[1] == 3) or (ports[0] + ports[1] == 7):
            return np.zeros(len(wavelength))

        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        # --- OPTIMIZATION START: Vectorized Integration ---
        
        # 1. Create spatial grid (z)
        num_steps = int((self.zmax - self.zmin) / self.dx) + 1
        z_vals = np.linspace(self.zmin, self.zmax, num_steps)
        
        # 2. Calculate gaps and arc length for the whole grid
        gaps = self.gap(z_vals)
        dgaps = self.dgap(z_vals)
        
        # Arc length calculation
        arc_term = np.sqrt(1 + (dgaps / 2) ** 2)
        z_dist = simpson(arc_term, z_vals) + extra_arc

        # 3. Broadcast arrays for simultaneous calculation
        ae_grid = ae[:, np.newaxis]
        ge_grid = ge[:, np.newaxis]
        ao_grid = ao[:, np.newaxis]
        go_grid = go[:, np.newaxis]
        gaps_grid = gaps[np.newaxis, :]

        # 4. Compute Integrands (N wavelengths x M spatial steps)
        even_integrand = ae_grid * np.exp(-ge_grid * gaps_grid)
        odd_integrand = ao_grid * np.exp(-go_grid * gaps_grid)

        # 5. Integrate along spatial axis (axis 1) using Simpson's rule
        even_integral = simpson(even_integrand, z_vals, axis=1)
        odd_integral = simpson(odd_integrand, z_vals, axis=1)

        # 6. Combine results
        mag = np.ones(len(wavelength))
        phase = np.zeros(len(wavelength))

        if part in ["both", "mag"]:
            mag = trig(np.pi * (even_integral + odd_integral) / wavelength)
        
        if part in ["both", "ph"]:
            phase = (
                np.pi * (even_integral - odd_integral) / wavelength
                + 2 * np.pi * neff * z_dist / wavelength
                + offset
            )

        return mag * np.exp(-1j * phase)
        # --- OPTIMIZATION END ---

    def gds(self, filename=None, extra=0, units="microns", view=False, sbend_h=0, sbend_v=0):
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == "nms":
            scale = 1
        elif units == "microns":
            scale = 10 ** -3
        else:
            raise ValueError("Invalid units")

        sc_zmin = float(self.zmin * scale)
        sc_zmax = float(self.zmax * scale)
        sc_width = float(self.width * scale)
        cL = sc_zmax - sc_zmin
        cH = self.gap(self.zmin) * scale / 2

        # Parametric functions
        paraTop = lambda x: (
            x * cL + sc_zmin,
            scale * self.gap(x * (self.zmax - self.zmin) + self.zmin) / 2 + sc_width / 2,
        )
        paraBottom = lambda x: (
            x * cL + sc_zmin,
            -scale * self.gap(x * (self.zmax - self.zmin) + self.zmin) / 2 - sc_width / 2,
        )
        sbend = sbend_h != 0 and sbend_v != 0
        sbendDown = lambda x: (sbend_h * x, -sbend_v / 2 * (1 - np.cos(np.pi * x)))
        sbendUp = lambda x: (sbend_h * x, sbend_v / 2 * (1 - np.cos(np.pi * x)))

        # --- GDSTK IMPLEMENTATION (Gradient removed) ---
        pathTop = gdstk.FlexPath(
            (sc_zmin - extra - sbend_h, cH + sc_width / 2 + sbend_v), sc_width
        )
        pathTop.horizontal(extra, relative=True)
        if sbend:
            pathTop.parametric(sbendDown, relative=True)
        pathTop.parametric(paraTop, relative=False)
        if sbend:
            pathTop.parametric(sbendUp, relative=True)
        pathTop.horizontal(extra, relative=True)

        pathBottom = gdstk.FlexPath(
            (sc_zmin - extra - sbend_h, -cH - sc_width / 2 - sbend_v), sc_width
        )
        pathBottom.horizontal(extra, relative=True)
        if sbend:
            pathBottom.parametric(sbendUp, relative=True)
        pathBottom.parametric(paraBottom, relative=False)
        if sbend:
            pathBottom.parametric(sbendDown, relative=True)
        pathBottom.horizontal(extra, relative=True)

        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        path_cell = lib.new_cell("C0")
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if filename is not None:
            lib.write_gds(filename)
        elif view:
            print("View not supported in Gdstk. Please save to GDS file.")


class GapFuncAntiSymmetric(DC):
    def __init__(self, width, thickness, gap, zmin, zmax, arc1, arc2, arc3, arc4, sw_angle=90):
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
        self.zmin = kwargs.get("zmin", self.zmin)
        self.zmax = kwargs.get("zmax", self.zmax)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle))

    def predict(self, ports, wavelength, extra_arc=0, part="both"):
        # Implementation unchanged from original
        if np.any(wavelength < 1450):
            warnings.warn("Wavelength is less than 1450nm", UserWarning)
        if np.any(wavelength > 1650):
            warnings.warn("Wavelength is greater than 1650nm", UserWarning)

        wavelength, width, thickness, sw_angle = self._clean_args(wavelength)
        n = len(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        if 1 in ports and 3 in ports:
            z_dist = self.arc1 + self.arc3 + extra_arc
        elif 1 in ports and 4 in ports:
            z_dist = self.arc1 + self.arc4 + extra_arc
        elif 2 in ports and 4 in ports:
            z_dist = self.arc2 + self.arc4 + extra_arc
        elif 2 in ports and 3 in ports:
            z_dist = self.arc2 + self.arc3 + extra_arc
        else:
            return np.zeros(len(wavelength))

        mag = np.ones(n)
        phase = np.zeros(n)
        for i in range(n):
            if part in ["both", "mag"]:
                f_mag = lambda z: float(
                    ae[i] * np.exp(-ge[i] * self.gap(z)) + ao[i] * np.exp(-go[i] * self.gap(z))
                )
                mag[i] = trig(np.pi * quad(f_mag, self.zmin, self.zmax)[0] / wavelength[i])
            if part in ["both", "ph"]:
                f_phase = lambda z: float(
                    ae[i] * np.exp(-ge[i] * self.gap(z)) - ao[i] * np.exp(-go[i] * self.gap(z))
                )
                phase[i] = (
                    np.pi * quad(f_phase, self.zmin, self.zmax)[0] / wavelength[i]
                    + 2 * np.pi * neff[i] * z_dist / wavelength[i]
                    + offset
                )

        return mag * np.exp(-1j * phase)

    def gds(self, filename=None, extra=0, units="microns", view=False, sbend_h=0, sbend_v=0):
        raise NotImplementedError("Generating GDS file of nonsymmetric coupler not supported yet.")


class HalfRing(DC):
    def __init__(self, width, thickness, radius, gap, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.radius = radius
        self.gap = gap
        if np.any(self.gap < 100):
            warnings.warn("Gap is less than 100nm, may produce invalid results", UserWarning)

    def update(self, **kwargs):
        super().update(**kwargs)
        self.radius = kwargs.get("radius", self.radius)
        self.gap = kwargs.get("gap", self.gap)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.radius, self.gap))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.radius, self.gap))

    def predict(self, ports, wavelength):
        if np.any(wavelength < 1450):
            warnings.warn("Wavelength is less than 1450nm", UserWarning)
        if np.any(wavelength > 1650):
            warnings.warn("Wavelength is greater than 1650nm", UserWarning)

        wavelength, width, thickness, sw_angle, radius, gap = self._clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        if 1 in ports and 3 in ports:
            z_dist = 2 * (radius + width / 2)
        elif (1 in ports and 4 in ports or (2 not in ports or 4 not in ports) and 2 in ports and 3 in ports):
            z_dist = np.pi * radius / 2 + radius + width / 2
        elif 2 in ports and 4 in ports:
            z_dist = np.pi * radius
        else:
            return np.zeros(len(wavelength))

        B = lambda x: np.pi * x * np.exp(-x) * (special.iv(1, x) + special.modstruve(-1, x))
        xe = ge * (radius + width / 2)
        xo = go * (radius + width / 2)
        return get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist)

    def gds(self, filename=None, view=False, extra=0, units="nms"):
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == "nms":
            scale = 1
        elif units == "microns":
            scale = 10 ** -3
        else:
            raise ValueError("Invalid units")

        sc_radius = float(self.radius * scale)
        sc_gap = float(self.gap * scale)
        sc_width = float(self.width * scale)

        # --- GDSTK IMPLEMENTATION ---
        pathTop = gdstk.FlexPath(
            (sc_radius, sc_radius + sc_width / 2 + sc_gap / 2 + extra), sc_width
        )
        pathTop.vertical(-extra, relative=True)
        pathTop.turn(sc_radius, -np.pi)  # 180 degree clockwise
        pathTop.vertical(extra, relative=True)

        pathBottom = gdstk.FlexPath(
            (-sc_radius - sc_width / 2 - extra, -sc_gap / 2 - sc_width / 2), sc_width
        )
        pathBottom.horizontal(2 * (sc_radius + sc_width / 2 + extra), relative=True)

        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        path_cell = lib.new_cell("C0")
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if filename is not None:
            lib.write_gds(filename)
        elif view:
            print("View not supported in Gdstk. Please save to GDS file.")


class HalfRacetrack(DC):
    def __init__(self, width, thickness, radius, gap, length, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.radius = radius
        self.gap = gap
        self.length = length
        if np.any(self.gap < 100):
            warnings.warn("Gap is less than 100nm, may produce invalid results", UserWarning)

    def update(self, **kwargs):
        super().update(**kwargs)
        self.radius = kwargs.get("radius", self.radius)
        self.gap = kwargs.get("gap", self.gap)
        self.length = kwargs.get("length", self.length)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.radius, self.gap, self.length))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.radius, self.gap, self.length))

    def predict(self, ports, wavelength):
        # Implementation unchanged from original
        if np.any(wavelength < 1450):
            warnings.warn("Wavelength is less than 1450nm", UserWarning)
        if np.any(wavelength > 1650):
            warnings.warn("Wavelength is greater than 1650nm", UserWarning)

        wavelength, width, thickness, sw_angle, radius, gap, length = self._clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        if 1 in ports and 3 in ports:
            z_dist = 2 * (radius + width / 2) + length
        elif (1 in ports and 4 in ports or (2 not in ports or 4 not in ports) and 2 in ports and 3 in ports):
            z_dist = np.pi * radius / 2 + radius + width / 2 + length
        elif 2 in ports and 4 in ports:
            z_dist = np.pi * radius + length
        else:
            return np.zeros(len(wavelength))

        B = lambda x: length * x / (radius + width / 2) + np.pi * x * np.exp(-x) * (special.iv(1, x) + special.modstruve(-1, x))
        xe = ge * (radius + width / 2)
        xo = go * (radius + width / 2)
        return get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist)

    def gds(self, filename=None, view=False, extra=0, units="nms"):
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == "nms":
            scale = 1
        elif units == "microns":
            scale = 10 ** -3
        else:
            raise ValueError("Invalid units")

        sc_radius = float(self.radius * scale)
        sc_gap = float(self.gap * scale)
        sc_width = float(self.width * scale)
        sc_length = float(self.length * scale)

        # --- GDSTK IMPLEMENTATION ---
        pathTop = gdstk.FlexPath(
            (sc_radius + sc_length / 2, sc_radius + sc_width / 2 + sc_gap / 2 + extra), sc_width
        )
        pathTop.vertical(-extra, relative=True)
        pathTop.turn(sc_radius, -np.pi / 2)
        pathTop.horizontal(-sc_length, relative=True)
        pathTop.turn(sc_radius, -np.pi / 2)
        pathTop.vertical(extra, relative=True)

        pathBottom = gdstk.FlexPath(
            (-sc_radius - sc_width / 2 - sc_length / 2 - extra, -sc_gap / 2 - sc_width / 2), sc_width
        )
        pathBottom.horizontal(2 * (sc_radius + sc_width / 2) + sc_length + 2 * extra, relative=True)

        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        path_cell = lib.new_cell("C0")
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if filename is not None:
            lib.write_gds(filename)
        elif view:
            print("View not supported in Gdstk. Please save to GDS file.")


class StraightCoupler(DC):
    def __init__(self, width, thickness, gap, length, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.gap = gap
        self.length = length
        if np.any(self.gap < 100):
            warnings.warn("Gap is less than 100nm, may produce invalid results", UserWarning)

    def update(self, **kwargs):
        super().update(**kwargs)
        self.gap = kwargs.get("gap", self.gap)
        self.length = kwargs.get("length", self.length)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.gap, self.length))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.gap, self.length))

    def predict(self, ports, wavelength):
        if np.any(wavelength < 1450):
            warnings.warn("Wavelength is less than 1450nm", UserWarning)
        if np.any(wavelength > 1650):
            warnings.warn("Wavelength is greater than 1650nm", UserWarning)

        wavelength, width, thickness, sw_angle, gap, length = self._clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        if 1 in ports and 3 in ports or 1 in ports and 4 in ports or 2 in ports and 4 in ports or 2 in ports and 3 in ports:
            z_dist = length
        else:
            return np.zeros(len(wavelength))

        B = lambda x: x
        xe = ge * length
        xo = go * length
        return get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist)

    def gds(self, filename=None, view=False, extra=0, units="nms", sbend_h=0, sbend_v=0):
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == "nms":
            scale = 1
        elif units == "microns":
            scale = 10 ** -3
        else:
            raise ValueError("Invalid units")

        sc_width = float(self.width * scale)
        sc_gap = float(self.gap * scale)
        sc_length = float(self.length * scale)

        sbend = sbend_h != 0 and sbend_v != 0
        sbendDown = lambda x: (sbend_h * x, -sbend_v / 2 * (1 - np.cos(np.pi * x)))
        sbendUp = lambda x: (sbend_h * x, sbend_v / 2 * (1 - np.cos(np.pi * x)))

        # --- GDSTK IMPLEMENTATION (Gradient Removed) ---
        pathTop = gdstk.FlexPath(
            (-sc_length / 2 - sbend_h - extra, sbend_v + sc_width / 2 + sc_gap / 2), sc_width
        )
        pathTop.horizontal(extra, relative=True)
        if sbend:
            pathTop.parametric(sbendDown, relative=True)
        pathTop.horizontal(sc_length, relative=True)
        if sbend:
            pathTop.parametric(sbendUp, relative=True)
        pathTop.horizontal(extra, relative=True)

        pathBottom = gdstk.FlexPath(
            (-sc_length / 2 - sbend_h - extra, -sbend_v - sc_width / 2 - sc_gap / 2), sc_width
        )
        pathBottom.horizontal(extra, relative=True)
        if sbend:
            pathBottom.parametric(sbendUp, relative=True)
        pathBottom.horizontal(sc_length, relative=True)
        if sbend:
            pathBottom.parametric(sbendDown, relative=True)
        pathBottom.horizontal(extra, relative=True)

        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        path_cell = lib.new_cell("C0")
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if filename is not None:
            lib.write_gds(filename)
        elif view:
            print("View not supported in Gdstk. Please save to GDS file.")


class Standard(DC):
    def __init__(self, width, thickness, gap, length, H, V, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.gap = gap
        self.length = length
        self.H = H
        self.V = V
        if np.any(self.gap < 100):
            warnings.warn("Gap is less than 100nm, may produce invalid results", UserWarning)

    def update(self, **kwargs):
        super().update(**kwargs)
        self.gap = kwargs.get("gap", self.gap)
        self.length = kwargs.get("length", self.length)
        self.H = kwargs.get("H", self.H)
        self.V = kwargs.get("V", self.V)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.gap, self.length, self.H, self.V))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.gap, self.length, self.H, self.V))

    def predict(self, ports, wavelength):
        if np.any(wavelength < 1450):
            warnings.warn("Wavelength is less than 1450nm", UserWarning)
        if np.any(wavelength > 1650):
            warnings.warn("Wavelength is greater than 1650nm", UserWarning)

        wavelength, width, thickness, sw_angle, gap, length, H, V = self._clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")

        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2

        m = (V * np.pi / 2) ** 2 / (H ** 2 + (V * np.pi / 2) ** 2)
        z_dist = length + 2 * np.sqrt(H ** 2 + (V * np.pi / 2) ** 2) / np.pi * special.ellipeinc(np.pi, m)
        
        if (1 in ports and 3 in ports or 1 in ports and 4 in ports or 2 in ports and 4 in ports or 2 in ports and 3 in ports):
             pass # z_dist is z_dist
        else:
            return np.zeros(len(wavelength))

        B = lambda x: x * (1 + 2 * H * np.exp(-V * x / length) * special.iv(0, V * x / length) / length)
        xe = ge * length
        xo = go * length
        return get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist)

    def gds(self, filename=None, view=False, extra=0, units="nms", sbend_h=0, sbend_v=0):
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == "nms":
            scale = 1
        elif units == "microns":
            scale = 10 ** -3
        else:
            raise ValueError("Invalid units")

        sc_width = float(self.width * scale)
        sc_gap = float(self.gap * scale)
        sc_length = float(self.length * scale)
        sc_H = float(self.H * scale)
        sc_V = float(self.V * scale)

        sbendDown = lambda x: (sc_H * x, -sc_V / 2 * (1 - np.cos(np.pi * x)))
        sbendUp = lambda x: (sc_H * x, sc_V / 2 * (1 - np.cos(np.pi * x)))

        sbend = sbend_h != 0 and sbend_v != 0
        sbendDownExtra = lambda x: (sbend_h * x, -sbend_v / 2 * (1 - np.cos(np.pi * x)))
        sbendUpExtra = lambda x: (sbend_h * x, sbend_v / 2 * (1 - np.cos(np.pi * x)))

        # --- GDSTK IMPLEMENTATION (Gradient Removed) ---
        pathTop = gdstk.FlexPath(
            (-sc_length / 2 - sc_H - sbend_h - extra, sc_V + sbend_v + sc_width / 2 + sc_gap / 2),
            sc_width
        )
        pathTop.horizontal(extra, relative=True)
        if sbend:
            pathTop.parametric(sbendDownExtra, relative=True)
        pathTop.parametric(sbendDown, relative=True)
        pathTop.horizontal(sc_length, relative=True)
        pathTop.parametric(sbendUp, relative=True)
        if sbend:
            pathTop.parametric(sbendUpExtra, relative=True)
        pathTop.horizontal(extra, relative=True)

        pathBottom = gdstk.FlexPath(
            (-sc_length / 2 - sc_H - sbend_h - extra, -sc_V - sbend_v - sc_width / 2 - sc_gap / 2),
            sc_width
        )
        pathBottom.horizontal(extra, relative=True)
        if sbend:
            pathBottom.parametric(sbendUpExtra, relative=True)
        pathBottom.parametric(sbendUp, relative=True)
        pathBottom.horizontal(sc_length, relative=True)
        pathBottom.parametric(sbendDown, relative=True)
        if sbend:
            pathBottom.parametric(sbendDownExtra, relative=True)
        pathBottom.horizontal(extra, relative=True)

        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        path_cell = lib.new_cell("C0")
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if filename is not None:
            lib.write_gds(filename)
        elif view:
            print("View not supported in Gdstk. Please save to GDS file.")


class DoubleHalfRing(DC):
    def __init__(self, width, thickness, radius, gap, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.radius = radius
        self.gap = gap
        if np.any(self.gap < 100):
            warnings.warn("Gap is less than 100nm, may produce invalid results", UserWarning)

    def update(self, **kwargs):
        super().update(**kwargs)
        self.radius = kwargs.get("radius", self.radius)
        self.gap = kwargs.get("gap", self.gap)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.radius, self.gap))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.radius, self.gap))

    def predict(self, ports, wavelength):
        # Implementation unchanged
        if np.any(wavelength < 1450):
             warnings.warn("Wavelength is less than 1450nm", UserWarning)
        if np.any(wavelength > 1650):
             warnings.warn("Wavelength is greater than 1650nm", UserWarning)
        wavelength, width, thickness, sw_angle, radius, gap = self._clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)
        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")
        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2
        if (1 in ports and 3 in ports or 1 in ports and 4 in ports or 2 in ports and 4 in ports or 2 in ports and 3 in ports):
            z_dist = np.pi * radius
        else:
            return np.zeros(len(wavelength))
        B = lambda x: 0.5 * np.pi * 2 * x * np.exp(-2 * x) * (special.iv(1, 2 * x) + special.modstruve(-1, 2 * x))
        xe = ge * (radius + width / 2)
        xo = go * (radius + width / 2)
        return get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist)

    def gds(self, filename, extra=0, units="nm", view=False):
        raise NotImplementedError("TODO: Write to GDS file")


class AngledHalfRing(DC):
    def __init__(self, width, thickness, radius, gap, theta, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.radius = radius
        self.gap = gap
        self.theta = theta
        if np.any(self.gap < 100):
            warnings.warn("Gap is less than 100nm, may produce invalid results", UserWarning)

    def update(self, **kwargs):
        super().update(**kwargs)
        self.radius = kwargs.get("radius", self.radius)
        self.gap = kwargs.get("gap", self.gap)
        self.theta = kwargs.get("theta", self.theta)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.radius, self.gap, self.theta))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.radius, self.gap, self.theta))

    def predict(self, ports, wavelength):
        # Implementation unchanged
        if np.any(wavelength < 1450):
             warnings.warn("Wavelength is less than 1450nm", UserWarning)
        if np.any(wavelength > 1650):
             warnings.warn("Wavelength is greater than 1650nm", UserWarning)
        wavelength, width, thickness, sw_angle, radius, gap, theta = self._clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)
        if not all(1 <= x <= 4 for x in ports):
            raise ValueError("Invalid Ports")
        if abs(ports[1] - ports[0]) == 2:
            trig = np.cos
            offset = 0
        else:
            trig = np.sin
            offset = np.pi / 2
        if 1 in ports and 3 in ports:
            z_dist = np.pi * (radius + width + gap)
        elif (1 in ports and 4 in ports or (2 not in ports or 4 not in ports) and 2 in ports and 3 in ports):
            z_dist = np.pi * (radius + width + gap) / 2 + np.pi * radius / 2
        elif 2 in ports and 4 in ports:
            z_dist = np.pi * radius
        else:
            return np.zeros(len(wavelength))
        B = lambda x: x
        xe = ge * theta * (radius + width / 2 + gap / 2)
        xo = go * theta * (radius + width / 2 + gap / 2)
        return get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist)

    def gds(self, filename, extra=0, units="nm", view=False):
        raise NotImplementedError("TODO: Write to GDS file")


class Waveguide(ABC):
    def __init__(self, width, thickness, length, sw_angle=90):
        self.width = width
        self.thickness = thickness
        self.length = length
        self.sw_angle = sw_angle
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
        if np.any(self.sw_angle > 90):
            warnings.warn("Sidewall Angle is greater than 90 degrees, may produce invalid results", UserWarning)

    def _clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.length))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.length))

    def update(self, **kwargs):
        self.width = kwargs.get("width", self.width)
        self.thickness = kwargs.get("thickness", self.thickness)
        self.length = kwargs.get("length", self.length)
        self.sw_angle = kwargs.get("sw_angle", self.sw_angle)

    def sparams(self, wavelength):
        n = 1 if np.isscalar(wavelength) else len(wavelength)
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, getting sparams doesn't make sense")
        s_matrix = np.zeros((2, 2, n), dtype="complex")
        s_matrix[0, 1] = self.predict(wavelength)
        s_matrix += np.transpose(s_matrix, (1, 0, 2))
        s_matrix = np.transpose(s_matrix, (2, 0, 1))
        return s_matrix

    def predict(self, wavelength):
        if np.any(wavelength < 1450):
            warnings.warn("Wavelength is less than 1450nm", UserWarning)
        if np.any(wavelength > 1650):
            warnings.warn("Wavelength is greater than 1650nm", UserWarning)
        wavelength, width, thickness, sw_angle, length = self._clean_args(wavelength)
        _, _, _, _, neff = get_coeffs(wavelength, width, thickness, sw_angle)
        z_dist = length
        phase = 2 * z_dist * neff * np.pi / wavelength
        return np.exp(-1j * phase)

    def gds(self, filename=None, extra=0, units="microns", view=False):
        if len(self._clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == "nms":
            scale = 1
        elif units == "microns":
            scale = 10 ** -3
        else:
            raise ValueError("Invalid units")

        sc_width = float(self.width * scale)
        sc_length = float(self.length * scale)

        # --- GDSTK IMPLEMENTATION ---
        path = gdstk.FlexPath((-sc_length / 2 - extra, 0), sc_width)
        path.horizontal(2 * extra + sc_length, relative=True)

        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        path_cell = lib.new_cell("C0")
        path_cell.add(path)

        if filename is not None:
            lib.write_gds(filename)
        elif view:
            print("View not supported in Gdstk. Please save to GDS file.")