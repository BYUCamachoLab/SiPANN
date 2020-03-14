import numpy as np
from abc import ABC, abstractmethod
from scipy.integrate import quad
import scipy.special as special
import pkg_resources
import joblib
import gdspy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

##########################################################################################
####  We initialize all ANN's and regressions as global objects to speed things up.  #####
##########################################################################################
cross_file = pkg_resources.resource_filename('SiPANN','LR/DC_coeffs.joblib')
DC_coeffs  = joblib.load(cross_file)

# cross_file = pkg_resources.resource_filename('SiPANN','LR/R_bent_wide.joblib')
# R_bent     = joblib.load(cross_file)

C          = 299792458

#########################################################################################
######################  Helper Functions used throughout classes  #######################
#########################################################################################
def get_neff(wavelength, width, thickness, sw_angle=90):
    """Return neff for a given waveguide profile

        Leverages Multivariate Linear Regression that maps wavelength, width, thickness and
        sidewall angle to effective index with silicon core and silicon dioxide cladding

        Args:
            wavelength (float/np.ndarray): wavelength
            width      (float/np.ndarray): width
            thickness  (float/np.ndarray): thickness
            sw_angle   (float/np.ndarray): sw_angle

        Returns:
            neff (float/np.ndarray): effective index of waveguide"""

    #clean everything
    wavelength, width, thickness, sw_angle = clean_inputs((wavelength, width, thickness, sw_angle))
    #get coefficients
    _, _, _, _, neff = get_coeffs(wavelength, width, thickness, sw_angle)

    return neff

def get_coeffs(wavelength, width, thickness, sw_angle):
    """Return coefficients and neff for a given waveguide profile as used in SCEE

        Leverages Multivariate Linear Regression that maps wavelength, width, thickness and
        sidewall angle to effective index and coefficients used in estimate of even and odd
        effective indices with silicon core and silicon dioxide cladding.

        Args:
            wavelength (float/np.ndarray): wavelength
            width      (float/np.ndarray): width
            thickness  (float/np.ndarray): thickness
            sw_angle   (float/np.ndarray): sw_angle

        Returns:
            ae   (float/np.ndarray): used in even mode estimation in neff + ae exp(ge * g)
            ao   (float/np.ndarray): used in odd mode estimation in neff + ao exp(go * g)
            ge   (float/np.ndarray): used in even mode estimation in neff + ae exp(ge * g)
            go   (float/np.ndarray): used in odd mode estimation in neff + ao exp(go * g)
            neff (float/np.ndarray): effective index of waveguide"""
    inputs = np.column_stack((wavelength, width, thickness, sw_angle))
    coeffs = DC_coeffs.predict(inputs)
    ae = coeffs[:,0]
    ao = coeffs[:,1]
    ge = coeffs[:,2]
    go = coeffs[:,3]
    neff = coeffs[:,4]

    return (ae, ao, ge, go, neff)

"""Plugs coeffs into actual closed form function"""
def get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist):
    """Return coupling as found in Columbia paper

        Uses general form of closed form solutions as found in M. Bahadori et al.,
        "Design Space Exploration of Microring Resonators in Silicon Photonic Interconnects: Impact of the Ring Curvature,"
        in Journal of Lightwave Technology, vol. 36, no. 13, pp. 2767-2782, 1 July1, 2018..

        Args:
            ae   (float/np.ndarray): used in even mode estimation in neff + ae exp(ge * g)
            ao   (float/np.ndarray): used in odd mode estimation in neff + ao exp(go * g)
            ge   (float/np.ndarray): used in even mode estimation in neff + ae exp(ge * g)
            go   (float/np.ndarray): used in odd mode estimation in neff + ao exp(go * g)
            neff (float/np.ndarray): effective index of waveguide
            wavelength (float/np.ndarray): wavelength
            gap (float/np.ndarray): gap distance
            B   (function): B function as found in paper
            xe  (float/np.ndarray): as found in paper
            xo  (float/np.ndarray): as found in paper
            offset  (float/np.ndarray): 0 or pi/2 depending on through/cross coupling
            trig  (float/np.ndarray): sin or cos depending on through/cross coupling
            z_dist  (float/np.ndarray): distance light will travel

        Returns:
            k/t   (complex np.ndarray): coupling coefficient found"""
    even_part = ae*np.exp(-ge*gap)*B(xe)/ge
    odd_part  = ao*np.exp(-go*gap)*B(xo)/go
    phase_part= 2*z_dist*neff

    mag =  trig( (even_part+odd_part)*np.pi / wavelength )
    phase = (even_part-odd_part+phase_part)*np.pi/wavelength + offset

    return mag*np.exp(-1j*phase)

def clean_inputs(inputs):
    """Makes all inputs as the same shape to allow passing arrays through

        Used to make sure all inputs have the same length - ie that it's trying
        to run a specific number of simulations, not a varying amount

        Args:
            inputs (tuple): can be a mixture of floats/np.ndarray of any amounts

        Returns:
            inputs (tuple): returns all inputs as same size np.ndarrays"""

    inputs = list(inputs)
    #make all scalars into numpy arrays
    for i in range(len(inputs)):
        if np.isscalar(inputs[i]):
            inputs[i] = np.array([inputs[i]])

    #take largest size of numpy arrays, or set value (if it's not 0)
    n = max([len(i) for i in inputs])

    #if it's smaller than largest, make array full of same value
    for i in range(len(inputs)):
        if len(inputs[i]) != n:
            if len(inputs[i]) != 1:
                raise ValueError("Mismatched Input Array Size")
            inputs[i] = np.full((n), inputs[i][0])

    return inputs

class DC(ABC):
    """
    Abstract Class that all directional couplers inherit from. Each DC will inherit from it and have initial arguments (in this order):

    width, thickness, sw_angle=90

    Also, each will have additional arguments as follows (in this order):

    GapFuncSymmetric:    gap (func), dgap (func), zmin, zmax
    RR:                  radius, gap
    Racetrack Resonator: radius, gap, length
    Straight:            gap, length
    Standard:            gap, length, H, V
    DoubleRR:            radius, gap
    CurvedRR:            radius, gap, theta

    Base Class for DC. All other DC classes should be based on this one, including same functions (so
        documentation should be the same). Ports are numbered as:
                2---\      /---4
                     ------
                     ------
                1---/      \---3
    """
    def __init__(self, width, thickness, sw_angle=90):
        self.width      = width
        self.thickness  = thickness
        self.sw_angle   = sw_angle

    def clean_args(self, wavelength):
        """Makes sure all attributes are the same size"""
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle))

    def update(self, **kwargs):
        """Takes in any parameter defined by __init__ and changes it."""
        self.width      = kwargs.get('width', self.width)
        self.thickness  = kwargs.get('thickness', self.thickness)
        self.sw_angle   = kwargs.get('sw_angle', self.sw_angle)

    def sparams(self, wavelength):
        """Returns sparams
        Args:
            wavelength(float/np.ndarray): wavelengths to get sparams at
        Returns:
            freq     (np.ndarray): frequency for s_matrix in Hz, size n (number of wavelength points)
            s_matrix (np.ndarray): size (4,4,n) complex matrix of scattering parameters
        """
        #get number of points to evaluate at
        if np.isscalar(wavelength):
            n = 1
        else:
            n = len(wavelength)

        #check to make sure the geometry isn't an array
        if len(self.clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, getting sparams doesn't make sense")
        s_matrix = np.zeros((4,4,n), dtype='complex')

        #calculate upper half of matrix (diagonal is 0)
        for i in range(1,5):
            for j in range(i,5):
                s_matrix[i-1,j-1] = self.predict((i,j), wavelength)

        #apply symmetry (note diagonal is 0, no need to subtract it)
        s_matrix += np.transpose(s_matrix, (1,0,2))
        freq = C/(wavelength*10**-9)

        #flip them so frequency is increasing
        if n != 1:
            freq = freq[::-1]
            s_matrix = s_matrix[:,:,::-1]

        #transpose so depth comes first
        s_matrix = np.transpose(s_matrix, (2, 0, 1))
        return (freq, s_matrix)

    @abstractmethod
    def predict(self, ports, wavelength):
        """Predicts the output when light is put in the bottom left port (see diagram above)

        Args:
            ports               (2-tuple): Specifies the port coming in and coming out
            wavelength (float/np.ndarray): wavelength(s) to predict at

        Returns:
            k/t (complex np.ndarray): returns the value of the light coming through"""
        pass

    @abstractmethod
    def gds(self, filename=None, extra=0, units='microns', view=False, sbend_h=0, sbend_v=0):
        """Writes the geometry to the gds file

        Args:
            filename (str): location to save file to, or if you don't want to defaults to None
            extra    (int): extra straight portion to add to ends of waveguides to make room in simulation
                                (input with units same as units input)
            units    (str): either 'microns' or 'nms'. Units to save gds file in
            view    (bool): whether to visually show gds file
            sbend_h  (int): how high to horizontally make additional sbends to move ports farther away.
                                Sbends insert after extra. Only available in couplers with all horizontal
                                ports (input with units same as units input)
            sbend_v  (int): same as sbend_h, but vertical distance.
        """
        pass

"""
This class will create arbitrarily shaped SYMMETRIC (ie both waveguides are same shape) directional couplers
"""
class GapFuncSymmetric(DC):
    def __init__(self, width, thickness, gap, dgap, zmin, zmax, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.gap  = gap
        self.dgap = dgap
        self.zmin = zmin
        self.zmax = zmax

    def update(self, **kwargs):
        super().update(**kwargs)
        self.gap  = kwargs.get('gap', self.gap)
        self.dgap = kwargs.get('dgap', self.dgap)
        self.zmin = kwargs.get('zmin', self.zmin)
        self.zmax = kwargs.get('zmax', self.zmax)

    def clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle))

    def predict(self, ports, wavelength, extra_arc=0, part='both'):
        """Has aditional 'part' parameter in case you only want magnitude (mag) or phase (ph)"""
        wavelength, width, thickness, sw_angle = self.clean_args(wavelength)
        n = len(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)
        #make sure ports are valid
        if not all(1 <= x <=4 for x in ports):
            raise ValueError('Invalid Ports')

        #if it's coming to itself, or to adjacent port
        if (ports[0] == ports[1]) or (ports[0] + ports[1] == 3) or (ports[0] + ports[1] == 7):
            return np.zeros(len(wavelength))

        #determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig   = np.cos
            offset = 0
        else:
            trig   = np.sin
            offset = np.pi/2

        #determine z distance
        arcFomula = lambda x: np.sqrt( 1 + (self.dgap(x)/2)**2 )
        z_dist = quad(arcFomula, self.zmin, self.zmax)[0] + extra_arc

        #calculate everything
        mag = np.ones(n)
        phase = np.zeros(n)
        for i in range(n):
            if part == 'both' or part == 'mag':
                f_mag  = lambda z: float(ae[i]*np.exp(-ge[i]*self.gap(z)) + ao[i]*np.exp(-go[i]*self.gap(z)))
                mag[i] = trig( np.pi * quad(f_mag, self.zmin, self.zmax)[0] / wavelength[i] )
            if part == 'both' or part == 'ph':
                f_phase  = lambda z: float(ae[i]*np.exp(-ge[i]*self.gap(z)) - ao[i]*np.exp(-go[i]*self.gap(z)))
                phase[i] = np.pi * quad(f_phase, self.zmin, self.zmax)[0] / wavelength[i] + 2*np.pi*neff[i]*z_dist/wavelength[i] + offset

        return mag*np.exp(-1j * phase)

    def gds(self, filename=None, extra=0, units='microns', view=False, sbend_h=0, sbend_v=0):
        #check to make sure the geometry isn't an array
        if len(self.clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == 'nms':
            scale = 1
        elif units == 'microns':
            scale = 10**-3
        else:
            raise ValueError('Invalid units')

        #scale to proper units
        sc_zmin = self.zmin*scale
        sc_zmax = self.zmax*scale
        sc_width= self.width*scale
        cL = (sc_zmax - sc_zmin)
        cH = self.gap(self.zmin) * scale / 2

        #make parametric functions
        paraTop    = lambda x: (x*(sc_zmax-sc_zmin)+sc_zmin, scale*self.gap(x*(self.zmax-self.zmin)+self.zmin)/2 + sc_width/2)
        paraBottom = lambda x: (x*(sc_zmax-sc_zmin)+sc_zmin, -scale*self.gap(x*(self.zmax-self.zmin)+self.zmin)/2 - sc_width/2)
        #dparaTop    = lambda x: (sc_zmax-sc_zmin, scale*(self.zmax-self.zmin)*self.dgap(x*(self.zmax-self.zmin)+self.zmin)/2)
        #dparaBottom = lambda x: (sc_zmax-sc_zmin, -scale*(self.zmax-self.zmin)*self.dgap(x*(self.zmax-self.zmin)+self.zmin)/2)

        sbend = False
        if sbend_h != 0 and sbend_v != 0:
            sbend = True
        sbendDown = lambda x: (sbend_h*x, -sbend_v/2*(1-np.cos(np.pi*x)))
        sbendUp   = lambda x: (sbend_h*x, sbend_v/2*(1-np.cos(np.pi*x)))
        dsbendDown = lambda x: (sbend_h, -np.pi*sbend_v/2*np.sin(np.pi*x))
        dsbendUp   = lambda x: (sbend_h, np.pi*sbend_v/2*np.sin(np.pi*x))

        #write to GDS
        pathTop = gdspy.Path(sc_width, (sc_zmin-extra-sbend_h, cH+sc_width/2+sbend_v))
        pathTop.segment(extra, '+x')
        if sbend: pathTop.parametric(sbendDown, dsbendDown)
        pathTop.parametric(paraTop, relative=False)
        if sbend: pathTop.parametric(sbendUp, dsbendUp)
        pathTop.segment(extra, '+x')

        pathBottom = gdspy.Path(sc_width, (sc_zmin-extra-sbend_h, -cH-sc_width/2-sbend_v))
        pathBottom.segment(extra, '+x')
        if sbend: pathBottom.parametric(sbendUp, dsbendUp)
        pathBottom.parametric(paraBottom, relative=False)
        if sbend: pathBottom.parametric(sbendDown, dsbendDown)
        pathBottom.segment(extra, '+x')

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell('C0')
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if view:
            gdspy.LayoutViewer(cells='C0')

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()

class GapFuncAntiSymmetric(DC):
    def __init__(self, width, thickness, gap, zmin, zmax, arc1, arc2, arc3, arc4, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.gap  = gap
        self.zmin = zmin
        self.zmax = zmax
        self.arc1 = arc1
        self.arc2 = arc2
        self.arc3 = arc3
        self.arc4 = arc4

    def update(self, **kwargs):
        super().update(**kwargs)
        self.gap   = kwargs.get('gap', self.gap)
        self.arc_l = kwargs.get('arc_l', self.arc_l)
        self.arc_u = kwargs.get('arc_u', self.arc_u)
        self.zmin  = kwargs.get('zmin', self.zmin)
        self.zmax  = kwargs.get('zmax', self.zmax)

    def clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle))

    def predict(self, ports, wavelength, extra_arc=0, part='both'):
        """Has aditional 'part' parameter in case you only want magnitude (mag) or phase (ph)"""
        wavelength, width, thickness, sw_angle = self.clean_args(wavelength)
        n = len(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)
        #make sure ports are valid
        if not all(1 <= x <=4 for x in ports):
            raise ValueError('Invalid Ports')

        #determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig   = np.cos
            offset = 0
        else:
            trig   = np.sin
            offset = np.pi/2

        #determine z distance
        if 1 in ports and 3 in ports:
            z_dist = self.arc1 + self.arc3 + extra_arc
        elif 1 in ports and 4 in ports:
            z_dist = self.arc1 + self.arc4 + extra_arc
        elif 2 in ports and 4 in ports:
            z_dist = self.arc2 + self.arc4 + extra_arc
        elif 2 in ports and 3 in ports:
            z_dist = self.arc2 + self.arc3 + extra_arc
        #if it's coming to itself, or to adjacent port
        else:
            return np.zeros(len(wavelength))

        #calculate everything
        mag = np.ones(n)
        phase = np.zeros(n)
        for i in range(n):
            if part == 'both' or part == 'mag':
                f_mag  = lambda z: float(ae[i]*np.exp(-ge[i]*self.gap(z)) + ao[i]*np.exp(-go[i]*self.gap(z)))
                mag[i] = trig( np.pi * quad(f_mag, self.zmin, self.zmax)[0] / wavelength[i] )
            if part == 'both' or part == 'ph':
                f_phase  = lambda z: float(ae[i]*np.exp(-ge[i]*self.gap(z)) - ao[i]*np.exp(-go[i]*self.gap(z)))
                phase[i] = np.pi * quad(f_phase, self.zmin, self.zmax)[0] / wavelength[i] + 2*np.pi*neff[i]*z_dist/wavelength[i] + offset

        return mag*np.exp(-1j * phase)

    def gds(self, filename=None, extra=0, units='microns', view=False, sbend_h=0, sbend_v=0):
        pass

"""
All the Different types of DC's with closed form solutions. These will be faster than defining it manually in the function form.
"""
class RR(DC):
    def __init__(self, width, thickness, radius, gap, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.radius = radius
        self.gap    = gap

    def update(self, **kwargs):
        super().update(**kwargs)
        self.radius = kwargs.get('radius', self.radius)
        self.gap    = kwargs.get('gap', self.gap)

    def clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.radius, self.gap))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.radius, self.gap))

    def predict(self, ports, wavelength):
        wavelength, width, thickness, sw_angle, radius, gap = self.clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        #make sure ports are valid
        if not all(1 <= x <=4 for x in ports):
            raise ValueError('Invalid Ports')

        #determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig   = np.cos
            offset = 0
        else:
            trig   = np.sin
            offset = np.pi/2

        #determine z distance
        if 1 in ports and 3 in ports:
            z_dist = 2 * (radius + width/2)
        elif 1 in ports and 4 in ports:
            z_dist = np.pi*radius/2 + radius+width/2
        elif 2 in ports and 4 in ports:
            z_dist = np.pi * radius
        elif 2 in ports and 3 in ports:
            z_dist = np.pi*radius/2 + radius+width/2
        #if it's coming to itself, or to adjacent port
        else:
            return np.zeros(len(wavelength))

        #calculate everything
        B = lambda x: np.pi*x*np.exp(-x)*(special.iv(1,x) + special.modstruve(-1,x))
        xe = ge*(radius + width/2)
        xo = go*(radius + width/2)
        return get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist)

    def gds(self, filename=None, view=False, extra=0, units='nms'):
        #check to make sure the geometry isn't an array
        if len(self.clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == 'nms':
            scale = 1
        elif units == 'microns':
            scale = 10**-3
        else:
            raise ValueError('Invalid units')

        #scale to proper units
        sc_radius = self.radius*scale
        sc_gap    = self.gap*scale
        sc_width  = self.width*scale

        #write to GDS
        pathTop = gdspy.Path(sc_width, (sc_radius, sc_radius+sc_width/2+sc_gap/2+extra))
        pathTop.segment(extra, '-y')
        pathTop.arc(sc_radius, 0, -np.pi)
        pathTop.segment(extra, '+y')

        pathBottom = gdspy.Path(sc_width, (-sc_radius-sc_width/2-extra, -sc_gap/2-sc_width/2))
        pathBottom.segment(2*(sc_radius+sc_width/2+extra), '+x')

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell('C0')
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if view:
            gdspy.LayoutViewer(cells='C0')

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()


class Racetrack(DC):
    def __init__(self, width, thickness, radius, gap, length,  sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.radius = radius
        self.gap    = gap
        self.length = length

    def update(self, **kwargs):
        super().update(**kwargs)
        self.radius = kwargs.get('radius', self.radius)
        self.gap    = kwargs.get('gap', self.gap)
        self.length = kwargs.get('length', self.length)

    def clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.radius, self.gap, self.length))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.radius, self.gap, self.length))

    def predict(self, ports, wavelength):
        wavelength, width, thickness, sw_angle, radius, gap, length = self.clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        #make sure ports are valid
        if not all(1 <= x <=4 for x in ports):
            raise ValueError('Invalid Ports')

        #determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig   = np.cos
            offset = 0
        else:
            trig   = np.sin
            offset = np.pi/2

        #determine z distance
        if 1 in ports and 3 in ports:
            z_dist = 2 * (radius + width/2) + length
        elif 1 in ports and 4 in ports:
            z_dist = np.pi*radius/2 + radius+width/2 + length
        elif 2 in ports and 4 in ports:
            z_dist = np.pi * radius + length
        elif 2 in ports and 3 in ports:
            z_dist = np.pi*radius/2 + radius+width/2 + length
        #if it's coming to itself, or to adjacent port
        else:
            return np.zeros(len(wavelength))

        #calculate everything
        B = lambda x: length*x/(radius+width/2) + np.pi*x*np.exp(-x)*(special.iv(1,x) + special.modstruve(-1,x))
        xe = ge*(radius + width/2)
        xo = go*(radius + width/2)
        return get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist)

    def gds(self, filename=None, view=False, extra=0, units='nms'):
        #check to make sure the geometry isn't an array
        if len(self.clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == 'nms':
            scale = 1
        elif units == 'microns':
            scale = 10**-3
        else:
            raise ValueError('Invalid units')

        #scale to proper units
        sc_radius = self.radius*scale
        sc_gap    = self.gap*scale
        sc_width  = self.width*scale
        sc_length = self.length*scale

        #write to GDS
        pathTop = gdspy.Path(sc_width, (sc_radius+sc_length/2, sc_radius+sc_width/2+sc_gap/2+extra))
        pathTop.segment(extra, '-y')
        pathTop.arc(sc_radius, 0, -np.pi/2)
        pathTop.segment(sc_length, '-x')
        pathTop.arc(sc_radius, -np.pi/2, -np.pi)
        pathTop.segment(extra, '+y')

        pathBottom = gdspy.Path(sc_width, (-sc_radius-sc_width/2-sc_length/2-extra, -sc_gap/2-sc_width/2))
        pathBottom.segment(2*(sc_radius+sc_width/2)+sc_length+2*extra, '+x')

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell('C0')
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if view:
            gdspy.LayoutViewer(cells='C0')

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()

class Straight(DC):
    def __init__(self, width, thickness, gap, length, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.gap    = gap
        self.length = length

    def update(self, **kwargs):
        super().update(**kwargs)
        self.gap    = kwargs.get('gap', self.gap)
        self.length = kwargs.get('length', self.length)

    def clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.gap, self.length))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.gap, self.length))

    def predict(self, ports, wavelength):
        wavelength, width, thickness, sw_angle, gap, length = self.clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        #make sure ports are valid
        if not all(1 <= x <=4 for x in ports):
            raise ValueError('Invalid Ports')

        #determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig   = np.cos
            offset = 0
        else:
            trig   = np.sin
            offset = np.pi/2

        #determine z distance
        if 1 in ports and 3 in ports:
            z_dist = length
        elif 1 in ports and 4 in ports:
            z_dist = length
        elif 2 in ports and 4 in ports:
            z_dist = length
        elif 2 in ports and 3 in ports:
            z_dist = length
        #if it's coming to itself, or to adjacent port
        else:
            return np.zeros(len(wavelength))

        #calculate everything
        B = lambda x: x
        xe = ge*length
        xo = go*length
        return get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist)

    def gds(self, filename=None, view=False, extra=0, units='nms', sbend_h=0, sbend_v=0):
        #check to make sure the geometry isn't an array
        if len(self.clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == 'nms':
            scale = 1
        elif units == 'microns':
            scale = 10**-3
        else:
            raise ValueError('Invalid units')

        #scale to proper units
        sc_width  = self.width*scale
        sc_gap    = self.gap*scale
        sc_length = self.length*scale

        #make parametric functions
        sbend = False
        if sbend_h != 0 and sbend_v != 0:
            sbend = True
        sbendDown = lambda x: (sbend_h*x, -sbend_v/2*(1-np.cos(np.pi*x)))
        sbendUp   = lambda x: (sbend_h*x, sbend_v/2*(1-np.cos(np.pi*x)))
        dsbendDown = lambda x: (sbend_h, -np.pi*sbend_v/2*np.sin(np.pi*x))
        dsbendUp   = lambda x: (sbend_h, np.pi*sbend_v/2*np.sin(np.pi*x))

        #write to GDS
        pathTop = gdspy.Path(sc_width, (-sc_length/2-sbend_h-extra, sbend_v+sc_width/2+sc_gap/2))
        pathTop.segment(extra, '+x')
        if sbend: pathTop.parametric(sbendDown, dsbendDown)
        pathTop.segment(sc_length, '+x')
        if sbend: pathTop.parametric(sbendUp, dsbendUp)
        pathTop.segment(extra, '+x')

        pathBottom = gdspy.Path(sc_width, (-sc_length/2-sbend_h-extra, -sbend_v-sc_width/2-sc_gap/2))
        pathBottom.segment(extra, '+x')
        if sbend: pathBottom.parametric(sbendUp, dsbendUp)
        pathBottom.segment(sc_length, '+x')
        if sbend: pathBottom.parametric(sbendDown, dsbendDown)
        pathBottom.segment(extra, '+x')

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell('C0')
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if view:
            gdspy.LayoutViewer(cells='C0')

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()


class Standard(DC):
    def __init__(self, width, thickness, gap, length, H, V, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.gap    = gap
        self.length = length
        self.H      = H
        self.V      = V

    def update(self, **kwargs):
        super().update(**kwargs)
        self.gap    = kwargs.get('gap', self.gap)
        self.length = kwargs.get('length', self.length)
        self.H      = kwargs.get('H', self.H)
        self.V      = kwargs.get('V', self.V)

    def clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.gap, self.length, self.H, self.V))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.gap, self.length, self.H, self.V))

    def predict(self, ports, wavelength):
        wavelength, width, thickness, sw_angle, gap, length, H, V = self.clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        #make sure ports are valid
        if not all(1 <= x <=4 for x in ports):
            raise ValueError('Invalid Ports')

        #determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig   = np.cos
            offset = 0
        else:
            trig   = np.sin
            offset = np.pi/2

        #determine z distance - length + 2*sbend length
        m      = (V*np.pi/2)**2 / (H**2 + (V*np.pi/2)**2)
        z_dist = length + 2* np.sqrt(H**2 + (V*np.pi/2)**2)/np.pi * special.ellipeinc(np.pi, m)
        if 1 in ports and 3 in ports:
            z_dist = z_dist
        elif 1 in ports and 4 in ports:
            z_dist = z_dist
        elif 2 in ports and 4 in ports:
            z_dist = z_dist
        elif 2 in ports and 3 in ports:
            z_dist = z_dist
        #if it's coming to itself, or to adjacent port
        else:
            return np.zeros(len(wavelength))

        #calculate everything
        B = lambda x: x * (1 + 2*H*np.exp(-V*x/length)*special.iv(0,V*x/length)/length)
        xe = ge*length
        xo = go*length
        return get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist)

    def gds(self, filename=None, view=False, extra=0, units='nms', sbend_h=0, sbend_v=0):
        #check to make sure the geometry isn't an array
        if len(self.clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == 'nms':
            scale = 1
        elif units == 'microns':
            scale = 10**-3
        else:
            raise ValueError('Invalid units')

        #scale to proper units
        sc_width  = self.width*scale
        sc_gap    = self.gap*scale
        sc_length = self.length*scale
        sc_H      = self.H*scale
        sc_V      = self.V*scale

        #make parametric functions
        sbendDown = lambda x: (sc_H*x, -sc_V/2*(1-np.cos(np.pi*x)))
        sbendUp   = lambda x: (sc_H*x, sc_V/2*(1-np.cos(np.pi*x)))
        dsbendDown = lambda x: (sc_H, -np.pi*sc_V/2*np.sin(np.pi*x))
        dsbendUp   = lambda x: (sc_H, np.pi*sc_V/2*np.sin(np.pi*x))

        sbend = False
        if sbend_h != 0 and sbend_v != 0:
            sbend = True
        sbendDownExtra = lambda x: (sbend_h*x, -sbend_v/2*(1-np.cos(np.pi*x)))
        sbendUpExtra   = lambda x: (sbend_h*x, sbend_v/2*(1-np.cos(np.pi*x)))
        dsbendDownExtra = lambda x: (sbend_h, -np.pi*sbend_v/2*np.sin(np.pi*x))
        dsbendUpExtra   = lambda x: (sbend_h, np.pi*sbend_v/2*np.sin(np.pi*x))

        #write to GDS
        pathTop = gdspy.Path(sc_width, (-sc_length/2-sc_H-sbend_h-extra, sc_V+sbend_v+sc_width/2+sc_gap/2))
        pathTop.segment(extra, '+x')
        if sbend: pathTop.parametric(sbendDownExtra, dsbendDownExtra)
        pathTop.parametric(sbendDown, dsbendDown)
        pathTop.segment(sc_length, '+x')
        pathTop.parametric(sbendUp, dsbendUp)
        if sbend: pathTop.parametric(sbendUpExtra, dsbendUpExtra)
        pathTop.segment(extra, '+x')

        pathBottom = gdspy.Path(sc_width, (-sc_length/2-sc_H-sbend_h-extra, -sc_V-sbend_v-sc_width/2-sc_gap/2))
        pathBottom.segment(extra, '+x')
        if sbend: pathBottom.parametric(sbendUpExtra, dsbendUpExtra)
        pathBottom.parametric(sbendUp, dsbendUp)
        pathBottom.segment(sc_length, '+x')
        pathBottom.parametric(sbendDown, dsbendDown)
        if sbend: pathBottom.parametric(sbendDownExtra, dsbendDownExtra)
        pathBottom.segment(extra, '+x')

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell('C0')
        path_cell.add(pathTop)
        path_cell.add(pathBottom)

        if view:
            gdspy.LayoutViewer(cells='C0')

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()

class DoubleRR(DC):
    def __init__(self, width, thickness, radius, gap, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.radius = radius
        self.gap    = gap

    def update(self, **kwargs):
        super().update(**kwargs)
        self.radius = kwargs.get('radius', self.radius)
        self.gap    = kwargs.get('gap', self.gap)

    def clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.radius, self.gap))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.radius, self.gap))

    def predict(self, ports, wavelength):
        wavelength, width, thickness, sw_angle, radius, gap = self.clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        #make sure ports are valid
        if not all(1 <= x <=4 for x in ports):
            raise ValueError('Invalid Ports')

        #determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig   = np.cos
            offset = 0
        else:
            trig   = np.sin
            offset = np.pi/2

        #determine z distance
        if 1 in ports and 3 in ports:
            z_dist = np.pi * radius
        elif 1 in ports and 4 in ports:
            z_dist = np.pi * radius
        elif 2 in ports and 4 in ports:
            z_dist = np.pi * radius
        elif 2 in ports and 3 in ports:
            z_dist = np.pi * radius
        #if it's coming to itself, or to adjacent port
        else:
            return np.zeros(len(wavelength))

        #calculate everything
        B = lambda x: 0.5*np.pi*2*x*np.exp(-2*x)*(special.iv(1,2*x) + special.modstruve(-1,2*x))
        xe = ge*(radius + width/2)
        xo = go*(radius + width/2)
        return get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist)

    def gds(filename, self, extra=0, units='nm'):
        raise NotImplemented('TODO: Write to GDS file')


class AngledRR(DC):
    def __init__(self, width, thickness, radius, gap, theta, sw_angle=90):
        super().__init__(width, thickness, sw_angle)
        self.radius = radius
        self.gap    = gap
        self.theta  = theta

    def update(self, **kwargs):
        super().update(**kwargs)
        self.radius = kwargs.get('radius', self.radius)
        self.gap    = kwargs.get('gap', self.gap)
        self.theta  = kwargs.get('theta', self.theta)

    def clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.radius, self.gap, self.theta))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.radius, self.gap, self.theta))

    def predict(self, ports, wavelength):
        wavelength, width, thickness, sw_angle, radius, gap, theta = self.clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        #make sure ports are valid
        if not all(1 <= x <=4 for x in ports):
            raise ValueError('Invalid Ports')

        #determine if cross or through port
        if abs(ports[1] - ports[0]) == 2:
            trig   = np.cos
            offset = 0
        else:
            trig   = np.sin
            offset = np.pi/2

        #determine z distance
        if 1 in ports and 3 in ports:
            z_dist = np.pi*(radius + width + gap)
        elif 1 in ports and 4 in ports:
            z_dist = np.pi*(radius + width + gap)/2 + np.pi*radius/2
        elif 2 in ports and 4 in ports:
            z_dist = np.pi * radius
        elif 2 in ports and 3 in ports:
            z_dist = np.pi*(radius + width + gap)/2 + np.pi*radius/2
        #if it's coming to itself, or to adjacent port
        else:
            return np.zeros(len(wavelength))

        #calculate everything
        B = lambda x: x
        xe = ge * theta * (radius + width/2 + gap/2)
        xo = go * theta * (radius + width/2 + gap/2)
        return get_closed_ans(ae, ao, ge, go, neff, wavelength, gap, B, xe, xo, offset, trig, z_dist)

    def gds(filename, self, extra=0, units='nm'):
        raise NotImplemented('TODO: Write to GDS file')

class Waveguide(ABC):
    """Lossless model for a straight waveguide. Ports are numbered as:

                1 ============== 2
    """
    def __init__(self, width, thickness, length, sw_angle=90):
        self.width      = width
        self.thickness  = thickness
        self.length     = length
        self.sw_angle   = sw_angle

    def clean_args(self, wavelength):
        """Makes sure all attributes are the same size"""
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.length))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.length))

    def update(self, **kwargs):
        """Takes in any parameter defined by __init__ and changes it."""
        self.width      = kwargs.get('width', self.width)
        self.thickness  = kwargs.get('thickness', self.thickness)
        self.length     = kwargs.get('thickness', self.length)
        self.sw_angle   = kwargs.get('sw_angle', self.sw_angle)

    def sparams(self, wavelength):
        """Returns sparams
        Args:
            wavelength(float/np.ndarray): wavelengths to get sparams at
        Returns:
            freq     (np.ndarray): frequency for s_matrix in Hz, size n (number of wavelength points)
            s_matrix (np.ndarray): size (4,4,n) complex matrix of scattering parameters
        """
        #get number of points to evaluate at
        if np.isscalar(wavelength):
            n = 1
        else:
            n = len(wavelength)

        #check to make sure the geometry isn't an array
        if len(self.clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, getting sparams doesn't make sense")
        s_matrix = np.zeros((2,2,n), dtype='complex')

        #calculate upper half of matrix (diagonal is 0)
        s_matrix[0,1] = self.predict(wavelength, (1,2))

        #apply symmetry (note diagonal is 0, no need to subtract it)
        s_matrix += np.transpose(s_matrix, (1,0,2))
        freq = C/(wavelength*10**-9)

        #flip them so frequency is increasing
        if n != 1:
            freq = freq[::-1]
            s_matrix = s_matrix[:,:,::-1]

        #transpose so depth comes first
        s_matrix = np.transpose(s_matrix, (2, 0, 1))
        return (freq, s_matrix)

    def predict(self, wavelength, ports=(1,2)):
        """Predicts the output when light is put in the bottom left port (see diagram above)

        Args:
            wavelength (float/np.ndarray): wavelength(s) to predict at
            ports               (2-tuple): Specifies the port coming in and coming out

        Returns:
            k/t (complex np.ndarray): returns the value of the light coming through"""
        wavelength, width, thickness, sw_angle, length = self.clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)

        #make sure ports are valid
        if not all(1 <= x <= 2 for x in ports):
            raise ValueError('Invalid Ports')

        #calculate everything
        z_dist = self.length
        phase = 2*z_dist*neff*np.pi/wavelength

        return np.exp(-1j*phase)

    def gds(self, filename=None, extra=0, units='microns', view=False):
        """Writes the geometry to the gds file

        Args:
            filename (str): location to save file to, or if you don't want to defaults to None
            extra    (int): extra straight portion to add to ends of waveguides to make room in simulation
                                (input with units same as units input)
            units    (str): either 'microns' or 'nms'. Units to save gds file in
        """
                #check to make sure the geometry isn't an array
        if len(self.clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")

        if units == 'nms':
            scale = 1
        elif units == 'microns':
            scale = 10**-3
        else:
            raise ValueError('Invalid units')

        #scale to proper units
        sc_width  = self.width*scale
        sc_length = self.length*scale

        #write to GDS
        path = gdspy.Path(sc_width, (-sc_length/2-extra, 0))
        path.segment(2*extra+sc_length, '+x')

        gdspy.current_library = gdspy.GdsLibrary()
        path_cell = gdspy.Cell('C0')
        path_cell.add(path)

        if view:
            gdspy.LayoutViewer(cells='C0')

        if filename is not None:
            writer = gdspy.GdsWriter(filename, unit=1.0e-6, precision=1.0e-9)
            writer.write_cell(path_cell)
            writer.close()
