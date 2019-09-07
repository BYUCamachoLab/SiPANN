import numpy as np
from abc import ABC, abstractmethod 
import scipy.integrate as integrate
import scipy.special as special
import pkg_resources
import joblib
import gdspy
from SiPANN.dc import *
from SiPANN.SiP import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


'''
Similarly to before, we initialize all ANN's and regressions as global objects to speed things up. 
'''
cross_file = pkg_resources.resource_filename('SiPANN','LR/DC_coeffs.joblib')
DC_coeffs  = joblib.load(cross_file)

cross_file = pkg_resources.resource_filename('SiPANN','LR/R_bent_wide.joblib')
R_bent     = joblib.load(cross_file)

C          = 299792458

'''
Racetrack waveguide arc used to connect to a racetrack directional coupler
          -------
        /         \
        \         /
          -------
   1 ----------------- 2

'''

class racetrack_sb_rr():
    def __init__(self, width, thickness, radius, gap, length, sw_angle=90, loss=[0.99]):
        self.width = width
        self.thickness = thickness
        self.radius = radius
        self.gap = gap
        self.length = length
        self.sw_angle = sw_angle
        self.loss = loss
        
    def update(self, **kwargs):
        self.width = kwargs.get('width', self.width)
        self.thickness    = kwargs.get('thickness', self.thickness)
        self.radius = kwargs.get('radius', self.radius)
        self.gap    = kwargs.get('gap', self.gap)
        self.length = kwargs.get('length', self.length)
        self.sw_angle = kwargs.get('sw_angle', self.sw_angle)
        
    def clean_args(self, wavelength):
        if wavelength is None:
            return clean_inputs((self.width, self.thickness, self.sw_angle, self.radius, self.gap, self.length))
        else:
            return clean_inputs((wavelength, self.width, self.thickness, self.sw_angle, self.radius, self.gap, self.length))   
        
    def predict(self, wavelength):
        wavelength, width, thickness, sw_angle, radius, gap, length = self.clean_args(wavelength)
        ae, ao, ge, go, neff = get_coeffs(wavelength, width, thickness, sw_angle)
        
        rr = Racetrack(self.width, self.thickness, self.radius, self.gap, self.length)
        k = rr.predict((1,4), wavelength)
        t = rr.predict((1,3), wavelength)
            
        #pull phase from coupler section
        phi_t = -np.unwrap(np.angle(t))

        #pull coupling from coupler section
        t_mag = np.abs(t)

        #pull phase from bent sections
        TE0_B = np.squeeze(bentWaveguide(wavelength=wavelength*1e-3,width=self.width*1e-3,thickness=self.thickness*1e-3,angle=self.sw_angle,radius=self.radius*1e-3))
        L_b = np.pi * radius #length of two bent waveguides (half the circle)
        phi_b = np.unwrap(2*np.pi*np.real(TE0_B) / wavelength) * (L_b)

        #pull phase from straight sections
        TE0 = np.squeeze(straightWaveguide(wavelength=wavelength*1e-3,width=self.width*1e-3,thickness=self.thickness*1e-3,angle=self.sw_angle))
        L_s = length #length of the coupler regiod
        phi_s = np.unwrap(2*np.pi*np.real(TE0) / wavelength) * L_s

        #get total phase
        phi = phi_t + phi_b + phi_s

        #calculate loss
        #lossTemp = self.loss.copy()
        #lossTemp[-1] = loss[-1] # assume uniform loss
        lossPoly = np.poly1d(self.loss)
        alpha = lossPoly(wavelength)

        #transfer function of resonator
        E = (t_mag - alpha*np.exp(1j*phi)) / (1-alpha*t_mag*np.exp(1j*phi)) * np.exp(-1j*phi)

        return E, alpha, t, phi
    
    def gds(self, filename=None, view=False, extra=0, units='nms'):
        #check to make sure the geometry isn't an array    
        if len(self.clean_args(None)[0]) != 1:
            raise ValueError("You have changing geometries, making gds doesn't make sense")
            
        if units == 'nms':
            scale = 1e-3
        elif units == 'microns':
            scale = 1
        else:
            raise ValueError('Invalid units')
            
        #scale to proper units
        sc_radius = self.radius*scale
        sc_gap    = self.gap*scale
        sc_width  = self.width*scale
        sc_length = self.length*scale
        
        #write to GDS
        pathTop = gdspy.Path(sc_width, (-sc_length/2, 2*sc_radius+sc_width/2+sc_gap/2))
        pathTop.segment(sc_length, '+x')
        pathTop.turn(sc_radius, 'rr')
        pathTop.segment(sc_length, '-x')
        pathTop.turn(sc_radius, 'rr')
        
        pathBottom = gdspy.Path(sc_width, (-sc_radius-sc_width/2-sc_length/2, -sc_gap/2-sc_width/2))
        pathBottom.segment(2*(sc_radius+sc_width/2)+sc_length, '+x')
        
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