# -*- coding: utf-8 -*-
#
# Copyright © Simphony Project Contributors
# Licensed under the terms of the MIT License
# (see simphony/__init__.py for details)

"""
simphony.library.sipann
=======================

This package contains the models for the SiPANN integration.
"""

import os
import pickle
from itertools import combinations_with_replacement as comb_w_r

# import ctypes
from numba import njit
from numba.extending import get_cython_function_address
import ctypes
import numpy as np
from scipy import special
from SiPANN import scee

from simphony.elements import Model
from simphony.tools import freq2wl, wl2freq, interpolate

#########################################################################################
# Integral Estimators. Make any coupling device as desired.
#########################################################################################
class GapFuncAntiSymmetric(Model):
    """Regression Based Solution for arbitrarily shaped anti-symmetric coupler

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    gap : function
        Gap function as one progresses along the waveguide (nm)
    zmin : float
        Where to begin integration in the gap function (nm)
    zmax : float
        Where to end integration in the gap function (nm)
    arc1, arc2, arc3, arc4 : float
        Arclength from entrance of each port till minimum coupling point (nm)
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, gap, zmin, zmax, arc1, arc2, arc3, arc4, width=0.5, thickness=0.22, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap
        self.zmin      = zmin
        self.zmax      = zmax
        self.arc1      = arc1
        self.arc2      = arc2
        self.arc3      = arc3
        self.arc4      = arc4
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized arbitrarily shaped anti-symmetric coupler.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.GapFuncAntiSymmetric(width=self.width, thickness=self.thickness, gap=self.gap, zmin=self.zmin, zmax=self.zmax, arc1=self.arc1, arc2=self.arc2, arc3=self.arc3, arc4=self.arc4, sw_angle=self.sw_angle)
        return item.sparams(wl)

class GapFuncSymmetric(Model):
    """Regression Based Solution for arbitrarily shaped symmetric coupler

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    gap : function
        Gap function as one progresses along the waveguide (nm)
    dgap : function
        Derivative of the gap function (nm)
    zmin : float
        Where to begin integration in the gap function (nm)
    zmax : float
        Where to end integration in the gap function (nm)
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, gap, dgap, zmin, zmax, width=0.5, thickness=0.22, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap
        self.dgap      = dgap
        self.zmin      = zmin
        self.zmax      = zmax
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized arbitrarily shaped symmetric coupler.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.GapFuncSymmetric(width=self.width, thickness=self.thickness, gap=self.gap, dgap=self.gap, zmin=self.zmin, zmax=self.zmax, sw_angle=self.sw_angle)
        return item.sparams(wl)

#########################################################################################
# All the Different types of DC's with closed form solutions. These will be faster than defining it manually in the function form above.
#########################################################################################
class StraightCoupler(Model):
    """Regression Based Closed Form solution of parallel straight waveguides

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    gap : float
        Distance between the two waveguides edge in microns.
    length : float
        Length of both waveguides in microns.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, gap=0.1, length=10.0, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap*1000
        self.length    = length*1000
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of parameterized parallel waveguides.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.StraightCoupler(width=self.width, thickness=self.thickness, gap=self.gap, length=self.length, sw_angle=self.sw_angle)
        return item.sparams(wl)

class HalfRacetrack(Model):
    """Regression Based Closed Form solution of half of a racetrack ring resonator

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    radius : float
        Distance from center of ring to middle of waveguide in microns.
    gap : float
        Minimum distance from ring waveguide edge to straight waveguide edge in microns.
    length : float
        Length of straight portion of ring waveguide in microns.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, gap=0.1, radius=10.0, length=2.5, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap*1000
        self.radius    = radius*1000
        self.length    = length*1000
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized half racetrack ring.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.HalfRacetrack(width=self.width, thickness=self.thickness, radius=self.radius, gap=self.gap, length=self.length, sw_angle=self.sw_angle)
        return item.sparams(wl)

class HalfRing(Model):
    """Regression Based Closed Form solution of half of a ring resonator

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    gap : float  
        Gap between the two waveguides in microns.
    radius : float  
        Radius of bent portions of waveguide
    sw_angle : float  
        Angle in degrees of sidewall of waveguide (between 80 and 90)
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, gap=0.1, radius=10.0, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap*1000
        self.radius    = radius*1000
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized half ring.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.HalfRing(width=self.width, thickness=self.thickness, radius=self.radius, gap=self.gap, sw_angle=self.sw_angle)
        return item.sparams(wl)

class Standard(Model):
    """Regression Based Closed Form solution of a standard shaped directional coupler

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness : float
        Thickness of waveguide in microns
    gap : float
        Minimum distance between the two waveguides edge in microns.
    length : float
        Length of the straight portion of both waveguides in microns.
    H : float
        Horizontal distance between end of coupler until straight portion in microns.
    H : float
        Vertical distance between end of coupler until straight portion in microns.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, gap=0.1, length=5.0, H=2.0, V=2.0, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap*1000
        self.length    = length*1000
        self.H         = H*1000
        self.V         = V*1000
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized standard directional coupler.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.Standard(width=self.width, thickness=self.thickness, gap=self.gap, length=self.length, H=self.H, V=self.V, sw_angle=self.sw_angle)
        return item.sparams(wl)

class DoubleHalfRing(Model):
    """Regression Based Closed Form solution of 2 coupling half rings

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    gap : float  
        Gap between the two waveguides in microns.
    gap : float
            Minimum distance from ring waveguide edge to other ring waveguide edge in microns.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, gap=0.1, radius=10.0, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap*1000
        self.radius    = radius*1000
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of parameterized 2 coupling half rings.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.DoubleHalfRing(width=self.width, thickness=self.thickness, radius=self.radius, gap=self.gap, sw_angle=self.sw_angle)
        return item.sparams(wl)

class AngledHalfRing(Model):
    """Regression Based Closed Form solution of half of a ring resonator pushed into a 
    straight coupling waveguide.

    Parameters
    ----------
    width : float  
        Width of the waveguide in microns.
    thickness: float  
        Thickness of the waveguide in microns.
    gap : float  
        Gap between the two waveguides in microns.
    radius : float  
        Radius of bent portions of waveguide
    theta : float
        Angle that the straight waveguide is curved in radians (???).
    sw_angle : float  
        Angle in degrees of sidewall of waveguide (between 80 and 90)
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, gap=0.1, radius=10.0, theta=np.pi/4, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.gap       = gap*1000
        self.radius    = radius*1000
        self.theta     = theta
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized angled half ring.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.AngledHalfRing(width=self.width, thickness=self.thickness, radius=self.radius, gap=self.gap, theta=self.theta, sw_angle=self.sw_angle)
        return item.sparams(wl)

class Waveguide(Model):
    """Lossless model for a straight waveguide. 
    
    Simple model that makes sparameters for a straight waveguide. May not be 
    the best option, but plays nice with other models in SCEE. Ports are numbered as::

        |  1 ----------- 2   |

    Parameters
    ----------
    width : float
        Width of the waveguide in microns
    thickness : float
        Thickness of waveguide in microns
    length : float
        Length of waveguide in microns.
    sw_angle : float, optional
        Sidewall angle of waveguide from horizontal in degrees. Defaults to 90.
    """
    pins = ('n1', 'n2') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def __init__(self, width=0.5, thickness=0.22, length=10.0, sw_angle=90.0):
        #resize everything to nms
        self.width     = width*1000
        self.thickness = thickness*1000
        self.length    = length*1000
        self.sw_angle  = sw_angle

    def s_parameters(self, freq):
        """
        Get the s-parameters of a parameterized waveguide.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        wl = freq2wl(freq) * 1e9

        item = scee.Waveguide(width=self.width, thickness=self.thickness, length=self.length, sw_angle=self.sw_angle)
        return item.sparams(wl)


#########################################################################################
# These couplers were made via optimization. It may be wise to run them once and save them using the simphony exporter
#########################################################################################
class Crossover1550(Model):
    """Regression Based form of a crossover at lambda=1550nm

    Regression based form of a 100/0 directional coupler.
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.

    def s_parameters(self, freq):
        """Get the s-parameters of a parameterized 50/50 directional coupler.
        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        #load and make gap function
        loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'COUPLER', 'crossover1550.npz'))
        x = loaded['GAP']
        b = loaded['LENGTH']

        #load scipy.special.binom as a C-compiled function
        addr = get_cython_function_address("scipy.special.cython_special", "binom")
        functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
        binom_fn = functype(addr)

        #load all seperate functions that we'll need
        n = len(x) - 1
        @njit
        def binom_in_njit(x, y):
            return binom_fn(x, y)
        @njit
        def bernstein(n,j,t):
            return binom_in_njit(n, j) * t ** j * (1 - t) ** (n - j)
        @njit
        def bez(t):
            n = len(x) - 1
            return np.sum(np.array([(x[j])*bernstein(n,j,t/b) for j in range(len(x))]),axis=0)
        @njit
        def dbez(t):
            return np.sum(np.array([n*(x[j])*(bernstein(n-1,j-1,t/b)-bernstein(n-1,j,t/b)) for j in range(len(x))]),axis=0)/b

        #resize everything to nms
        width     = 500
        thickness = 220

        #switch to wavelength
        wl = freq2wl(freq) * 1e9

        item = scee.GapFuncSymmetric(width, thickness, bez, dbez, 0, b)
        return item.sparams(wl)

class FiftyFifty(Model):
    """Regression Based form of a 50/50 directional coupler at lambda=1550nm
    """
    pins = ('n1', 'n2', 'n3', 'n4') #: The default pin names of the device
    freq_range = (182800279268292.0, 205337300000000.0) #: The valid frequency range for this model.


    def s_parameters(self, freq):
        """Get the s-parameters of a parameterized 50/50 directional coupler.
        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        #load and make gap function
        loaded = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'COUPLER', 'fiftyfifty.npz'))
        x = loaded['GAP']
        b = loaded['LENGTH']

        #load scipy.special.binom as a C-compiled function
        addr = get_cython_function_address("scipy.special.cython_special", "binom")
        functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
        binom_fn = functype(addr)

        #load all seperate functions that we'll need
        n = len(x) - 1
        @njit
        def binom_in_njit(x, y):
            return binom_fn(x, y)
        @njit
        def bernstein(n,j,t):
            return binom_in_njit(n, j) * t ** j * (1 - t) ** (n - j)
        @njit
        def bez(t):
            n = len(x) - 1
            return np.sum(np.array([(x[j])*bernstein(n,j,t/b) for j in range(len(x))]),axis=0)
        @njit
        def dbez(t):
            return np.sum(np.array([n*(x[j])*(bernstein(n-1,j-1,t/b)-bernstein(n-1,j,t/b)) for j in range(len(x))]),axis=0)/b

        #resize everything to nms
        width     = 500
        thickness = 220

        #switch to wavelength
        wl = freq2wl(freq) * 1e9

        item = scee.GapFuncSymmetric(width, thickness, bez, dbez, 0, b)
        return item.sparams(wl)


