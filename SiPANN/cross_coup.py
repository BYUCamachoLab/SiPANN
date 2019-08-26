import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scipy.integrate as integrate
import scipy.special as special
import pkg_resources
import joblib


'''
Similarly to before, we initialize all ANN's and regressions as global objects to speed things up. 
'''
cross_file = pkg_resources.resource_filename('SiPANN','LR/DC_coeffs.joblib')
DC_coeffs = joblib.load(cross_file)

cross_file = pkg_resources.resource_filename('SiPANN','LR/R_bent_wide.joblib')
R_bent = joblib.load(cross_file)

"""
All the closed form solutions for Different Structures
"""
#part terms can be mag, ph, or both
def straight(wave, width, thickness, length, gap, sw_angle=90, term='k', part='mag'):
    """Return coupling coefficients for a Racetrack Resonator
    
    Args:
        wave      (float/np.ndarray): wavelength in nm (1450 - 1650nm)
        width     (float/np.ndarray): width of waveguide in nm (400 - 600nm)
        thickness (float/np.ndarray): thickness of waveguide in nm (180 - 240nm)
        gap       (float/np.ndarray): gap between waveguides in nm (above 100nm)
        length    (float/np.ndarray): Length of waveguides in nm
        sw_angle  (float/np.ndarray): angle of waveguide walls in degrees (between 80 and 90)
        term      (str): either 't' or 'k'
        part      (str): 'mag', 'phase', or 'both'. Choose both if you want both phase and magnitude
        
    Returns:
        (complex np.ndarray): The coupling coefficient"""
        
    #clean everything
    wave, width, thickness, length, gap = clean_inputs((wave, width, thickness, sw_angle, length, gap))
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness, sw_angle)
    
    #set up closed form solutions
    B = lambda x: x
    xe = ge*length
    xo = go*length
    z = length

    #get closed form solution
    return get_closed_ans(ae, ao, ge, go, neff, wave, B, xe, xo, z, gap, term, part)
  
    

def curved(wave, width, thickness, length, gap, H, V, sw_angle=90, term='k', part='mag'):
    """Return coupling coefficients for a Racetrack Resonator
    
    Args:
        wave      (float/np.ndarray): wavelength in nm (1450 - 1650nm)
        width     (float/np.ndarray): width of waveguide in nm (400 - 600nm)
        thickness (float/np.ndarray): thickness of waveguide in nm (180 - 240nm)
        H         (float/np.ndarray): horizontal distance of s-bend in nm
        V         (float/np.ndarray): vertical distance of s-bend in nm
        gap       (float/np.ndarray): gap between bus and ring in nm(above 100nm)
        sw_angle  (float/np.ndarray): angle of waveguide walls in degrees (between 80 and 90)
        term      (str): either 't' or 'k'
        part      (str): 'mag', 'phase', or 'both'. Choose both if you want both phase and magnitude
        
    Returns:
        (complex np.ndarray): The coupling coefficient"""
        
    #clean everything
    wave, width, thickness, sw_angle, length, gap, H, V = clean_inputs((wave, width, thickness, sw_angle, length, gap, H, V))
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness, sw_angle)
    
    #calculate everything
    B = lambda x: x*(1 + 2*H*np.exp(-V*x/L)*special.iv(0,V*x/L)/L)
    xe = ge*length
    xo = go*length
    z = 2*H + L

    #get closed form solution
    return get_closed_ans(ae, ao, ge, go, neff, wave, B, xe, xo, z, gap, term, part)

def racetrack(wave, width, thickness, radius, gap, length, sw_angle=90, term='k', part='mag'):
    """Return coupling coefficients for a Racetrack Resonator
    
    Args:
        wave      (float/np.ndarray): wavelength in nm (1450 - 1650nm)
        width     (float/np.ndarray): width of waveguide in nm (400 - 600nm)
        thickness (float/np.ndarray): thickness of waveguide in nm (180 - 240nm)
        radius    (float/np.ndarray): radius of ring in nm 
        gap       (float/np.ndarray): gap between bus and ring in nm(above 100nm)
        length    (float/np.ndarray): Length of straight portion of resonator in nm
        sw_angle  (float/np.ndarray): angle of waveguide walls in degrees (between 80 and 90)
        term      (str): either 't' or 'k'
        part      (str): 'mag', 'phase', or 'both'. Choose both if you want both phase and magnitude 
        
    Returns:
        (complex np.ndarray): The coupling coefficient"""
    #clean everything
    wave, width, thickness, sw_angle, radius, gap, length = clean_inputs((wave, width, thickness, sw_angle, radius, gap, length))
    #get coefficients
    ae, ao, ge, go, neff_str = get_coeffs(wave, width, thickness, sw_angle)
    neff_bent = R_bent.predict( np.column_stack((wave/1000, width/1000, thickness/1000, radius/1000, sw_angle)) )
    
    #calculate everything
    B = lambda x: length*x/(radius+width/2) + np.pi*x*np.exp(-x)*(special.iv(1,x) + special.modstruve(-1,x))
    xe = ge*(radius + width/2)
    xo = go*(radius + width/2)
    z = 2*(radius + width) + length

    #get closed form solution
    #determine which parameter to get
    if term == 'k':
        trig = np.sin
        offset = np.pi/2
        z_str = (radius + width/2 + length)
        z_bent = np.pi*radius / 2
    elif term == 't':
        trig = np.cos
        offset = 0
        z_str = 2*(radius + width/2) + length
        z_bent = 0
    else:
        raise ValueError("Bad term parameter")
    
     #calculate magnitude
    if part == 'mag' or part == 'both':
        temp = ae*np.exp(-ge*gap)*B(xe)/ge + ao*np.exp(-go*gap)*B(xo)/go
        mag =  trig( temp*np.pi / wave )
    
    #calculate phase
    if part == 'ph' or part == 'both':
        temp = ae*np.exp(-ge*gap)*B(xe)/ge - ao*np.exp(-go*gap)*B(xo)/go + 2*(z_str*neff_str + z_bent*neff_bent)
        phase = (temp*np.pi/wave + offset)
    
    if part == 'mag':
        phase = 0
    if part == 'ph':
        mag = 1

    return mag*np.exp(-1j*phase)


def rr(wave, width, thickness, radius, gap, sw_angle=90, term='k', part='mag'):
    """Return coupling coefficients for a Ring Resonator
    
    Args:
        wave      (float/np.ndarray): wavelength in nm (1450 - 1650nm)
        width     (float/np.ndarray): width of waveguide in nm (400 - 600nm)
        thickness (float/np.ndarray): thickness of waveguide in nm (180 - 240nm)
        radius    (float/np.ndarray): radius of ring in nm 
        gap       (float/np.ndarray): gap between bus and ring (above 100nm)
        sw_angle  (float/np.ndarray): angle of waveguide walls in degrees (between 80 and 90)
        term      (str): either 't' or 'k'
        part      (str): 'mag', 'phase', or 'both'. Choose both if you want both phase and magnitude
        
    Returns:
        (complex np.ndarray): The coupling coefficient
        """
    #clean everything
    wave, width, thickness, sw_angle, radius, gap = clean_inputs((wave, width, thickness, sw_angle, radius, gap))
    #get coefficients
    ae, ao, ge, go, neff_str = get_coeffs(wave, width, thickness, sw_angle)
    neff_bent = R_bent.predict( np.column_stack((wave/1000, width/1000, thickness/1000, radius/1000, sw_angle)) )
    
    #calculate everything
    B = lambda x: np.pi*x*np.exp(-x)*(special.iv(1,x) + special.modstruve(-1,x))
    xe = ge*(radius + width/2)
    xo = go*(radius + width/2)

    #get closed form solution
    #determine which parameter to get
    if term == 'k':
        trig = np.sin
        offset = np.pi/2
        z_str = (radius + width/2)
        z_bent = np.pi*radius / 2
    elif term == 't':
        trig = np.cos
        offset = 0
        z_str = 2*(radius + width/2)
        z_bent = 0
    else:
        raise ValueError("Bad term parameter")
    
     #calculate magnitude
    if part == 'mag' or part == 'both':
        temp = ae*np.exp(-ge*gap)*B(xe)/ge + ao*np.exp(-go*gap)*B(xo)/go
        mag =  trig( temp*np.pi / wave )
    
    #calculate phase
    if part == 'ph' or part == 'both':
        temp = ae*np.exp(-ge*gap)*B(xe)/ge - ao*np.exp(-go*gap)*B(xo)/go + 2*(z_str*neff_str + z_bent*neff_bent)
        phase = (temp*np.pi/wave + offset)
    
    if part == 'mag':
        phase = 0
    if part == 'ph':
        mag = 1

    return mag*np.exp(-1j*phase)


def double_rr(wave, width, thickness, radius, gap, sw_angle=90, term='k', part='mag'):
    """Return coupling coefficients for a Racetrack Resonator
    
    Args:
        wave      (float/np.ndarray): wavelength in nm (1450 - 1650nm)
        width     (float/np.ndarray): width of waveguide in nm (400 - 600nm)
        thickness (float/np.ndarray): thickness of waveguide in nm (180 - 240nm)
        radius    (float/np.ndarray): radius of rings in nm 
        gap       (float/np.ndarray): gap between rings in nm(above 100nm)
        sw_angle  (float/np.ndarray): angle of waveguide walls in degrees (between 80 and 90)
        term      (str): either 't' or 'k'
        part      (str): 'mag', 'phase', or 'both'. Choose both if you want both phase and magnitude
        
    Returns:
        (complex np.ndarray): The coupling coefficient"""
        
    #clean everything
    wave, width, thickness, sw_angle, radius, gap = clean_inputs((wave, width, thickness, sw_angle, radius, gap))
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness, sw_angle)
    
    #calculate everything
    B = lambda x: (np.pi*2*x*np.exp(-2*x)*(special.iv(1,2*x) + special.modstruve(-1,2*x)))/2
    xe = ge*(radius + width/2)
    xo = go*(radius + width/2)
    z = 2*(radius + width)

    #get closed form solution
    return get_closed_ans(ae, ao, ge, go, neff, wave, B, xe, xo, z, gap, term, part)



def pushed_rr(wave, width, thickness, radius, d, theta, sw_angle=90, term='k', part='mag'):
    """Return coupling coefficients for a Racetrack Resonator
    
    Args:
        wave      (float/np.ndarray): wavelength in nm (1450 - 1650nm)
        width     (float/np.ndarray): width of waveguide in nm (400 - 600nm)
        thickness (float/np.ndarray): thickness of waveguide in nm (180 - 240nm)
        radius    (float/np.ndarray): radius of ring in nm 
        d         (float/np.ndarray): gap between bus and ring in nm (above 100nm)
        theta     (float/np.ndarray): total angle of pushed region
        sw_angle  (float/np.ndarray): angle of waveguide walls in degrees (between 80 and 90)
        term      (str): either 't' or 'k'
        part      (str): 'mag', 'phase', or 'both'. Choose both if you want both phase and magnitude
        
    Returns:
        (complex np.ndarray): The coupling coefficient"""
        
    #clean everything
    wave, width, thickness, sw_angle, radius, d, theta = clean_inputs((wave, width, thickness, sw_angle, radius, d, theta))
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness, sw_angle)
    
    #calculate everything
    B = lambda x: x
    xe = ge*theta*(radius + width/2 + d/2)
    xo = go*theta*(radius + width/2 + d/2)
    z = 2*(radius + width)

    #get closed form solution
    return get_closed_ans(ae, ao, ge, go, neff, wave, B, xe, xo, z, gap, term, part)


"""
The most important one, it takes in a function of gap size and a range to sweep over
"""
def any_gap(wave, width, thickness, g, dg, zmin, zmax, sw_angle=90, term='k', part='both'):
    """Return coupling coefficients for a Racetrack Resonator
    
    Args:
        wave      (float/np.ndarray): wavelength in nm (1450 - 1650nm)
        width     (float/np.ndarray): width of waveguide in nm (400 - 600nm)
        thickness (float/np.ndarray): thickness of waveguide in nm (180 - 240nm)
        g         (function): function that takes in a single float, and returns gap distance in nm
        dg        (function): function that takes in a single float, and returns derivative of gap
        zmin      (float/np.ndarray): Initial point of directional coupler in nm
        zmax      (float/np.ndarray): Final point of directional coupler in nm
        sw_angle  (float/np.ndarray): angle of waveguide walls in degrees (between 80 and 90)
        term      (str): either 't' or 'k'
        part      (str): 'mag', 'phase', or 'both'. Choose both if you want both phase and magnitude
        
    Returns:
        (complex np.ndarray): The coupling coefficient"""
        
    #determine which parameter to get
    if term == 'k':
        trig = np.sin
        offset = np.pi/2
    elif term == 't':
        trig = np.cos
        offset = 0
    else:
        raise ValueError("Bad term parameter")

    #clean everything
    if np.ndim(g(0)) == 0:
        wave, width, thickness, sw_angle = clean_inputs((wave, width, thickness, sw_angle))
    else:
        wave, width, thickness, sw_angle, _ = clean_inputs((wave, width, thickness, sw_angle, g(0)))
    n = len(wave)
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness, sw_angle)
    
    #find arcLength if needed
    if part == 'ph' or part == 'both':
        arcFormula = lambda x: np.sqrt( 1 + dg(x)**2 )
        arcL = integrate.quad(arcFormula, zmin, zmax)[0]    
        
    #if g has many lengths to sweep over
    if np.ndim(g(0)) == 0:
        mag = np.zeros(n)
        phase = np.zeros(n)
        for i in range(n):
            #get mag
            if part == 'mag' or part == 'both':
                f = lambda z: float(ae[i]*np.exp(-ge[i]*g(z)) + ao[i]*np.exp(-go[i]*g(z)))
                mag[i] = trig( np.pi*integrate.quad(f, zmin, zmax)[0]/wave[i] )

            #get phase
            if part == 'ph' or part == 'both':
                f = lambda z: float(ae[i]*np.exp(-ge[i]*g(z)) - ao[i]*np.exp(-go[i]*g(z)))
                phase[i] = np.pi*integrate.quad(f, zmin, zmax)[0]/wave[i] + 2*np.pi*neff[i]*arcL/wave[i] + offset

    else:
        mag = np.zeros(n)
        phase = np.zeros(n)
        for i in range(n):
            #get mag
            if part == 'mag' or part == 'both':
                f = lambda z: ae[i]*np.exp(-ge[i]*g(z)[i]) + ao[i]*np.exp(-go[i]*g(z)[i])
                mag[i] = trig( np.pi*integrate.quad(f, zmin, zmax)[0]/wave[i] )

            #get phase
            if part == 'ph' or part == 'both':
                f = lambda z: ae[i]*np.exp(-ge[i]*g(z)[i]) - ao[i]*np.exp(-go[i]*g(z)[i]) + 2*neff[i]
                phase[i] = np.pi*integrate.quad(f, zmin, zmax)[0]/wave[i] + 2*np.pi*neff[i]*arcL/wave[i] + offset
    
    if part == 'mag':
        phase = 0
    if part == 'ph':
        mag = 1

    return mag*np.exp(-1j*phase)

"""
HELPER FUNCTIONS
"""
    
"""Plugs coeffs into actual closed form function"""
def get_closed_ans(ae, ao, ge, go, neff, wave, B, xe, xo, z, gap, term='k', part='mag'):
    #determine which parameter to get
    if term == 'k':
        trig = np.sin
        offset = np.pi/2
    elif term == 't':
        trig = np.cos
        offset = 0
    else:
        raise ValueError("Bad term parameter")
    
     #calculate magnitude
    if part == 'mag' or part == 'both':
        temp = ae*np.exp(-ge*gap)*B(xe)/ge + ao*np.exp(-go*gap)*B(xo)/go
        mag =  trig( temp*np.pi / wave )
    
    #calculate phase
    if part == 'ph' or part == 'both':
        temp = ae*np.exp(-ge*gap)*B(xe)/ge - ao*np.exp(-go*gap)*B(xo)/go + 2*z*neff
        phase = (temp*np.pi/wave + offset)
    
    if part == 'mag':
        phase = 0
    if part == 'ph':
        mag = 1

    return mag*np.exp(-1j*phase)
    
    
"""Returns all of the coefficients"""
def get_coeffs(wave, width, thickness, sw_angle):
    #get coeffs from LR model - needs numbers in nm
    inputs = np.column_stack((wave, width, thickness, sw_angle))
    coeffs = DC_coeffs.predict(inputs)
    ae = coeffs[:,0]
    ao = coeffs[:,1]
    ge = coeffs[:,2]
    go = coeffs[:,3]
    neff = coeffs[:,4]
    
    return (ae, ao, ge, go, neff)
    

"""Makes all inputs as the same shape to allow passing arrays through"""
def clean_inputs(inputs):
    inputs = list(inputs)
    #make all scalars into numpy arrays
    for i in range(len(inputs)):
        if np.isscalar(inputs[i]):
            inputs[i] = np.array([inputs[i]])
    
    #take largest size of numpy arrays
    n = max([len(i) for i in inputs])
    
    #if it's smaller than largest, make array full of same value
    for i in range(len(inputs)):
        if len(inputs[i]) != n:
            if len(inputs[i]) != 1:
                raise ValueError("Mismatched Input Array Size")
            inputs[i] = np.full((n), inputs[i][0])
            
    return tuple(inputs)


"""EVERYTHING BELOW THIS IS FOR TESTING"""
def any_gap_testing(wave, width, thickness, g, zmin, zmax, arc, sw_angle=90, term='k', part='mag'):
    #determine which parameter to get
    if term == 'k':
        trig = np.sin
        offset = np.pi/2
    elif term == 't':
        trig = np.cos
        offset = 0
    else:
        raise ValueError("Bad term parameter")
        
    #clean everything
    if np.ndim(g(0)) == 0:
        wave, width, thickness, sw_angle, arc = clean_inputs((wave, width, thickness, sw_angle, arc))
    else:
        wave, width, thickness, arc, _ = clean_inputs((wave, width, thickness, arc, g(0)))

        wave, thickness, sw_angle, ph = clean_inputs((wave, thickness, sw_angle, ph))
    n = len(wave)
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness, sw_angle)
    
    #if g has many lengths to sweep over
    if np.ndim(g(0)) == 0:
        mag = np.zeros(n)
        phase = np.zeros(n)
        for i in range(n):
            #get mag
            if part == 'mag' or part == 'both':
                f = lambda z: float(ae[i]*np.exp(-ge[i]*g(z)) + ao[i]*np.exp(-go[i]*g(z)))
                mag[i] = trig( np.pi*integrate.quad(f, zmin, zmax)[0]/wave[i] )

            #get phase
            if part == 'ph' or part == 'both':
                f = lambda z: float(ae[i]*np.exp(-ge[i]*g(z)) - ao[i]*np.exp(-go[i]*g(z)))
                phase[i] = np.pi*integrate.quad(f, zmin, zmax)[0]/wave[i] + 2*np.pi*arc[i]*neff[i]/wave[i] + offset

    else:
        mag = np.zeros(n)
        phase = np.zeros(n)
        for i in range(n):
            #get mag
            if part == 'mag' or part == 'both':
                f = lambda z: ae[i]*np.exp(-ge[i]*g(z)[i]) + ao[i]*np.exp(-go[i]*g(z)[i])
                mag[i] = trig( np.pi*integrate.quad(f, zmin, zmax)[0]/wave[i] )

            #get phase
            if part == 'ph' or part == 'both':
                f = lambda z: ae[i]*np.exp(-ge[i]*g(z)[i]) - ao[i]*np.exp(-go[i]*g(z)[i])
                phase[i] = np.pi*integrate.quad(f, zmin, zmax)[0]/wave[i] + 2*np.pi*arc[i]*neff[i]/wave[i] + offset
    
    if part == 'mag':
        phase = 0
    if part == 'ph':
        mag = 1

    return mag*np.exp(-1j*phase)
