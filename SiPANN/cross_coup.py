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


"""
All the closed form solutions for Different Structures
"""
#part terms can be mag, ph, or both
def straight(wave, width, thickness, length, gap, term='k', part='mag'):
    #clean everything
    wave, width, thickness, length, gap = clean_inputs((wave, width, thickness, length, gap))
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness)
    
    #set up closed form solutions
    B = lambda x: x
    xe = ge*length
    xo = go*length
    z = length

    #get closed form solution
    return get_closed_ans(ae, ao, ge, go, neff, wave, B, xe, xo, z, gap, term, part)
  
    

def curved(wave, width, thickness, length, gap, H, V, term='k', part='mag'):
    #clean everything
    wave, width, thickness, length, gap, H, V = clean_inputs((wave, width, thickness, length, gap, H, V))
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness)
    
    #calculate everything
    B = lambda x: x*(1 + 2*H*np.exp(-V*x/L)*special.iv(0,V*x/L)/L)
    xe = ge*length
    xo = go*length
    z = 2*H + L

    #get closed form solution
    return get_closed_ans(ae, ao, ge, go, neff, wave, B, xe, xo, z, gap, term, part)



def racetrack(wave, width, thickness, radius, gap, length, term='k', part='mag'):
    #clean everything
    wave, width, thickness, radius, gap, length = clean_inputs((wave, width, thickness, radius, gap, length))
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness)
    
    #calculate everything
    B = lambda x: length*x/(radius+width/2) + np.pi*x*np.exp(-x)*(special.iv(1,x) + special.modstruve(-1,x))
    xe = ge*(radius + width/2)
    xo = go*(radius + width/2)
    z = 2*(radius + width) + length

    #get closed form solution
    return get_closed_ans(ae, ao, ge, go, neff, wave, B, xe, xo, z, gap, term, part)



def rr(wave, width, thickness, radius, gap, term='k', part='mag'):
    #clean everything
    wave, width, thickness, radius, gap = clean_inputs((wave, width, thickness, radius, gap))
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness)
    
    #calculate everything
    B = lambda x: np.pi*x*np.exp(-x)*(special.iv(1,x) + special.modstruve(-1,x))
    xe = ge*(radius + width/2)
    xo = go*(radius + width/2)
    z = 2*(radius + width/2)
    #z += 2*np.pi*(radius + width/2)/4

    #get closed form solution
    return get_closed_ans(ae, ao, ge, go, neff, wave, B, xe, xo, z, gap, term, part)



def double_rr(wave, width, thickness, radius, gap, term='k', part='mag'):
    #clean everything
    wave, width, thickness, radius, gap = clean_inputs((wave, width, thickness, radius, gap))
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness)
    
    #calculate everything
    B = lambda x: (np.pi*2*x*np.exp(-2*x)*(special.iv(1,2*x) + special.modstruve(-1,2*x)))/2
    xe = ge*(radius + width/2)
    xo = go*(radius + width/2)
    z = 2*(radius + width)

    #get closed form solution
    return get_closed_ans(ae, ao, ge, go, neff, wave, B, xe, xo, z, gap, term, part)



def pushed_rr(wave, width, thickness, radius, d, theta, term='k', part='mag'):
    #clean everything
    wave, width, thickness, radius, d, theta = clean_inputs((wave, width, thickness, radius, d, theta))
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness)
    
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
def any_gap(wave, width, thickness, g, zmin, zmax, term='k', part='mag'):
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
        wave, width, thickness = clean_inputs((wave, width, thickness))
    else:
        wave, width, thickness, _ = clean_inputs((wave, width, thickness, g(0)))
    n = len(wave)
    #get coefficients
    ae, ao, ge, go, neff = get_coeffs(wave, width, thickness)
    
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
                f = lambda z: float(ae[i]*np.exp(-ge[i]*g(z)) - ao[i]*np.exp(-go[i]*g(z)) + 2*neff[i])
                phase[i] = np.pi*integrate.quad(f, zmin, zmax)[0]/wave[i] + offset

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
                phase[i] = np.pi*integrate.quad(f, zmin, zmax)[0]/wave[i] + offset
    
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
def get_coeffs(wave, width, thickness):
    #get coeffs from LR model - needs numbers in nm
    inputs = np.column_stack((wave, width, thickness))
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


"""Everything below here is temporary and used for testing"""
def rr_450_220(wave, width, thickness, radius, gap):
    #get coeffs from paper
    ae = 0.177967
    ao = 0.049910
    ge = 0.011898
    go = 0.006601
    
    #calculate everything
    B = lambda x: np.pi*x*np.exp(-x)*(special.iv(1,x) + special.modstruve(-1,x))
    
    xe = ge*(radius + width/2)
    xo = go*(radius + width/2)

    temp = ae*np.exp(-ge*gap)*B(xe)/ge + ao*np.exp(-go*gap)*B(xo)/go
    return np.sin( temp*np.pi / wave )
