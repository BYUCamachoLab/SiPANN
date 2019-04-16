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
def straight(wave, width, thickness, length):
    #clean everything
    wave, width, length = clean_inputs((wave, width, length))
    #get coefficients
    ae, ao, ge, go = get_coeffs(wave, width, thickness)
    
    #calculate everything
    B = lambda x: x
    xe = ge*length
    xo = go*length

    temp = ae*np.exp(-ge*gap)*B(xe)/ge + ao*np.exp(-go*gap)*B(xo)/go
    return np.sin( temp*np.pi / wave )

def curved(wave, width, thickness, length, gap, H, V):
    #clean everything
    wave, width, length, gap, H, V = clean_inputs((wave, width, length, gap, H, V))
    #get coefficients
    ae, ao, ge, go = get_coeffs(wave, width, thickness)
    
    #calculate everything
    B = lambda x: x*(1 + 2*H*np.exp(-V*x/L)*special.iv(0,V*x/L)/L)
    xe = ge*length
    xo = go*length

    temp = ae*np.exp(-ge*gap)*B(xe)/ge + ao*np.exp(-go*gap)*B(xo)/go
    return np.sin( temp*np.pi / wave )

def racetrack(wave, width, thickness, radius, gap, length):
    #clean everything
    wave, width, radius, gap, length = clean_inputs((wave, width, radius, gap, length))
    #get coefficients
    ae, ao, ge, go = get_coeffs(wave, width, thickness)
    
    #calculate everything
    B = lambda x: length*x/(R+width/2) + np.pi*x*np.exp(-x)*(special.iv(1,x) + special.modstruve(-1,x))
    xe = ge*(radius + width/2)
    xo = go*(radius + width/2)

    temp = ae*np.exp(-ge*gap)*B(xe)/ge + ao*np.exp(-go*gap)*B(xo)/go
    return np.sin( temp*np.pi / wave )

def rr(wave, width, thickness, radius, gap):
    #clean everything
    wave, width, thickness, radius, gap = clean_inputs((wave, width, thickness, radius, gap))
    #get coefficients
    ae, ao, ge, go = get_coeffs(wave, width, thickness)
    
    #calculate everything
    B = lambda x: np.pi*x*np.exp(-x)*(special.iv(1,x) + special.modstruve(-1,x))
    xe = ge*(radius + width/2)
    xo = go*(radius + width/2)

    temp = ae*np.exp(-ge*gap)*B(xe)/ge + ao*np.exp(-go*gap)*B(xo)/go
    return np.sin( temp*np.pi / wave )

def pushed_rr(wave, width, thickness, radius, d, theta):
    #clean everything
    wave, width, radius, d, theta = clean_inputs((wave, width, radius, d, theta))
    #get coefficients
    ae, ao, ge, go = get_coeffs(wave, width, thickness)
    
    #calculate everything
    B = lambda x: x
    xe = ge*theta*(radius + width/2 + d/2)
    xo = go*theta*(radius + width/2 + d/2)

    temp = ae*np.exp(-ge*gap)*B(xe)/ge + ao*np.exp(-go*gap)*B(xo)/go
    return np.sin( temp*np.pi / wave )

"""
The most important one, it takes in a function of gap size and a range to sweep over
"""
def any_gap(wave, width, thickness, g, zmin, zmax):
    #clean everything
    wave, width, thickness, _ = clean_inputs((wave, width, thickness, g(0)))
    n = len(wave)
    #get coefficients
    ae, ao, ge, go = get_coeffs(wave, width, thickness)
    
    #if g has many lengths to sweep over
    if np.isscalar(g(0)):
        ans = np.zeros(n)
        for i in range(n):
            f = lambda z: ae[i]*np.exp(-ge[i]*g(z)) + ao[i]*np.exp(-go[i]*g(z))
            ans[i] = np.sin( np.pi*integrate.quad(f, zmin, zmax)[0]/wave[i] )
        return ans
    else:
        ans = np.zeros(n)
        for i in range(n):
            f = lambda z: ae[i]*np.exp(-ge[i]*g(z)[i]) + ao[i]*np.exp(-go[i]*g(z)[i])
            ans[i] = np.sin( np.pi*integrate.quad(f, zmin, zmax)[0]/wave[i] )
        return ans

"""Returns all of the coefficients"""
def get_coeffs(wave, width, thickness):
    #get coeffs from LR model - needs numbers in nm
    inputs = np.column_stack((wave, width, thickness))
    coeffs = DC_coeffs.predict(inputs)
    ae = coeffs[:,0]
    ao = coeffs[:,1]
    ge = coeffs[:,2]
    go = coeffs[:,3]
    
    return (ae, ao, ge, go)
    

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