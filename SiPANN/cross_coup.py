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

def rr(wave, width, thickness, radius, gap):
    #get coeffs from LR model
    inputs = np.array([[wave*.001, width*.001, thickness*.001]])
    coeffs = DC_coeffs.predict(inputs)
    ae = coeffs[0,0]
    ao = coeffs[0,1]
    ge = coeffs[0,2]
    go = coeffs[0,3]
    
    #calculate everything
    B = lambda x: np.pi*x*np.exp(-x)*(special.iv(1,x) + special.modstruve(-1,x))
    
    xe = ge*(radius + width/2)
    xo = go*(radius + width/2)

    temp = ae*np.exp(-ge*gap)*B(xe)/ge + ao*np.exp(-go*gap)*B(xo)/go
    return np.sin( temp*np.pi / wave )

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

#def any(wave, width, thickness, f)