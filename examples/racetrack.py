import matplotlib.pyplot as plt
import numpy as np

from SiPANN import comp
from SiPANN import dc

# units in nanometers
r = 12e3
w = 500
t = 220
wavelength = np.linspace(1510, 1580, 1000)
gap = 200
length = 4.5e3

rr = comp.racetrack_sb_rr(w, t, r, gap, length, loss=[0.999])
E, alpha, t, phi = rr.predict(wavelength)

rr.gds(filename="test.gds")

plt.figure()
plt.plot(wavelength, np.abs(E) ** 2)
plt.show()
