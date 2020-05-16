# ------------------------------------------------------------------------- #
# Import libraries
# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #
# Input arguments
# ------------------------------------------------------------------------- #
import argparse
import sys

import h5py
import nlopt
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as sio
from scipy.signal import find_peaks

from SiPANN import SiP

parser = argparse.ArgumentParser()
parser.add_argument("iter", type=int)
parser.add_argument("time", type=int)
args = parser.parse_args()

currentIter = args.iter
time = args.time

# ------------------------------------------------------------------------- #
# Tuning parameters
# ------------------------------------------------------------------------- #

# Initial algorithm starting points
radius = 12
couplerLength = 4.5
gap = 0.2
width = 0.5
thickness = 0.22
angle = 85
widthCoupler = 0.5

# Peak thresholding
peakThreshold = 0.35

# Value bounds
radiusMin = 11.9
radiusMax = 12.1
couplerLengthMin = 4.48
couplerLengthMax = 4.52
widthMin = 0.45
widthMax = 0.55
thicknessMin = 0.2
thicknessMax = 0.24
gapMin = 0.05
gapMax = 0.3
angleMin = 80
angleMax = 90
couplerWidthMin = 0.4
couplerWidthMax = 0.6

algorithmGlobal = nlopt.G_MLSL_LDS
# algorithmGlobal  = nlopt.GN_DIRECT_L_RAND
# algorithmGlobal  = nlopt.GN_ISRES
# algorithmGlobal   = nlopt.GN_CRS2_LM
algorithmLocal = nlopt.LN_SBPLX

maxtime_global = time
maxtime_local = time

polyOrder = 1
polyOrderGap = 3

# ------------------------------------------------------------------------- #
# Load relevant info
# ------------------------------------------------------------------------- #

# Load the measurement data
f = h5py.File("waferData.mat", "r")
data = np.array(f.get("power"))  # Get a certain dataset
power = np.squeeze(data[:, currentIter])
wavelength = np.squeeze(np.array(f.get("wavelength")))
die = np.squeeze(np.array(f.get("dies")))
die = int(np.squeeze(die[currentIter]))
device = np.squeeze(np.array(f.get("devices")))
device = int(np.squeeze(device[currentIter]))


filenamePrefix = "sweep/iter{:d}_die{:d}_device{:d}".format(currentIter, die, device)

print(filenamePrefix)

# ------------------------------------------------------------------------- #
# Preprocessing
# ------------------------------------------------------------------------- #

# Pull the coupler and loss data
a, b, w = SiP.extractor(power, wavelength)
print(w)
print(a)

# Fit the coupler and loss data
ap = np.polyfit(w, a, polyOrder)
az = np.poly1d(ap)
bp = np.polyfit(w, b, polyOrder)
bz = np.poly1d(bp)

# Identify the peaks for the measurement data
peaks, _ = find_peaks(1 - power, height=peakThreshold)
wavelengthPeaks = wavelength[peaks]
numPeaks = peaks.size

# ------------------------------------------------------------------------- #
# Relevant functions
# ------------------------------------------------------------------------- #

# Define a cost function that can locate the correct FSR
def costFunction_FSR(x, grad):
    radius = x[0]
    couplerLength = x[1]
    width = x[2]
    thickness = x[3]
    widthCoupler = x[4]
    gap = x[5]
    angle = x[6]

    E, alpha, t, _ = SiP.racetrack_AP_RR_TF(
        wavelength,
        radius=radius,
        couplerLength=couplerLength,
        gap=gap,
        width=width,
        widthCoupler=widthCoupler,
        angle=angle,
        thickness=thickness,
        loss=ap,
    )

    throughPort = np.abs(np.squeeze(E)) ** 2

    error1 = np.mean(np.abs(throughPort - power) ** 2)

    bp_sim = np.polyfit(wavelength, t, polyOrderGap)
    bz_sim = np.poly1d(bp_sim)

    error2 = np.mean(np.abs(bz_sim(w) - b) ** 2)

    # Pull the peaks from the simulation
    peaksSim, _ = find_peaks(1 - throughPort, height=peakThreshold)

    # calculate the number of peaks
    wavelengthSim = wavelength[peaksSim]
    powerSim = throughPort[peaksSim]

    if wavelengthSim.size > numPeaks:
        wavelengthSim = wavelengthSim[0:numPeaks]
        powerSim = powerSim[0:numPeaks]
    elif wavelengthSim.size < numPeaks:
        wavelengthSim = np.append(
            wavelengthSim, np.zeros((numPeaks - wavelengthSim.size,))
        )
        powerSim = np.append(powerSim, np.zeros((numPeaks - powerSim.size,)))
    # [[0,numPeaks-1]]
    # Estimate the error
    error1 = np.sum(np.abs(wavelengthSim - wavelengthPeaks) ** 2)
    error = error1 * 1e8 + error2 * 1e6
    print([error1, error2, error])
    return error


def costFunction_loss(params, gapA, x, grad):
    radius = params[0]
    couplerLength = params[1]
    width = params[2]
    thickness = params[3]
    gap = params[4]

    # Evaluate the functin
    E, alpha, t, _ = SiP.racetrack_AP_RR_TF(
        wavelength,
        radius=radius,
        couplerLength=couplerLength,
        gap=gap,
        width=width,
        thickness=thickness,
        loss=x,
    )

    throughPort = np.abs(np.squeeze(E)) ** 2

    ap_sim = np.polyfit(wavelength, alpha, polyOrder)
    az_sim = np.poly1d(ap_sim)

    error = np.mean(np.abs(az_sim(wavelength) - az(wavelength)) ** 2)

    print(error)

    return error


def costFunction_coupling(params, x, grad):
    radius = params[0]
    couplerLength = params[1]
    gap = x[0]
    width = params[2]
    thickness = params[3]

    # Evaluate the functin
    E, alpha, t, _ = SiP.racetrack_AP_RR_TF(
        wavelength,
        radius=radius,
        couplerLength=couplerLength,
        gap=gap,
        width=width,
        thickness=thickness,
    )

    throughPort = np.abs(np.squeeze(E)) ** 2

    bp_sim = np.polyfit(wavelength, t, polyOrder)
    bz_sim = np.poly1d(bp_sim)

    error = np.mean(np.abs(bz_sim(w) - b) ** 2)
    print(error)

    return error


def costFunctionFinal(x, grad):
    radius = x[0]
    couplerLength = x[1]
    width = x[2]
    thickness = x[3]
    widthCoupler = x[4]
    gap = x[5]
    angle = x[6]
    loss = x[7:]

    E, alpha, t, _ = SiP.racetrack_AP_RR_TF(
        wavelength,
        radius=radius,
        couplerLength=couplerLength,
        gap=gap,
        width=width,
        thickness=thickness,
        loss=loss,
    )

    throughPort = np.abs(np.squeeze(E)) ** 2

    error = np.mean(np.abs(throughPort - power) ** 2)
    print(error)
    return error


def plotFinal(x):
    radius = x[0]
    couplerLength = x[1]
    width = x[2]
    thickness = x[3]
    widthCoupler = x[4]
    gap = x[5]
    angle = x[6]
    loss = x[7:]

    E, alpha, t, alpha_s = SiP.racetrack_AP_RR_TF(
        wavelength,
        radius=radius,
        couplerLength=couplerLength,
        gap=gap,
        width=width,
        widthCoupler=widthCoupler,
        angle=angle,
        thickness=thickness,
        loss=loss,
    )

    # get the transmission spectrum
    throughPort = np.abs(np.squeeze(E)) ** 2

    peaksSim, _ = find_peaks(1 - throughPort, height=peakThreshold)
    print(peaksSim)

    fig = plt.figure(figsize=(7, 7))
    plt.subplot(3, 3, 1)
    plt.plot(wavelength, power)
    plt.plot(wavelength, throughPort, "--")
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("Power (a.u.)")
    plt.grid(True)

    plt.subplot(3, 3, 4)
    plt.plot(wavelength, 10 * np.log10(power))
    plt.plot(wavelength, 10 * np.log10(throughPort), "--")
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("Power (dB)")
    plt.grid(True)

    plt.subplot(3, 3, 2)
    plt.plot(wavelength, power, "o")
    plt.plot(wavelength, throughPort)
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("Power (a.u.)")
    plt.grid(True)
    plt.xlim(wavelength[peaksSim[0]] - 0.5e-3, wavelength[peaksSim[0]] + 0.5e-3)

    plt.subplot(3, 3, 3)
    plt.plot(wavelength, power, "o")
    plt.plot(wavelength, throughPort)
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("Power (a.u.)")
    plt.grid(True)
    plt.xlim(wavelength[peaksSim[-1]] - 0.5e-3, wavelength[peaksSim[-1]] + 0.5e-3)

    plt.subplot(3, 3, 5)
    plt.plot(wavelength, 10 * np.log10(power), "o")
    plt.plot(wavelength, 10 * np.log10(throughPort))
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("Power (dB)")
    plt.grid(True)
    plt.xlim(wavelength[peaksSim[0]] - 0.5e-3, wavelength[peaksSim[0]] + 0.5e-3)

    plt.subplot(3, 3, 6)
    plt.plot(wavelength, 10 * np.log10(power), "o")
    plt.plot(wavelength, 10 * np.log10(throughPort))
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("Power (dB)")
    plt.grid(True)
    plt.xlim(wavelength[peaksSim[-1]] - 0.5e-3, wavelength[peaksSim[-1]] + 0.5e-3)

    plt.subplot(3, 2, 5)
    plt.plot(w, a, "o")
    plt.plot(wavelength, alpha)
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(w, b, "o")
    plt.plot(wavelength, t)
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("Coupling")
    plt.grid(True)

    fig.suptitle("Iter={:d}, Die={:d}, Device={:d}".format(currentIter, die, device))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results.png")
    plt.show()

    # Save the data as a numpy archive
    np.savez(
        "data.npz",
        radius=radius,
        couplerLength=couplerLength,
        gap=gap,
        width=width,
        thickness=thickness,
        loss=loss,
        E=E,
        alpha=alpha,
        t=t,
        alpha_r=alpha_s,
    )


# ------------------------------------------------------------------------- #
# Step 2: Simaltaenously solve for the right FSR and gap
# ------------------------------------------------------------------------- #

print("=================")
print("FSR")

lowerBounds = [
    radiusMin,
    couplerLengthMin,
    widthMin,
    thicknessMin,
    couplerWidthMin,
    gapMin,
    angleMin,
]
upperBounds = [
    radiusMax,
    couplerLengthMax,
    widthMax,
    thicknessMax,
    couplerWidthMax,
    gapMax,
    angleMax,
]
x0 = [radius, couplerLength, width, thickness, widthCoupler, gap, angle]

numParams = len(x0)

# Do a global optimization first
opt = nlopt.opt(algorithmGlobal, numParams)
opt.set_lower_bounds(lowerBounds)
opt.set_upper_bounds(upperBounds)
opt.set_min_objective(costFunction_FSR)
opt.set_maxtime(maxtime_global)

localopt = nlopt.opt(algorithmLocal, numParams)
localopt.set_lower_bounds(lowerBounds)
localopt.set_upper_bounds(upperBounds)
localopt.set_min_objective(costFunction_FSR)
localopt.set_maxtime(10)
tol = 1e-3
localopt.set_ftol_rel(tol)
opt.set_local_optimizer(localopt)

x0 = opt.optimize(x0)
print("=================")
print("Global")
print(x0)
print(costFunction_FSR(x0, 0))

# Then do a local optimization
opt = nlopt.opt(algorithmLocal, numParams)
opt.set_lower_bounds(lowerBounds)
opt.set_upper_bounds(upperBounds)
opt.set_min_objective(costFunction_FSR)
opt.set_maxtime(maxtime_local)
tol = 1e-3
opt.set_ftol_rel(tol)
x0 = opt.optimize(x0)
print("=================")
print("Local")
print(x0)
print(costFunction_FSR(x0, 0))


# ------------------------------------------------------------------------- #
# Step 4: Final optimization
# ------------------------------------------------------------------------- #

P0 = list(x0) + list(ap)

print(P0)

numParams = len(P0)
opt = nlopt.opt(algorithmLocal, numParams)
opt.set_min_objective(costFunctionFinal)
opt.set_maxtime(maxtime_local)
# P0 = opt.optimize(P0)
print("=================")
print("skipping the final optimization")
print("Local")
print(P0)
print(costFunctionFinal(P0, 0))

# ------------------------------------------------------------------------- #
# Step 5: Save
# ------------------------------------------------------------------------- #

plotFinal(P0)
