"""
 SiP.py - A library of different silicon photonic device compact models
 leveraging artificial neural networks

Changes                                       (Author) (Date)
  Initilization .............................. (AMH) - 22-01-2019
  Documentation Changes........................(ERP) - 20-04-2020

Current devices:                              (Author)(Date last modified)
  Straight waveguide (TE/TM) ................. (AMH) - 22-01-2019
  Bent waveguide ............................. (AMH) - 22-01-2019
  Evanescent waveguide coupler ............... (AMH) - 22-01-2019
  Racetrack ring resonator ................... (AMH) - 22-01-2019
  Rectangular ring resonator ................. (AMH) - 22-01-2019

"""
# ---------------------------------------------------------------------------- #
# Import libraries
# ---------------------------------------------------------------------------- #
import numpy as np
import pkg_resources
import skrf as rf
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.signal import peak_widths

from SiPANN import import_nn

# ---------------------------------------------------------------------------- #
# Initialize ANNs
# ---------------------------------------------------------------------------- #

"""
We initialize all of the ANNs as global objects for speed. This is especially
useful for optimization routines and GUI's that need to make several ANN
evaluations quickly.
"""

gap_FILE = pkg_resources.resource_filename("SiPANN", "ANN/TIGHT_ANGLE_GAP")
ANN_gap = import_nn.ImportNN(gap_FILE)

straight_FILE = pkg_resources.resource_filename("SiPANN", "ANN/TIGHT_ANGLE_STRAIGHT")
ANN_straight = import_nn.ImportNN(straight_FILE)

bent_FILE = pkg_resources.resource_filename("SiPANN", "ANN/TIGHT_ANGLE_BENT_RAND")
ANN_bent = import_nn.ImportNN(bent_FILE)

"""
Let's initialize all of the linear regression functions.
"""

gap_FILE0 = pkg_resources.resource_filename("SiPANN", "LR/R_gap0.pkl")
gap_FILE1 = pkg_resources.resource_filename("SiPANN", "LR/R_gap1.pkl")
LR_gap = [import_nn.ImportLR(gap_FILE0), import_nn.ImportLR(gap_FILE1)]

straight_FILE = pkg_resources.resource_filename("SiPANN", "LR/R_straight.pkl")
LR_straight = import_nn.ImportLR(straight_FILE)

bent_FILE = pkg_resources.resource_filename("SiPANN", "LR/R_bent.pkl")
LR_bent = import_nn.ImportLR(bent_FILE)


# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

# Generalized N-dimensional products. Useful for "vectorizing" the ANN calculations
# with multiple inputs
def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


# ---------------------------------------------------------------------------- #
# Strip waveguide
# ---------------------------------------------------------------------------- #


def straightWaveguide(wavelength, width, thickness, sw_angle=90, derivative=None):
    """Calculates the first effective index value of the TE mode. Can also
    calculate derivatives with respect to any of the inputs. This is especially
    useful for calculating the group index, or running gradient based
    optimization routines.

    Each of the inputs can either be a one dimensional numpy array or a scalar. This
    is especially useful if you want to sweep over multiple parameters and include
    all of the possible permutations of the sweeps.

    The output is a multidimensional array. The size of each dimension corresponds with
    the size of each of the inputs, such that
    DIM1 = size(wavelength)
    DIM2 = size(width)
    DIM3 = size(thickness)
    DIM4 = size(sw_angle)

    So if I swept 100 wavelength points, 1 width, 10 possible thicknesses, and 2 sw_angles, then
    the dimension of each output effective index (or higher order derivative) would
    be: (100,1,10,2).

    Parameters
    ----------
    wavelength : float or ndarray (W1,)
        Wavelength points to evaluate
    width : float or ndarray (W2,)
        Width of the waveguides in microns
    thickness : float or ndarray (T,)
        Thickness of the waveguides in microns
    sw_angle : float or ndarray (A,)
        Sidewall angle from horizontal in degrees, ie 90 makes a square. Defaults to 90.
    derivative : int
        Order of the derivative to take. Defaults to None.

    Returns
    -------
    TE0 : ndarray
        First TE effective index with size (W1,W2,T,A,), or if derivative's are included (W1,W2,T,A,4,)
    """
    # Santize the input
    if type(wavelength) is np.ndarray:
        wavelength = np.squeeze(wavelength)
    else:
        wavelength = np.array([wavelength])
    width = np.squeeze(width) if type(width) is np.ndarray else np.array([width])
    if type(thickness) is np.ndarray:
        thickness = np.squeeze(thickness)
    else:
        thickness = np.array([thickness])
    if type(sw_angle) is np.ndarray:
        sw_angle = np.squeeze(sw_angle)
    else:
        sw_angle = np.array([sw_angle])

    # Run through neural network
    INPUT = cartesian_product([wavelength, width, thickness, sw_angle])

    if derivative is None:
        OUTPUT = LR_straight.predict(INPUT)
    else:
        numRows = INPUT.shape[0]
        OUTPUT = np.zeros((numRows, 4))
        # Loop through the derivative of all the outputs
        for k in range(4):
            OUTPUT[:, k] = np.squeeze(
                ANN_straight.differentiate(INPUT, d=(0, k, derivative))
            )

    # process the output
    if derivative is None:
        tensorSize = (wavelength.size, width.size, thickness.size, sw_angle.size)
    else:
        tensorSize = (wavelength.size, width.size, thickness.size, sw_angle.size, 4)
    return np.reshape(OUTPUT, tensorSize)


def straightWaveguide_S(wavelength, width, thickness, length, sw_angle=90):
    """Calculates the analytic scattering matrix of a simple straight waveguide
    with length L.

    Parameters
    -----------
    wavelength : ndarray (N,)
        Wavelength points to evaluate
    width : float
        Width of the waveguides in microns
    thickness : float
        Thickness of the waveguides in microns
    sw_angle : float
        Sidewall angle from horizontal in degrees, ie 90 makes a square. Defaults to 90.
    L : float
        Length of the waveguide in microns

    Returns
    -------
    S : ndarray (N,2,2)
        Scattering matrix for each wavelength
    """
    TE0 = straightWaveguide(wavelength, width, thickness, sw_angle)

    neff = np.squeeze(TE0)

    N = wavelength.shape[0]
    S = np.zeros((N, 2, 2), dtype="complex128")
    S[:, 0, 1] = np.exp(1j * 2 * np.pi * length * neff / wavelength)
    S[:, 1, 0] = np.exp(1j * 2 * np.pi * length * neff / wavelength)
    return S


# ---------------------------------------------------------------------------- #
# Bent waveguide
# ---------------------------------------------------------------------------- #


def bentWaveguide(wavelength, width, thickness, radius, sw_angle=90, derivative=None):
    """Calculates the first effective index value of the TE mode of a bent
    waveguide. Can also calculate derivatives with respect to any of the
    inputs. This is especially useful for calculating the group index, or
    running gradient based optimization routines.

    Each of the inputs can either be a one dimensional numpy array or a scalar. This
    is especially useful if you want to sweep over multiple parameters and include
    all of the possible permutations of the sweeps.

    The output is a multidimensional array. The size of each dimension corresponds with
    the size of each of the inputs, such that
    DIM1 = size(wavelength)
    DIM2 = size(width)
    DIM3 = size(thickness)
    DIM4 = size(radius)
    DIM5 = size(sw_angle)

    So if I swept 100 wavelength points, 1 width, 10 possible thicknesses, 5 radii, and 2 sw_angles, then
    the dimension of each output effective index (or higher order derivative) would
    be: (100,1,10,5,2).

    Parameters
    ----------
    wavelength : float or ndarray (W1,)
        Wavelength points to evaluate
    width : float or ndarray (W2,)
        Width of the waveguides in microns
    thickness : float or ndarray (T,)
        Thickness of the waveguides in microns
    radius : float or ndarray (R,)
        Radius of waveguide in microns.
    sw_angle : float or ndarray (A,)
        Sidewall angle from horizontal in degrees, ie 90 makes a square. Defaults to 90.
    derivative : int
        Order of the derivative to take. Defaults to None.

    Returns
    -------
    TE0 : ndarray
        First TE effective index with size (W1,W2,T,R,A,), or if derivative's are included (W1,W2,T,R,A,5,)
    """
    # Santize the input
    if type(wavelength) is np.ndarray:
        wavelength = np.squeeze(wavelength)
    else:
        wavelength = np.array([wavelength])
    width = np.squeeze(width) if type(width) is np.ndarray else np.array([width])
    if type(thickness) is np.ndarray:
        thickness = np.squeeze(thickness)
    else:
        thickness = np.array([thickness])
    if type(radius) is np.ndarray:
        radius = np.squeeze(radius)
    else:
        radius = np.array([radius])
    if type(sw_angle) is np.ndarray:
        sw_angle = np.squeeze(sw_angle)
    else:
        sw_angle = np.array([sw_angle])

    # Run through neural network
    INPUT = cartesian_product([wavelength, width, thickness, radius, sw_angle])

    if derivative is None:
        OUTPUT = LR_bent.predict(INPUT)
    else:
        numRows = INPUT.shape[0]
        OUTPUT = np.zeros((numRows, 2))
        # Loop through the derivative of all the outputs
        for k in range(2):
            OUTPUT[:, k] = np.squeeze(
                ANN_bent.differentiate(INPUT, d=(0, k, derivative))
            )

    # process the output
    if derivative is None:
        tensorSize = (
            wavelength.size,
            width.size,
            thickness.size,
            radius.size,
            sw_angle.size,
        )
    else:
        tensorSize = (
            wavelength.size,
            width.size,
            thickness.size,
            radius.size,
            sw_angle.size,
            5,
        )
    return np.reshape(OUTPUT, tensorSize)


def bentWaveguide_S(wavelength, width, thickness, radius, angle, sw_angle=90):
    """Calculates the analytic scattering matrix of bent waveguide with
    specific radius and circle length.

    Parameters
    -----------
    wavelength : ndarray (N,)
        Wavelength points to evaluate
    width : float
        Width of the waveguides in microns
    thickness : float
        Thickness of the waveguides in microns
    radius : float
        Radius of waveguide in microns.
    angle : float
        Number of radians of circle that bent waveguide transverses
    sw_angle : float
        Sidewall angle from horizontal in degrees, ie 90 makes a square. Defaults to 90.

    Returns
    -------
    S : ndarray (N,2,2)
        Scattering matrix for each wavelength
    """
    # Pull effective indices from ANN
    TE0 = bentWaveguide(wavelength, width, thickness, radius, sw_angle)
    neff = np.squeeze(TE0)

    N = wavelength.shape[0]
    S = np.zeros((N, 2, 2), dtype="complex128")
    S[:, 0, 1] = np.exp(1j * 2 * np.pi * radius * neff * angle / wavelength)
    S[:, 1, 0] = np.exp(1j * 2 * np.pi * radius * neff * angle / wavelength)
    return S


# ---------------------------------------------------------------------------- #
# Evanescent waveguide coupler
# ---------------------------------------------------------------------------- #


def evWGcoupler(wavelength, width, thickness, gap, sw_angle=90, derivative=None):
    """Calculates the even and odd effective indice values of the TE mode of
    parallel waveguides. Can also calculate derivatives with respect to any of
    the inputs. This is especially useful for calculating the group index, or
    running gradient based optimization routines.

    Each of the inputs can either be a one dimensional numpy array or a scalar. This
    is especially useful if you want to sweep over multiple parameters and include
    all of the possible permutations of the sweeps.

    The output is a multidimensional array. The size of each dimension corresponds with
    the size of each of the inputs, such that
    DIM1 = size(wavelength)
    DIM2 = size(width)
    DIM3 = size(thickness)
    DIM4 = size(gap)
    DIM5 = size(sw_angle)

    So if I swept 100 wavelength points, 1 width, 10 possible thicknesses, 5 radii, and 2 sw_angles, then
    the dimension of each output effective index (or higher order derivative) would
    be: (100,1,10,5,2).

    Parameters
    ----------
    wavelength : float or ndarray (W1,)
        Wavelength points to evaluate
    width : float or ndarray (W2,)
        Width of the waveguides in microns
    thickness : float or ndarray (T,)
        Thickness of the waveguides in microns
    gap : float or ndarray (G,)
        Gap distance between waveguides
    sw_angle : float or ndarray (A,)
        Sidewall angle from horizontal in degrees, ie 90 makes a square. Defaults to 90.
    derivative : int
        Order of the derivative to take. Defaults to None.

    Returns
    -------
    TE0 : ndarray
        First TE effective index with size (W1,W2,T,G,A,), or if derivative's are included (W1,W2,T,G,A,5,)
    """
    # Santize the input
    if type(wavelength) is np.ndarray:
        wavelength = np.squeeze(wavelength)
    else:
        wavelength = np.array([wavelength])
    width = np.squeeze(width) if type(width) is np.ndarray else np.array([width])
    if type(thickness) is np.ndarray:
        thickness = np.squeeze(thickness)
    else:
        thickness = np.array([thickness])
    gap = np.squeeze(gap) if type(gap) is np.ndarray else np.array([gap])
    if type(sw_angle) is np.ndarray:
        sw_angle = np.squeeze(sw_angle)
    else:
        sw_angle = np.array([sw_angle])

    # Run through neural network
    INPUT = cartesian_product([wavelength, width, thickness, gap, sw_angle])

    if derivative is None:
        OUTPUT0 = LR_gap[0].predict(INPUT)
        OUTPUT1 = LR_gap[1].predict(INPUT)
    else:
        numRows = INPUT.shape[0]
        OUTPUT = np.zeros((numRows, 4))
        # Loop through the derivative of all the outputs
        for k in range(4):
            OUTPUT[:, k] = np.squeeze(
                ANN_gap.differentiate(INPUT, d=(0, k, derivative))
            )

    # process the output
    if derivative is None:
        tensorSize = (
            wavelength.size,
            width.size,
            thickness.size,
            gap.size,
            sw_angle.size,
        )
    else:
        tensorSize = (
            wavelength.size,
            width.size,
            thickness.size,
            gap.size,
            sw_angle.size,
            5,
        )
    TE0 = np.reshape(OUTPUT0, tensorSize)
    TE1 = np.reshape(OUTPUT1, tensorSize)
    return TE0, TE1


def evWGcoupler_S(wavelength, width, thickness, gap, couplerLength, sw_angle=90):
    """Calculates the analytic scattering matrix of a simple, parallel
    waveguide directional coupler using the ANN.

    Parameters
    -----------
    wavelength : ndarray (N,)
        Wavelength points to evaluate
    width : float
        Width of the waveguides in microns
    thickness : float
        Thickness of the waveguides in microns
    gap : float
        gap in the coupler region in microns
    couplerLength : float
        Length of the coupling region in microns

    Returns
    -------
    S : ndarray (N,4,4)
        Scattering matrix
    """
    N = wavelength.shape[0]

    # Get the fundamental mode of the waveguide itself
    # TE0 = straightWaveguide(wavelength,width,thickness)
    # n0 = np.squeeze(TE0)

    # Get the modes of the coupler structure
    cTE0, cTE1 = evWGcoupler(wavelength, width, thickness, gap, sw_angle)
    n1 = np.squeeze(cTE0)  # Get the first mode of the coupler region
    n2 = np.squeeze(cTE1)  # Get the second mode of the coupler region
    # dn = n1 - n2  # Find the modal differences
    # -------- Formulate the S matrix ------------ #
    # x =  np.exp(-1j*2*np.pi*n0*couplerLength/wavelength) * np.cos(np.pi*dn/wavelength*couplerLength)
    # y =  1j * np.exp(-1j*2*np.pi*n0*couplerLength/wavelength) * np.sin(np.pi*dn/wavelength*couplerLength)

    Beta1 = 2 * np.pi * n1 / wavelength
    Beta2 = 2 * np.pi * n2 / wavelength
    x = (
        1
        / np.sqrt(4)
        * (np.exp(1j * Beta1 * couplerLength) + np.exp(1j * Beta2 * couplerLength))
    )
    y = (
        1
        / np.sqrt(4)
        * (
            np.exp(1j * Beta1 * couplerLength)
            + np.exp(1j * Beta2 * couplerLength - 1j * np.pi)
        )
    )

    S = np.zeros((N, 4, 4), dtype="complex128")

    # Row 1
    S[:, 0, 1] = x
    S[:, 0, 3] = y
    # Row 2
    S[:, 1, 0] = x
    S[:, 1, 2] = y
    # Row 3
    S[:, 2, 1] = y
    S[:, 2, 3] = x
    # Row 4
    S[:, 3, 0] = y
    S[:, 3, 2] = x
    return S


# ---------------------------------------------------------------------------- #
# Racetrack Ring Resonator
# ---------------------------------------------------------------------------- #


def racetrack_AP_RR(
    wavelength, radius=5, couplerLength=5, gap=0.2, width=0.5, thickness=0.2
):
    """This particular transfer function assumes that the coupling sides of the
    ring resonator are straight, and the other two sides are curved. Therefore,
    the roundtrip length of the RR is 2*pi*radius + 2*couplerLength.

    We assume that the round parts of the ring have negligble coupling compared to
    the straight sections.

    Parameters
    -----------
    wavelength : ndarray (N,)
        Wavelength points to evaluate
    radius : float
        Radius of the sides in microns
    couplerLength : float
        Length of the coupling region in microns
    gap : float
        Gap in the coupler region in microns
    width : float
        Width of the waveguides in microns
    thickness : float
        Thickness of the waveguides in microns

    Returns
    -------
    S : ndarray (N,4,4)
        Scattering matrix
    """
    # Sanitize the input
    wavelength = np.squeeze(wavelength)
    # N = wavelength.shape[0]

    # Calculate coupling scattering matrix
    couplerS = evWGcoupler_S(wavelength, width, thickness, gap, couplerLength)

    # Calculate bent scattering matrix
    bentS = bentWaveguide_S(wavelength, width, thickness, radius, np.pi)

    # Calculate straight scattering matrix
    straightS = straightWaveguide_S(wavelength, width, thickness, couplerLength)

    # Cascade all the waveguide sections
    Sw = rf.connect_s(bentS, 1, straightS, 0)
    Sw = rf.connect_s(Sw, 1, bentS, 0)

    # Connect coupler to waveguide section
    S = rf.connect_s(couplerS, 2, Sw, 0)

    # Close the ring
    S = rf.innerconnect_s(S, 2, 3)

    ## Cascade final coupler
    # S = rf.connect_s(S, 2, couplerS, 2)
    # S = rf.innerconnect_s(S, 2,5)

    # Output final s matrix
    return S


def racetrack_AP_RR_TF(
    wavelength,
    sw_angle=90,
    radius=12,
    couplerLength=4.5,
    gap=0.2,
    width=0.5,
    thickness=0.2,
    widthCoupler=0.5,
    loss=[0.99],
    coupling=[0],
):
    """This particular transfer function assumes that the coupling sides of the
    ring resonator are straight, and the other two sides are curved. Therefore,
    the roundtrip length of the RR is 2*pi*radius + 2*couplerLength. This model
    also includes loss. (??? Need Verification on last line)

    We assume that the round parts of the ring have negligble coupling compared to
    the straight sections.

    Parameters
    -----------
    wavelength : ndarray (N,)
        Wavelength points to evaluate
    radius : float
        Radius of the sides in microns
    couplerLength : float
        Length of the coupling region in microns
    gap : float
        Gap in the coupler region in microns
    width : float
        Width of the waveguides in microns
    thickness : float
        Thickness of the waveguides in microns

    Returns
    -------
    E : ndarray
        Complex array of size (N,)
    alpha : ndarray
        Array of size (N,)
    t : ndarray
        Array of size (N,)
    alpha_s : ndarray
        Array of size (N,)
    phi : ndarray
        Array of size (N,)
    """
    # Sanitize the input
    wavelength = np.squeeze(wavelength)
    # N = wavelength.shape[0]

    # calculate coupling
    cTE0, cTE1 = evWGcoupler(
        wavelength=wavelength,
        width=widthCoupler,
        thickness=thickness,
        sw_angle=sw_angle,
        gap=gap,
    )
    n1 = np.squeeze(cTE0)  # Get the first mode of the coupler region
    n2 = np.squeeze(cTE1)  # Get the second mode of the coupler region
    Beta1 = 2 * np.pi * n1 / wavelength
    Beta2 = 2 * np.pi * n2 / wavelength
    x = 0.5 * (np.exp(1j * Beta1 * couplerLength) + np.exp(1j * Beta2 * couplerLength))
    y = 0.5 * (
        np.exp(1j * Beta1 * couplerLength)
        + np.exp(1j * Beta2 * couplerLength - 1j * np.pi)
    )

    alpha_c = np.sqrt(np.abs(x) ** 2 + np.abs(y) ** 2)

    t_c = x
    # k_c = y

    # Construct the coupling polynomial
    # couplingPoly = np.poly1d(coupling)

    # r = np.abs(x) - couplingPoly(wavelength)
    # k = np.abs(y)

    # calculate bent waveguide
    TE0_B = np.squeeze(
        bentWaveguide(
            wavelength=wavelength,
            width=width,
            thickness=thickness,
            sw_angle=sw_angle,
            radius=radius,
        )
    )

    # calculate straight waveguide
    TE0 = np.squeeze(
        straightWaveguide(
            wavelength=wavelength, width=width, thickness=thickness, sw_angle=sw_angle
        )
    )

    # Calculate round trip length
    # L = 2 * np.pi * radius + 2 * couplerLength

    # calculate total loss
    # alpha = np.squeeze(np.exp(- np.imag(TE0) * 2*couplerLength - np.imag(TE0_B)*2*np.pi*radius - lossPoly(wavelength)*L))
    alpha_t = np.exp(
        -np.imag(TE0) * 2 * couplerLength - np.imag(TE0_B) * 2 * np.pi * radius
    )
    alpha_m = np.squeeze(alpha_c * alpha_t)
    offset = np.mean(alpha_m)
    lossTemp = loss.copy()
    lossTemp[-1] = loss[-1] - (1 - offset)
    lossPoly = np.poly1d(loss)
    alpha = lossPoly(wavelength)
    alpha_s = alpha - alpha_m

    # calculate phase shifts
    phi_c = np.unwrap(np.angle(t_c))
    BetaStraight = np.unwrap(2 * np.pi * np.real(TE0) / wavelength)
    BetaBent = np.unwrap(2 * np.pi * np.real(TE0_B) / wavelength)
    phi_r = np.squeeze(BetaStraight * couplerLength + BetaBent * 2 * np.pi * radius)
    phi = np.unwrap(phi_r + phi_c)

    t = np.abs(t_c) / alpha_c

    ## Cascade final coupler
    # E = np.exp(1j*(np.pi+phi)) * (alpha - r*np.exp(-1j*phi))/(1-r*alpha*np.exp(1j*phi))
    E = (
        (t - alpha * np.exp(1j * phi))
        / (1 - alpha * t * np.exp(1j * phi))
        * (t_c / np.conj(t_c))
        * alpha_c
        * np.exp(-1j * phi_c)
    )

    # Output final s matrix
    return E, alpha, t, alpha_s, phi


# ---------------------------------------------------------------------------- #
# Rectangular Ring Resonator
# ---------------------------------------------------------------------------- #


def rectangularRR(
    wavelength,
    radius=5,
    couplerLength=5,
    sideLength=5,
    gap=0.2,
    width=0.5,
    thickness=0.2,
):
    """This particular transfer function assumes that all four sides of the
    ring resonator are straight and that the corners are rounded. Therefore,
    the roundtrip length of the RR is 2*pi*radius + 2*couplerLength +
    2*sideLength.

    We assume that the round parts of the ring have negligble coupling compared to
    the straight sections.

    Parameters
    -----------
    wavelength : ndarray (N,)
        Wavelength points to evaluate
    radius : float
        Radius of the sides in microns
    couplerLength : float
        Length of the coupling region in microns
    sideLength : float
        Length of each side not coupling in microns
    gap : float
        Gap in the coupler region in microns
    width : float
        Width of the waveguides in microns
    thickness : float
        Thickness of the waveguides in microns

    Returns
    -------
    S : ndarray (N,4,4)
        Scattering matrix
    """
    # Sanitize the input

    # Calculate transfer function output

    #
    raise NotImplementedError("Hasn't been implemented yet")


# ---------------------------------------------------------------------------- #
# Loss and coupling extractor
# ---------------------------------------------------------------------------- #


def extractor(power, wavelength):
    peakThreshold = 0.3
    distanceThreshold = 4e-3 / (wavelength[1] - wavelength[0])

    peaks, _ = find_peaks(1 - power, height=peakThreshold, distance=distanceThreshold)
    results_half = peak_widths(1 - power, peaks, rel_height=0.5)

    FWHM = results_half[0][:-1] * (wavelength[1] - wavelength[0])
    FSR = np.diff(wavelength[peaks])

    F = FSR / FWHM
    E = 1 / power[peaks[:-1]]

    A = np.cos(np.pi / F) / (1 + np.sin(np.pi / F))
    B = 1 - 1 / E * (1 - np.cos(np.pi / F)) / (1 + np.cos(np.pi / F))

    a = np.sqrt(A / B) + np.sqrt(A / B - A)
    b = np.sqrt(A / B) - np.sqrt(A / B - A)

    w = wavelength[peaks[:-1]]
    return a, b, w
