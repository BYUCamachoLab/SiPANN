"""This file contains integration of SCEE into various other tools, allowing 
for cascading of complex structures"""

import numpy as np


_SPEED_OF_LIGHT = 299792458  # m/s


def _wl2freq(wl):
    """Convenience function for converting from wavelength to frequency.

    Parameters
    ----------
    wl : float
        The wavelength in SI units (m).

    Returns
    -------
    freq : float
        The frequency in SI units (Hz).
    """
    return _SPEED_OF_LIGHT / wl


def export_interconnect(sparams, wavelength, filename, clear=True):
    """Exports scattering parameters to a file readable by interconnect.

    Parameters
    -----------
    sparams : ndarray
        Numpy array of size (N, d, d) where N is the number of frequency points and d the number of ports
    wavelength : ndarray
        Numpy array of wavelengths (in nm, like the rest of SCEE) of size (N)
    filename : string
        Location to save file
    clear : bool, optional
        If True, empties the file first. Defaults to True.
    """
    # set things up
    _, d, _ = sparams.shape
    if clear:
        open(filename, "w").close()
    with open(filename, "ab") as file:
        # make frequencies
        freq = _wl2freq(wavelength * 1e-9)

        # iterate through sparams saving
        for in_ in range(d):
            for out in range(d):
                # put things together
                sp = sparams[:, in_, out]
                temp = np.vstack((freq, np.abs(sp), np.unwrap(np.angle(sp)))).T

                # Save header
                header = (
                    f'("port {out+1}", "TE", 1, "port {in_+1}", 1, "transmission")\n'
                )
                header += f"{temp.shape}"

                # save data
                np.savetxt(file, temp, header=header, comments="")
