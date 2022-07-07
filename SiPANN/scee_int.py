import numpy as np
from simphony import Model
from simphony.layout import Circuit
from simphony.pins import Pin, PinList
from simphony.tools import freq2wl, wl2freq

########################################################################
####### This file contains integration of SCEE into various other ######
#######    tools, allowing for cascading of complex structures    ######
########################################################################


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
        freq = wl2freq(wavelength * 1e-9)

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


class SimphonyWrapper(Model):
    """Class that wraps SCEE models for use in simphony.

    Model passed into class CANNOT have varying geometries, as a device such as this
    can't be cascaded properly.

    Parameters
    -----------
    model : DC
        Chosen compact model from ``SiPANN.scee`` module. Can be any model that inherits from
        the DC abstract class
    sigmas : dict, optional
        Dictionary mapping parameters to sigma values for use in monte_carlo simulations. Note sigmas should
        be in values of nm. Defaults to an empty dictionary.
    """

    def __init__(self, model, sigmas=dict()):
        self.model = model
        self.sigmas = sigmas
        self.pins = PinList(
            [
                Pin(self.model, "n1"),
                Pin(self.model, "n2"),
                Pin(self.model, "n3"),
                Pin(self.model, "n4"),
            ]
        )  #: The default pin names of the device
        self.model.pins = self.pins
        self.model.circuit = Circuit(self.model)

        self.model.freq_range = (
            182800279268292.0,
            205337300000000.0,
        )  #: The valid frequency range for this model.

        self.model.s_parameters = self.s_parameters
        self.model.monte_carlo_s_parameters = self.monte_carlo_s_parameters
        self.model.regenerate_monte_carlo_parameters = (
            self.regenerate_monte_carlo_parameters
        )

        # save actual parameters for switching back from monte_carlo
        self.og_params = self.model.__dict__.copy()
        self.rand_params = {}

        # make sure there's no varying geometries
        args = self.model._clean_args(None)
        if len(args[0]) != 1:
            raise ValueError(
                "You have changing geometries, use in simphony doesn't make sense!"
            )

        self.regenerate_monte_carlo_parameters()

    def s_parameters(self, freq):
        """Get the s-parameters of SCEE Model.

        Parameters
        ----------
        freq : np.ndarray
            A frequency array to calculate s-parameters over (in Hz).

        Returns
        -------
        s : np.ndarray
            Returns the calculated s-parameter matrix.
        """
        # convert wavelength to frequency
        wl = freq2wl(freq) * 1e9

        return self.model.sparams(wl)

    def monte_carlo_s_parameters(self, freq):
        """Get the s-parameters of SCEE Model with slightly changed parameters.

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

        # perturb params and get sparams
        self.model.update(**self.rand_params)
        sparams = self.model.sparams(wl)

        # restore parameters to originals
        self.model.update(**self.og_params)

        return sparams

    def regenerate_monte_carlo_parameters(self):
        """Varies parameters based on passed in sigma dictionary.

        Iterates through sigma dictionary to change each of those
        parameters, with the mean being the original values found in
        model.
        """
        # iterate through all params that should be tweaked
        for param, sigma in self.sigmas.items():
            self.rand_params[param] = np.random.normal(self.og_params[param], sigma)
