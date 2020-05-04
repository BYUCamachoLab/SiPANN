SCEE and Interconnect
=====================

The SCEE module in SiPANN also has built in functionality to export any
of it’s models directly into a format readable by Lumerical Interconnect
via the ``export_interconnect()`` function. This gives the user multiple
options (Interconnect or Simphony) to cascade devices into complex
structures. To export to a Interconnect file is as simple as a function
call. First we declare all of our imports:

.. code:: ipython3

    import numpy as np
    from SiPANN import scee

Then make our device and calculate it’s scattering parameters (we
arbitrarily choose a half ring resonator here)

.. code:: ipython3

    r = 10000
    w = 500
    t = 220
    wavelength = np.linspace(1500, 1600)
    gap = 100
    
    hr = scee.HalfRing(w, t, r, gap)
    sparams = hr.sparams(wavelength)

And then export. Note ``export_interconnect`` takes in wavelengths in
nms, but the Lumerical file will have frequency in meters, as is
standard in Interconnect. To export:

.. code:: ipython3

    filename = "halfring_10microns_sparams.txt"
    scee.export_interconnect(sparams, wavelength, filename)

As a final parameter, ``export_interconnect`` also has a ``clear=True``
parameter that will empty the file being written to before writing. If
you’d like to append to an existing file, simply set ``clear=False``.

This is available in script form
`here <https://github.com/contagon/SiPANN/blob/master/examples/Tutorials/ExportToInterConnect.ipynb>`__

