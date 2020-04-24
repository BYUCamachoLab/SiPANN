#############################
SCEE Module
#############################

These methods are based on the model SCEE found in [CITE PAPER WHEN PUBLISHED].

*********************
Helper Functions
*********************

Base Directional Coupler Class
==============================
.. autoclass:: SiPANN.scee.DC
    :members:

Waveguide
==========================
.. autoclass:: SiPANN.scee.Waveguide
    :members:
    :inherited-members:

Effective Index Finder
==============================
.. autofunction:: SiPANN.scee.get_neff

Integrations Coefficient Finder
=================================
.. autofunction:: SiPANN.scee.get_coeffs

Input Cleaner
==============================
.. autofunction:: SiPANN.scee.clean_inputs


*********************
Coupling Devices
*********************

Symmetric Coupler
======================
.. autoclass:: SiPANN.scee.GapFuncSymmetric
    :members:
    :inherited-members:

Non-Symmetric Coupler
==========================
.. autoclass:: SiPANN.scee.GapFuncAntiSymmetric
    :members:
    :inherited-members:

Coupling Straight Waveguides
==============================
.. autoclass:: SiPANN.scee.StraightCoupler
    :members:
    :inherited-members:

Standard Directional Coupler
====================================
.. autoclass:: SiPANN.scee.Standard
    :members:
    :inherited-members:

Half Ring
==========================
.. autoclass:: SiPANN.scee.HalfRing
    :members:
    :inherited-members:

Half Racetrack Resonator
==========================
.. autoclass:: SiPANN.scee.HalfRacetrack
    :members:
    :inherited-members:

Double Half Ring
==========================
.. autoclass:: SiPANN.scee.DoubleHalfRing
    :members:
    :inherited-members:

Pushed Half Ring
==========================
.. autoclass:: SiPANN.scee.AngledHalfRing
    :members:
    :inherited-members: