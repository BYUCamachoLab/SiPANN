#############################
Directional Couplers Module
#############################

*********************
Helper Functions
*********************

Base Directional Coupler Class
==============================
.. autoclass:: SiPANN.dc.DC
    :members:

Effective Index Finder
==============================
.. autofunction:: SiPANN.dc.get_neff

Integration Coefficient Finder
==============================
.. autofunction:: SiPANN.dc.get_coeffs

Input Cleaner
==============================
.. autofunction:: SiPANN.dc.clean_inputs


*********************
Coupling Devices
*********************

Symmetric Coupler
======================
.. autoclass:: SiPANN.dc.GapFuncSymmetric
    :members:
    :inherited-members:

Non-Symmetric Coupler
==========================
.. autoclass:: SiPANN.dc.GapFuncAntiSymmetric
    :members:
    :inherited-members:

Half Racetrack Resonator
==========================
.. autoclass:: SiPANN.dc.Racetrack
    :members:
    :inherited-members:

Coupling Straight Waveguides
==============================
.. autoclass:: SiPANN.dc.Straight
    :members:
    :inherited-members:

Standard Shaped Directional Coupler
====================================
.. autoclass:: SiPANN.dc.Standard
    :members:
    :inherited-members:

Double Half Ring
==========================
.. autoclass:: SiPANN.dc.DoubleRR
    :members:
    :inherited-members:

Half Ring
==========================
.. autoclass:: SiPANN.dc.RR
    :members:
    :inherited-members:

Pushed Half Ring
==========================
.. autoclass:: SiPANN.dc.AngledRR
    :members:
    :inherited-members:

Waveguide
==========================
.. autoclass:: SiPANN.dc.Waveguide
    :members:
    :inherited-members: