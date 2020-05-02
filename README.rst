**************************
SiPANN
**************************
.. image:: https://img.shields.io/pypi/v/SiPANN.svg
   :target: https://pypi.python.org/pypi/SiPANN
   :alt: Pypi Version
.. image:: https://readthedocs.org/projects/sipann/badge/?version=latest
  :target: https://sipann.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
.. image:: https://img.shields.io/pypi/l/sphinx_rtd_theme.svg
   :target: https://pypi.python.org/pypi/sphinx_rtd_theme/
   :alt: License
.. image:: https://img.shields.io/github/last-commit/contagon/SiPANN.svg
  :target: https://github.com/contagon/SiPANN/commits/master
  :alt: Latest Commit


**Si**\ licon **P**\ hotonics with **A**\ rtificial **N**\ eural **N**\ etworks. SiPANN aims to implement various silicon photonics simulators based on machine learning techniques found in literature. The majority of these techniques are linear regression or neural networks. As a results SiPANN can return scattering parameters of (but not limited to)

* Half Rings
* Arbitrarily shaped directional couplers
* Racetrack Resonators
* Waveguides

And with the help of `simphony`_ and SiPANN's accompanying simphony wrapper

* Ring Resonators
* Doubly Coupled Rings
* Hybrid Devices (ie Green Machine)

.. _simphony: https://github.com/BYUCamachoLab/simphony

Installation
=============

SiPANN is distributed on PyPI_ and can be installed with ``pip``:

.. code:: console

   pip install SiPANN

Developmental Build
#####################


If you want a developmental build, it can be had by executing

.. code:: console
   
   git clone https://github.com/contagon/SiPANN.git
   pip install -e SiPANN/


This development version allows you to make changes to this code directly (or pull changes from GitHub) without having to reinstall SiPANN each time.

You should then be able to run the examples and tutorials in the examples folder, and call SiPANN from any other python file.

.. note::
    If installing on Windows, one of SiPANN's dependencies, ``gdspy``, requires a C compiler for installation. This can be bypassed by first installing the ``gdspy`` wheel. This is done by downloading the wheel from gdspy_, navigating to the location of the wheel, and executing

    .. code:: console

        pip install gds*.whl

    After this simply install SiPANN using your desired method.

.. _gdspy: https://github.com/heitzmann/gdspy/releases
.. _PyPI: https://pypi.org/project/SiPANN/


References
=============

SiPANN is based on a variety of methods found in various papers, including:

[1] A. Hammond, E. Potokar, and R. Camacho, "Accelerating silicon photonic parameter extraction using artificial neural networks," OSA Continuum  2, 1964-1973 (2019). 


Bibtex citation
=================

.. code::

    @misc{SiP-ANN_2019,
    	    title={SiP-ANN},
	    author={Easton Potokar, Alec M. Hammond, Ryan M. Camacho},
	    year={2019},
	    publisher={GitHub},
	    howpublished={{https://github.com/contagon/SiP-ANN}}
    }
