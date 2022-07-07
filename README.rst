**************************
SiPANN 2.0.0
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
.. image:: https://github.com/contagon/SiPANN/workflows/build%20(pip)/badge.svg
  :target: https://github.com/contagon/SiPANN/actions?query=workflow%3A%22build+%28pip%29%22
  :alt: build

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
==========

SiPANN is based on a variety of methods found in various papers, including:

[1] A. Hammond, E. Potokar, and R. Camacho, "Accelerating silicon photonic parameter extraction using artificial neural networks," OSA Continuum  2, 1964-1973 (2019).


Bibtex citation
===============

.. code::

    @misc{SiP-ANN_2019,
    	    title={SiP-ANN},
	    author={Easton Potokar, Alec M. Hammond, Ryan M. Camacho},
	    year={2019},
	    publisher={GitHub},
	    howpublished={{https://github.com/contagon/SiP-ANN}}
    }


Releasing
=========

Make sure you have committed a changelog file titled 
"[major].[minor].[patch]-changelog.md" before bumping version. 

To bump version prior to a release, run one of the following commands:

.. code:: bash

   bumpversion major
   bumpversion minor
   bumpversion patch

This will automatically create a git tag in the repository with the 
corrresponding version number and commit the modified files (where version
numbers were updated). Pushing the tags (a manual process) to the remote will 
automatically create a new release. Releases are automatically published to 
PyPI and GitHub when git tags matching the "v*" pattern are created 
(e.g. "v0.2.1"), as bumpversion does.

To view the tags on the local machine, run :code:`git tag`. To push the tags to
the remote server, you can run :code:`git push origin <tagname>`.

For code quality, please run isort and black before committing (note that the
latest release of isort may not work through VSCode's integrated terminal, and
it's safest to run it separately through another terminal).
