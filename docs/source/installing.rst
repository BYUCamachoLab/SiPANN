**************************
Installation
**************************


SiPANN is distributed on PyPI_ and can be installed with ``pip``:

.. code:: console

   pip install SiPANN

Developmental Build
======================

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
