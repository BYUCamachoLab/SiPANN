# Installation

SiPANN is distributed via [PyPI](https://pypi.org/project/SiPANN/) and can be 
installed with ``pip``:

```bash
pip install SiPANN
```

## Developmental Build

If you want a developmental build, clone and install the package to your 
environment in editable mode:

```bash
git clone https://github.com/BYUCamachoLab/SiPANN.git
pip install -e SiPANN/
```

You should then be able to run the examples and tutorials in the examples 
folder and import SiPANN from any other python file.

````{note}
If installing on Windows, one of SiPANN's dependencies, ``gdspy``, requires a 
C compiler for installation. This can be bypassed by first installing the 
``gdspy`` wheel. This is done by downloading the wheel from the 
[gdspy repository](https://github.com/heitzmann/gdspy/releases), navigating 
your terminal to the location of the wheel, and executing

```bash
pip install gds*.whl
```

After this simply install SiPANN using your desired method.
````
