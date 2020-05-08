#############################
SCEE Optimization
#############################

These are various functions to perform and aid in using SCEE as an inverse design optimizer. 

Do to how fast SCEE is, inverse design of power splitting directional couplers can be achieved via an optimizer. This has been implemented and can be used via the `SiPANN.scee_opt` module, speficially the `make_coupler` function. It implements a global optimization, then a local optimization to best find the ideal coupler.

This is done by defining the length of the coupler and various control points along the coupler as parameters that our optimizer can choose that result in a $\kappa$ closest to $\kappa_{goal}$. The coupler is then defined using the control points plugged into a Bezier Curve. Note that the Bezier curve defined by the control points is the gap between waveguides, not the geometry of the waveguides themselves. However, since each of these directional couplers is symmetric the inner boundary of the waveguides are just half of the gap.

Further, for our objective function, we compute $\kappa$ for a sweep of wavelength points using SCEE, and then calculate the MSE by comparing it to $\kappa_{goal}$. Various constraints are also put on the coupler, like ensuring the edges of the coupler are far enough apart and making sure there's no bends that are too sharp. To learn more about the process, see [INSERT PAPER WHEN PUBLISHED].


*************************
Optimizer and Utilities
*************************

Inverse Design Optimizer
==============================
.. autofunction:: SiPANN.scee_opt.make_coupler

Saving Couplers
==============================
.. autofunction:: SiPANN.scee_opt.save_coupler

Loading Couplers
==============================
.. autofunction:: SiPANN.scee_opt.load_coupler

Premade Couplers
==============================
.. autofunction:: SiPANN.scee_opt.premade_coupler


*********************
Helper Functions
*********************

Bernstein Transformation
==============================
.. autofunction:: SiPANN.scee_opt.bernstein_quick

Bezier Function
==============================
.. autofunction:: SiPANN.scee_opt.bezier_quick



