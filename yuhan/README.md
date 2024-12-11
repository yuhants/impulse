
Projective limits on DM-single nucleon cross section for spheres

The workflow of the code goes like this
1. Compute scattering rate vs. momentum transfer (dR/dq) for different DM and mediator massses.
   This part use the following code:
    - ../dm_xsec/rate_massive_born.py (for small spheres, assuming Born approximation)
    - ../dm_xsec/rate_massive_mediator.py (classical scattering calculation modeled
                                           after Dave's code)
    - ../dm_xsec/rate_massless_mediator.py (assume analytic formula that works for
                                            massless mediator and point charge limit)

    The second piece of code requires numerical integration and is designed to run on a cluster
    with parallel nodes. It is also numerically unstable for very light mediator (m_phi < 0.001 eV).

2. Derive 3-sigma limits on single nucleon coupling (alpha_n) from the rates
   This is done in the Jupyter notebooks:
    - ./set_limits_alpha_n_born.ipynb
    - ./set_limits_alpha_n_massive_mediator.ipynb
    - ./set_limits_alpha_n_massless_mediator.ipynb
   
   This is the point where momentum threshold and exposure are imposed. Projection onto the z axis is
   also done here.

3. Turn the alpha_n limits into limits on DM-single nucleon cross section assuming some parametrization
   This is done in the Jupyter notebooks
    - ./set_limits_light_dm.ipynb
    - ./set_limits_composite_dm.ipynb
