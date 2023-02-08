Tutorial
========

Conservation Laws
-----------------

This tutorial shows how to implement a new method for a known scalar conservation
law. There are existing examples of this in the code that cover Burgers' equation,
the continuity equation, the heat equation, etc. For some diversity, we will
look at the traffic flow model

.. math::

    \begin{cases}
    u_t + f(u)_x = 0, \\
    u(0, x) = u_0(x), \\
    u(t, a) = u_a(t),
    \end{cases}

where :math:`u_0` is the initial condition, :math:`u_a` is the inflow boundary
condition (assuming the right boundary is the inflow) and the flux is given by

.. math::

    f(u) = u (1 - u).

To implement a scheme for his model, we will make use of the existing
infrastructure based on :class:`~pyshocks.SchemeBase`. This involves making
a subclass and implementing a subset functions based on the
:func:`~functools.singledispatch` mechanism. They are:

* :func:`~pyshocks.flux`,
* :func:`~pyshocks.numerical_flux`,
* :func:`~pyshocks.apply_operator`,
* :func:`~pyshocks.predict_timestep`,

where the first two functions are meant to assist only for flux-based models
as the one above. In the most general case, it suffices to implement
:func:`~pyshocks.apply_operator`.

As we are dealing with a conservation law, we would subclass the existing
:class:`~pyshocks.ConservationLawScheme`. This class already implements
:func:`~pyshocks.apply_operator` so it suffices to implement the flux functions.
We start with a new class

.. literalinclude:: ../examples/traffic-flow-model.py
    :lines: 4-15
    :language: python
    :linenos:

for which we would need to implement the physical flux above

.. literalinclude:: ../examples/traffic-flow-model.py
    :lines: 18-25
    :language: python
    :linenos:

Now, we are in a position to implement a numerical scheme for this model.
For this, we use the standard upwind scheme

.. literalinclude:: ../examples/traffic-flow-model.py
    :lines: 28-39
    :language: python
    :linenos:

Note that the velocity :math:`(1 - 2 u)` is always positive in the traffic flow
model, so the upwind scheme always chooses the upwind value. This then suffices
to implement the scheme. To evolve the traffic flow model in time, we can use
the following setup

.. literalinclude:: ../examples/traffic-flow-model.py
    :lines: 72-90
    :language: python
    :linenos:

The complete example can be found in
:download:`examples/traffic-flow-model.py <../examples/traffic-flow-model.py>`.
