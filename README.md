# Lattice Bolzmann Method for fluid flow

A quick and dirty prototype to see how easy would it be to implement this
method of fluid flow simulation.

This method is aimed at serious simulations, so for small in-game simulations
it is an overkill, and also quite inaccurate. Naive implementation as it is
done here becomes numerically unstable over few time steps. I also didn't
want to go into several layers of discrete integrations to implement a simple
fluid source.