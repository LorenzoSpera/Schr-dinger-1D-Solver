# Schrödinger Equation 1D Solver & Wave-Packet Animation

A numerical solver for the 1D time-dependent Schrödinger equation with
visualization tools.\
This project simulates Gaussian wave‑packet evolution in free space or
in the presence of a rectangular potential barrier.

## Overview

The solver integrates the Schrödinger equation:

\[ i`\hbar `{=tex}`\frac{\partial \psi}{\partial t}`{=tex} =
-`\frac{\hbar^2}{2m}`{=tex}`\frac{\partial^2 \psi}{\partial x^2}`{=tex} +
V(x)`\psi`{=tex}, \]

using finite differences in space and a small explicit time‑stepping
scheme.

It generates animations of: - Re(ψ) - Im(ψ) - \|ψ\|²

## Features

-   Gaussian wave packet initialization\
-   Potential barrier scattering\
-   Finite-difference Laplacian\
-   Simple explicit time evolution\
-   Matplotlib animations\
-   GIF export via PillowWriter

## Usage

``` python
from Solver_sch import schrodinger_solver
from IPython.display import HTML

solver = schrodinger_solver(N=1000, nsteps=500)
ani = solver.animate_wave_function()
HTML(ani.to_jshtml())
```

To save:

``` python
from matplotlib.animation import PillowWriter
ani.save("psi.gif", writer=PillowWriter(fps=30))
```

## Dependencies

-   numpy\
-   scipy\
-   matplotlib\
-   IPython

## Author

Lorenzo Spera\
University of Perugia\
Contact: lorenzo.spera@studenti.unipg.it