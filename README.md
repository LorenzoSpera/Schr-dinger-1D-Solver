# Schrödinger Equation 1D Solver & Wave-Packet Animation

A numerical solver for the 1D time-dependent Schrödinger equation with
visualization tools.\
This project simulates Gaussian wave‑packet evolution in free space or
in the presence of a rectangular potential barrier.

## Overview

The solver integrates the Schrödinger equation:

$$ i\hbar \frac{\partial \psi}{\partial t} =
\bigg(-\frac{\hbar^2}{2m}\frac{\partial^2 \psi}{\partial x^2} +
V(x)\bigg)\psi 
$$

using finite differences in space and a small explicit time‑stepping
scheme.

It generates animations of: 

$$ 
\Re(\psi(x,t)),  |\psi(x,t)|^2 
$$

## Features

-   Gaussian wave packet initialization
-   Potential barrier scattering
-   Finite-difference Laplacian
-   Simple explicit time evolution
-   Matplotlib animations
-   GIF export via PillowWriter

## Usage

``` python
from Solver_sch import schrodinger_solver
from IPython.display import display

solver = schrodinger_solver(N=1000, nsteps=500)
animation = solver.animate_wave_function()

```

To display:

``` python

display(animation)
```

To display the animation for $|\psi(x,t)|^2$ in the presence of the potential barrier:

``` python
potential_animation = solver.animate_wave_function_potential()
display(potential animation)

```

## Example Animation

Below is an example of the wave-packet evolution:

![Wave packet evolution](animation/psi_real_animation.gif)

## Dependencies

-   numpy\
-   scipy\
-   matplotlib\
-   IPython

## Author

Lorenzo Spera\
University of Perugia\
Contact: lorenzo.spera@studenti.unipg.it