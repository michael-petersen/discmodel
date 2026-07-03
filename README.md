# discmodel

**Simple exponential disc generation for testing galaxy morphology routines.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ObservationalExpansions/discmodel/blob/main/LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/ObservationalExpansions/discmodel/badge.svg?branch=main)](https://coveralls.io/github/ObservationalExpansions/discmodel?branch=main)


## Installation

Installation of `discmodel` currently proceeds from local builds after cloning this repository:
```
git clone https://github.com/ObservationalExpansions/discmodel.git
```

```
pip install .
```

## Quickstart example

```python
import discmodel

N= 1_000_000
a = 1.0
M = 1.0
rmax = 5.0
nbins = 100
mmax,nmax = 2,8

# generate N distributed points
D = discmodel.DiscGalaxy(N=N, a=a, M=M)
D.generate_image(rmax=rmax, nbins=nbins)

# now you have
D.img
```

## Displaying a model image

`generate_image()` bins the particle masses into `D.img`, a 2D surface-density
image in the disc plane. The package does not require `matplotlib`, but it is
useful for inspecting the result:

```python
import matplotlib.pyplot as plt

import discmodel

N = 200_000
a = 1.0
M = 1.0
rmax = 5.0
nbins = 150

D = discmodel.DiscGalaxy(N=N, a=a, M=M, rmax=rmax, zscale=0.05*a)
D.generate_image(rmax=rmax, nbins=nbins)

fig, ax = plt.subplots(figsize=(6, 5))
image = ax.imshow(
    D.img,
    origin="lower",
    extent=(-rmax, rmax, -rmax, rmax),
    cmap="magma",
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Exponential disc surface density")
fig.colorbar(image, ax=ax, label="mass per pixel")
fig.tight_layout()
plt.show()
```

The same particle realization also keeps the underlying phase-space arrays:
`D.x`, `D.y`, `D.z` for positions and `D.u`, `D.v`, `D.w` for velocities.
Setting `zscale=0.0` gives a flat disc, while a positive `zscale` samples an
isothermal vertical slab.
