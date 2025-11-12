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

```
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