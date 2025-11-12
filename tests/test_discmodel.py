import numpy as np

import discmodel

def test_discmodel_initialization():
    """Test initialization of DiscGalaxy class."""
    N = 1000
    a = 3.0
    M = 1.0
    vcirc = 200.0
    rmax = 30.0

    disc = discmodel.DiscGalaxy(N=N, a=a, M=M, vcirc=vcirc, rmax=rmax)

    assert disc.N == N
    assert disc.a == a
    assert disc.M == M
    assert disc.vcirc == vcirc
    assert disc.rmax == rmax * a

    assert len(disc.x) == N
    assert len(disc.y) == N
    assert len(disc.z) == N
    assert len(disc.u) == N
    assert len(disc.v) == N
    assert len(disc.w) == N

    # Check that mass is correctly assigned
    expected_mass = M / N
    assert np.all(disc.mass == expected_mass)