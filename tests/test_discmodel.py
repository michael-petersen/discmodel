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

def test_discmodel_rotation():
    """Test the rotation angles in DiscModel."""
    N = 1000
    a = 3.0
    M = 1.0
    vcirc = 200.0
    rmax = 30.0

    disc = discmodel.DiscGalaxy(N=N, a=a, M=M, vcirc=vcirc, rmax=rmax)

    R1 = disc.make_rotation_matrix(45.0,0.,30.,False)
    # check R1 is 3x3
    assert R1.shape == (3,3)

    R2 = disc.make_rotation_matrix(45.0,0.,30.,True)
    # check R2 is 3x3
    assert R2.shape == (3,3)

    # check R1 and R2 are different
    assert not np.allclose(R1,R2)

    # apply the rotation
    disc.rotate_disc(45.0,0.,30.,False)
    # check lengths are unchanged
    assert len(disc.x) == N
    assert len(disc.y) == N 
    assert len(disc.z) == N

    disc.rotate_disc(45.0,0.,30.,True)
    # check lengths are unchanged
    assert len(disc.x) == N
    assert len(disc.y) == N 
    assert len(disc.z) == N

    # try special case of N=1
    N = 1

    disc = discmodel.DiscGalaxy(N=N, a=a, M=M, vcirc=vcirc, rmax=rmax)

    # apply the rotation
    disc.rotate_disc(45.0,0.,30.,False)



def test_discmodel_phasespace_input():
    """Test initialization of DiscGalaxy class with phasespace input."""
    N = 1000
    a = 3.0
    M = 1.0
    vcirc = 200.0
    rmax = 30.0

    disc = discmodel.DiscGalaxy(N=N, a=a, M=M, vcirc=vcirc, rmax=rmax)

    newdisc = discmodel.DiscGalaxy(phasespace=(disc.x,disc.y,disc.z,disc.u,disc.v,disc.w))


def test_discmodel_version():
    """Test that the version string is correctly set."""
    import discmodel
    assert isinstance(discmodel.__version__, str)

def test_discmodel_image():
    N = 1000
    a = 3.0
    M = 1.0
    vcirc = 200.0
    rmax = 30.0
    noiselevel = -100.0
    nbins = 50

    disc = discmodel.DiscGalaxy(N=N, a=a, M=M, vcirc=vcirc, rmax=rmax)
    disc.generate_image(rmax,nbins,noiselevel=noiselevel)

    # add some noise
    noiselevel = 0.1
    disc.generate_image(rmax,nbins,noiselevel=noiselevel)

    # check that r and p are set
    assert hasattr(disc,'r')
    assert hasattr(disc,'p')


def test_discmodel_expansion():
    N = 1000
    a = 3.0
    M = 1.0
    vcirc = 200.0
    rmax = 30.0
    noiselevel = -100.0
    nbins = 50

    disc = discmodel.DiscGalaxy(N=N, a=a, M=M, vcirc=vcirc, rmax=rmax)

    # try to compute expansion before image - should raise error

    disc.generate_image(rmax,nbins,noiselevel=noiselevel)

    # add some noise
    noiselevel = 0.1
    disc.generate_image(rmax,nbins,noiselevel=noiselevel)


    # compute the expansion
    E1 = disc.make_expansion(mmax=4,nmax=4,rscl=1.0,xmax=rmax,noisy=False)

    # compute a noisy expansion
    E2 = disc.make_expansion(mmax=4,nmax=4,rscl=1.0,xmax=rmax,noisy=True)
    assert E2 is not None
    # compute the expansion from the particles
    E3 = disc.make_particle_expansion(mmax=4,nmax=4,rscl=1.0)
    assert E3 is not None
    # compute A1
    a1 = disc.compute_a1(E1)

    # check it is a float
    assert isinstance(a1,float)


def test_discmodel_resampling():
    N = 1000
    a = 3.0
    M = 1.0
    vcirc = 200.0
    rmax = 30.0
    noiselevel = -100.0
    nbins = 50

    disc = discmodel.DiscGalaxy(N=N, a=a, M=M, vcirc=vcirc, rmax=rmax)
    disc.generate_image(rmax,nbins,noiselevel=noiselevel)

    # compute the expansion
    E1 = disc.make_expansion(mmax=4,nmax=4,rscl=1.0,xmax=rmax,noisy=False)

    newdisc = disc.resample_expansion(E1)
    
    # check newdisc is Nx2 (sampled from 2d image only)
    assert newdisc.shape == (N, 2)
