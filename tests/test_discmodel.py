import numpy as np
import pytest

import discmodel
import discmodel.discmodel as discmodel_module

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
    assert newdisc.N == N


def test_discmodel_initialization_does_not_reset_global_rng():
    """Test DiscGalaxy uses its own RNG for reproducible initialization."""
    np.random.seed(1)
    expected = np.random.rand()

    np.random.seed(1)
    discmodel.DiscGalaxy(N=10)
    actual = np.random.rand()

    assert actual == expected


def test_discmodel_seed_reproducibility():
    """Test seeded DiscGalaxy instances generate reproducible phase space."""
    disc1 = discmodel.DiscGalaxy(N=20, seed=123)
    disc2 = discmodel.DiscGalaxy(N=20, seed=123)
    disc3 = discmodel.DiscGalaxy(N=20, seed=124)

    for attr in ("x", "y", "z", "u", "v", "w"):
        assert np.allclose(getattr(disc1, attr), getattr(disc2, attr))

    assert not np.allclose(disc1.x, disc3.x)


def test_discmodel_basic_disc_has_isothermal_slab_thickness():
    """Test basic DiscGalaxy samples a non-flat isothermal slab."""
    N = 10000
    a = 3.0
    vcirc = 200.0
    zscale = 0.2

    disc = discmodel.DiscGalaxy(N=N, a=a, vcirc=vcirc, zscale=zscale)

    assert disc.zscale == zscale
    assert not np.allclose(disc.z, 0.0)
    assert not np.allclose(disc.w, 0.0)
    assert np.std(disc.z) == pytest.approx(zscale, rel=0.05)
    assert np.std(disc.w) == pytest.approx((vcirc/a)*zscale, rel=0.05)


def test_discmodel_basic_disc_allows_zero_zscale():
    """Test zscale=0 recovers a flat disc without failing."""
    disc = discmodel.DiscGalaxy(N=100, zscale=0.0)

    assert disc.zscale == 0.0
    assert np.all(disc.z == 0.0)
    assert np.all(disc.w == 0.0)


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


def test_discmodel_expansion_requires_flex(monkeypatch):
    monkeypatch.setattr(discmodel_module, "HAS_FLEX", False)
    disc = discmodel.DiscGalaxy(N=10)

    with pytest.raises(ImportError, match="flex is not available"):
        disc.make_expansion(mmax=4, nmax=4, rscl=1.0)

    with pytest.raises(ImportError, match="flex is not available"):
        disc.make_particle_expansion(mmax=4, nmax=4, rscl=1.0)


def test_discmodel_expansion_requires_image_data(monkeypatch):
    monkeypatch.setattr(discmodel_module, "HAS_FLEX", True)
    disc = discmodel.DiscGalaxy(N=10)

    with pytest.raises(RuntimeError, match="Run generate_image first"):
        disc.make_expansion(mmax=4, nmax=4, rscl=1.0)


def test_discmodel_noisy_expansion_requires_noisy_image(monkeypatch):
    monkeypatch.setattr(discmodel_module, "HAS_FLEX", True)
    disc = discmodel.DiscGalaxy(N=10)
    disc.generate_image(rmax=30.0, nbins=5)

    with pytest.raises(RuntimeError, match="Run generate_image with noiselevel first"):
        disc.make_expansion(mmax=4, nmax=4, rscl=1.0, noisy=True)


def test_discmodel_particle_expansion_uses_flex(monkeypatch):
    class DummyFlex:
        def __init__(self, rscl, mmax, nmax, rval, phi, mass):
            self.rscl = rscl
            self.mmax = mmax
            self.nmax = nmax
            self.rval = rval
            self.phi = phi
            self.mass = mass

    class DummyFlexModule:
        FLEX = DummyFlex

    monkeypatch.setattr(discmodel_module, "HAS_FLEX", True)
    monkeypatch.setattr(discmodel_module, "flex", DummyFlexModule)

    disc = discmodel.DiscGalaxy(N=10, M=2.0)
    expansion = disc.make_particle_expansion(mmax=3, nmax=2, rscl=1.5)

    assert expansion.rscl == 1.5
    assert expansion.mmax == 3
    assert expansion.nmax == 2
    assert expansion.rval.shape == (10,)
    assert expansion.phi.shape == (10,)
    assert np.all(expansion.mass == 0.2)


def test_discmodel_resampling_requires_lintsampler(monkeypatch):
    monkeypatch.setattr(discmodel_module, "HAS_LINTSAMPLER", False)
    disc = discmodel.DiscGalaxy(N=10)

    with pytest.raises(ImportError, match="lintsampler is not available"):
        disc.resample_expansion(object())


def test_discmodel_compute_a1_with_dummy_expansion():
    class DummyExpansion:
        coscoefs = np.array([[1.0, 0.0], [2.0, 0.0]])
        sincoefs = np.array([[0.0, 0.0], [0.0, 0.0]])

    disc = discmodel.DiscGalaxy(N=10)

    assert disc.compute_a1(DummyExpansion()) == 2.0


def test_discmodel_expansion():
    pytest.importorskip("flex")

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
    pytest.importorskip("flex")
    pytest.importorskip("lintsampler")

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


def test_discmodel_resampling_uses_flattened_reconstruction_grid(monkeypatch):
    class DummyExpansion:
        def laguerre_reconstruction(self, rr, pp):
            rr = np.asarray(rr)
            pp = np.asarray(pp)
            assert rr.ndim == 1
            assert pp.ndim == 1
            assert rr.shape == pp.shape
            self.reconstruction = np.ones_like(rr)

    class DummyDensityGrid:
        def __init__(self, axes, rndmpdf):
            self.axes = axes
            self.rndmpdf = rndmpdf

        def _calculate_faverages(self):
            return np.ones_like(self.vertex_densities)

        def _calculate_volumes(self):
            return np.ones_like(self.vertex_densities)

    class DummyLintSampler:
        def __init__(self, grid):
            self.grid = grid

        def sample(self, n):
            return np.zeros((n, 2))

    class DummyLintSamplerModule:
        DensityGrid = DummyDensityGrid
        LintSampler = DummyLintSampler

    monkeypatch.setattr(discmodel_module, "HAS_LINTSAMPLER", True)
    monkeypatch.setattr(discmodel_module, "lintsampler", DummyLintSamplerModule)

    disc = discmodel.DiscGalaxy(N=10)
    disc.generate_image(rmax=30.0, nbins=5)

    pos = disc.resample_expansion(DummyExpansion())

    assert pos.shape == (10, 2)
