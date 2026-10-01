import numpy as np
import pytest
from numpy.testing import assert_allclose

from layerlens import bernoulli, layer_profile, solve


@pytest.mark.parametrize("p", [-10000, -100, -1, -1e-8, 0, 1e-8, 1, 100, 10000])
def test_bernoulli_identity_and_limits(p):
    assert bernoulli(p) >= 0
    assert np.isfinite(bernoulli(p))
    assert_allclose(bernoulli(-p) - bernoulli(p), p, atol=2e-15)
    if abs(p) < 1e-7:
        assert_allclose(bernoulli(p), 1 - p / 2, atol=1e-15)


@pytest.mark.parametrize("p", [-1000, -20, -1e-8, 0, 1e-8, 20, 1000])
def test_exact_nodes_and_exponential_field(p):
    x = np.linspace(0, 1, 48) ** 1.3
    s = solve(x, velocity=p, diffusivity=1, boundary=(0.2, 1.3))

    def exact(q):
        return 0.2 + 1.1 * layer_profile(q, p)

    assert_allclose(s.concentration, exact(x), atol=2e-12)
    q = np.linspace(0, 1, 1001)
    assert_allclose(s.reconstruct(q, kind="exponential"), exact(q), atol=2e-12)
    assert np.max(np.abs(s.residual)) < 1e-10


def test_independent_diffusion_parabola():
    x = np.linspace(0, 1, 45)
    s = solve(x, velocity=0, diffusivity=2, source=4, boundary=(0, 0))
    assert_allclose(s.concentration, x * (1 - x), atol=2e-14)
    assert_allclose(s.flux, 4 * ((x[:-1] + x[1:]) / 2) - 2, atol=1e-13)


def test_discontinuous_diffusion_resistances():
    x = np.array([0, 0.1, 0.4, 0.5, 0.8, 1.0])
    d = np.array([1, 1, 1, 0.01, 0.01])
    resistance = np.cumsum(np.diff(x) / d)
    exact = np.r_[0, resistance / resistance[-1]]
    s = solve(x, velocity=0, diffusivity=d)
    assert_allclose(s.concentration, exact, atol=1e-14)
    assert_allclose(s.flux, -1 / resistance[-1], atol=1e-14)


@pytest.mark.parametrize("velocity", [-25, 0, 25])
def test_positive_source_reaction_and_global_balance(velocity):
    rng = np.random.default_rng(314)
    x = np.r_[0, np.cumsum(rng.uniform(0.01, 0.05, 99))]
    d = rng.uniform(0.1, 2, len(x) - 1)
    r = rng.uniform(0, 2, len(x) - 2)
    f = rng.uniform(0, 1, len(x) - 2)
    s = solve(x, velocity=velocity, diffusivity=d, reaction=r, source=f)
    assert np.min(s.concentration) >= -1e-13
    vol = (x[2:] - x[:-2]) / 2
    balance = s.flux[-1] - s.flux[0] + np.sum((r * s.concentration[1:-1] - f) * vol)
    assert abs(balance) < 5e-12
    assert_allclose(balance, s.balance_error, atol=1e-14)


def test_central_oscillates_sg_and_upwind_bounded():
    x = np.linspace(0, 1, 21)
    results = {
        k: solve(x, velocity=1, diffusivity=0.01, scheme=k) for k in ("sg", "central", "upwind")
    }
    assert results["central"].concentration.min() < -0.4
    for k in ("sg", "upwind"):
        assert results[k].concentration.min() >= -1e-14
        assert results[k].concentration.max() <= 1 + 1e-14


def test_manufactured_variable_diffusion_second_order():
    errors = []
    for n in (32, 64, 128):
        x = np.linspace(0, 1, n + 1)
        xi = x[1:-1]
        c = 1 + xi + np.sin(np.pi * xi)
        cp = 1 + np.pi * np.cos(np.pi * xi)
        cpp = -(np.pi**2) * np.sin(np.pi * xi)
        s = solve(
            x,
            velocity=2,
            diffusivity=1 + (x[:-1] + x[1:]) / 2,
            reaction=0.7,
            source=cp - (1 + xi) * cpp + 0.7 * c,
            boundary=(1, 2),
        )
        errors.append(np.max(np.abs(s.concentration - (1 + x + np.sin(np.pi * x)))))
    assert np.all(np.log2(np.array(errors[:-1]) / errors[1:]) > 1.95)


def test_smallest_system_and_readonly_snapshot():
    x = np.array([0.0, 0.4, 1.0])
    s = solve(x, velocity=0, diffusivity=1, boundary=(2, 3))
    x[1] = 0.5
    assert_allclose(s.concentration, [2, 2.4, 3])
    assert s.x[1] == 0.4
    with pytest.raises(ValueError):
        s.concentration[0] = 5
    for q in (-0.01, 1.01, np.nan):
        with pytest.raises(ValueError):
            s.reconstruct(q)
    with pytest.raises(ValueError):
        s.reconstruct(0.3, kind="cubic")


@pytest.mark.parametrize(
    "changes",
    [
        {"x": [0, 1]},
        {"x": [0, 0, 1]},
        {"x": [0, np.nan, 1]},
        {"x": [[0, 0.5, 1]]},
        {"velocity": [1, 2]},
        {"velocity": np.inf},
        {"diffusivity": 0},
        {"diffusivity": -1},
        {"diffusivity": [1]},
        {"source": [1, 2]},
        {"reaction": -1},
        {"boundary": [0]},
        {"boundary": [0, np.inf]},
        {"scheme": "magic"},
    ],
)
def test_invalid_inputs(changes):
    kwargs = dict(x=[0, 0.5, 1], velocity=1, diffusivity=1)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        solve(**kwargs)


def test_profiles_against_direct_formula():
    t = np.linspace(0, 1, 51)
    for p in (-10, -0.01, 0.01, 10):
        assert_allclose(layer_profile(t, p), np.expm1(p * t) / np.expm1(p), atol=2e-15)
    with pytest.raises(ValueError):
        layer_profile([-1, 0], 1)
    with pytest.raises(ValueError):
        bernoulli(np.inf)
