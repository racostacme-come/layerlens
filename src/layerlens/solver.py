"""Vertex-centered finite volumes for J' + r*c = f, J = v*c - D*c'."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_banded

FloatArray = NDArray[np.float64]


def _finite(value: ArrayLike, name: str) -> FloatArray:
    out = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out


def _field(value: ArrayLike, size: int, name: str) -> FloatArray:
    out = _finite(value, name)
    if out.ndim == 0:
        return np.full(size, float(out))
    if out.shape != (size,):
        raise ValueError(f"{name} must be scalar or shape ({size},)")
    return out.copy()


def bernoulli(z: ArrayLike) -> FloatArray:
    """Evaluate B(z)=z/(exp(z)-1), including B(0)=1, without overflow.

    Input must be finite. Very large positive arguments may underflow to zero.
    The return is a NumPy array (zero-dimensional for scalar input).
    """
    z = _finite(z, "Bernoulli argument")
    q = np.abs(z)
    small = q < 1.0e-4
    b = np.empty_like(q)
    s = q[small]
    b[small] = 1 - s / 2 + s**2 / 12 - s**4 / 720
    t = q[~small]
    b[~small] = t * np.exp(-t) / (-np.expm1(-t))
    return np.where(z < 0, b + q, b)


def layer_profile(t: ArrayLike, peclet: ArrayLike) -> FloatArray:
    """Stable (exp(P*t)-1)/(exp(P)-1) on 0 <= t <= 1.

    Broadcasting is supported. This is the exact normalized, constant-flux
    solution on an interval with constant coefficients and no reaction/source.
    """
    t, p = np.broadcast_arrays(_finite(t, "t"), _finite(peclet, "Peclet number"))
    if np.any((t < 0) | (t > 1)):
        raise ValueError("t must be in [0, 1]")
    w = np.empty_like(t)
    small = np.abs(p) < 1.0e-5
    s, q = t[small], p[small]
    w[small] = s * (1 + q * (s - 1) / 2 + q**2 * (2 * s**2 - 3 * s + 1) / 12)
    pos = (p > 0) & ~small
    s, q = t[pos], p[pos]
    w[pos] = np.exp(q * (s - 1)) * np.expm1(-q * s) / np.expm1(-q)
    neg = (p < 0) & ~small
    w[neg] = np.expm1(p[neg] * t[neg]) / np.expm1(p[neg])
    return w


@dataclass(frozen=True)
class Solution:
    """Solution arrays, edge fluxes, and interior dual-cell residuals.

    Arrays are independent read-only snapshots. Residuals have units of flux;
    balance_error is their sum, not a continuum-error estimate.
    """

    x: FloatArray
    concentration: FloatArray
    flux: FloatArray
    residual: FloatArray
    peclet: FloatArray
    scheme: str

    @property
    def balance_error(self) -> float:
        """Net discrete balance defect over the interior dual cells."""
        return float(np.sum(self.residual))

    def reconstruct(self, points: ArrayLike, *, kind: str = "linear") -> FloatArray:
        """Interpolate within the domain; never extrapolate.

        `exponential` solves the homogeneous edge problem using the same frozen
        coefficients. It is not generally exact with variable D, r, or f.
        """
        q = _finite(points, "points")
        if np.any((q < self.x[0]) | (q > self.x[-1])):
            raise ValueError("reconstruction points must lie in the domain")
        if kind not in ("linear", "exponential"):
            raise ValueError("kind must be 'linear' or 'exponential'")
        i = np.clip(np.searchsorted(self.x, q, side="right") - 1, 0, len(self.x) - 2)
        t = (q - self.x[i]) / (self.x[i + 1] - self.x[i])
        w = t if kind == "linear" else layer_profile(t, self.peclet[i])
        return self.concentration[i] * (1 - w) + self.concentration[i + 1] * w


def solve(
    x: ArrayLike,
    *,
    velocity: float,
    diffusivity: ArrayLike,
    boundary: tuple[float, float] = (0.0, 1.0),
    reaction: ArrayLike = 0.0,
    source: ArrayLike = 0.0,
    scheme: str = "sg",
) -> Solution:
    """Solve stationary 1D transport with two Dirichlet boundary values.

    x: strictly increasing finite nodes, including both endpoints (at least 3).
    velocity: constant finite scalar, either sign.
    diffusivity: positive scalar or one value per edge, frozen on that edge.
    reaction: nonnegative scalar or one value per interior node.
    source: scalar or one value per interior node; lumped dual-cell quadrature.
    scheme: 'sg', 'central', or 'upwind'. Central is a diagnostic comparator;
        its off-diagonal signs may fail when an edge |Peclet| exceeds 2.

    Storage/work are O(N) via SciPy's pivoted banded direct solve. A finite
    solution does not imply good conditioning; inspect the returned residuals.
    """
    x = _finite(x, "x").copy()
    if x.ndim != 1 or len(x) < 3:
        raise ValueError("x must be a 1D array with at least 3 nodes")
    h = np.diff(x)
    if not np.all(np.isfinite(h)) or np.any(h <= 0):
        raise ValueError("x must be strictly increasing with finite spacing")
    v = _finite(velocity, "velocity")
    if v.ndim != 0:
        raise ValueError("velocity must be a scalar")
    v = float(v)
    d = _field(diffusivity, len(h), "diffusivity")
    r = _field(reaction, len(x) - 2, "reaction")
    f = _field(source, len(x) - 2, "source")
    bc = _finite(boundary, "boundary")
    if bc.shape != (2,):
        raise ValueError("boundary must contain two values")
    if np.any(d <= 0) or np.any(r < 0):
        raise ValueError("diffusivity must be positive and reaction nonnegative")
    if scheme not in ("sg", "central", "upwind"):
        raise ValueError("scheme must be 'sg', 'central', or 'upwind'")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        g = d / h
        p = v / g
        if np.any(g == 0) or not np.all(np.isfinite(p)):
            raise ValueError("coefficients exceed floating-point range")
        if scheme == "sg":
            a, b = g * bernoulli(-p), g * bernoulli(p)
        elif scheme == "central":
            a, b = g + v / 2, g - v / 2
        else:
            a, b = g + max(v, 0), g + max(-v, 0)
        volume = (h[:-1] + h[1:]) / 2
        diag = a[1:] + b[:-1] + r * volume
        rhs = f * volume
        rhs[0] += a[0] * bc[0]
        rhs[-1] += b[-1] * bc[1]
        band = np.zeros((3, len(x) - 2))
        band[0, 1:] = -b[1:-1]
        band[1] = diag
        band[2, :-1] = -a[1:-1]
        c = np.concatenate(([bc[0]], solve_banded((1, 1), band, rhs), [bc[1]]))
        flux = a * c[:-1] - b * c[1:]
        residual = np.diff(flux) + (r * c[1:-1] - f) * volume
    if not np.all(np.isfinite(c)):
        raise FloatingPointError("nonfinite solution")
    arrays = (x, c, flux, residual, p)
    for array in arrays:
        array.setflags(write=False)
    return Solution(*arrays, scheme)
