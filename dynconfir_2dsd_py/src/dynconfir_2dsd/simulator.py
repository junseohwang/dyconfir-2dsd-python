from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


ArrayLike = Any


@dataclass
class TwoDSDResult:
    """Container returned by :func:`r2dsd`.

    Attributes
    ----------
    rt:
        Response times in seconds.
    response:
        Boundary hit: ``-1`` lower, ``+1`` upper, ``0`` timeout.
    conf:
        Continuous confidence variable returned by the dynConfiR 2DSD special case.
    evidence_term:
        The pre-WEV confidence evidence term. For 2DSD this is the core confidence variable.
    visibility_term:
        Visibility process draw. Present for completeness; ignored for the 2DSD setting where
        ``w=1``.
    drift_draw:
        Trial-wise drift draw after adding across-trial variability ``sv``.
    """

    rt: np.ndarray
    response: np.ndarray
    conf: np.ndarray
    evidence_term: np.ndarray
    visibility_term: np.ndarray
    drift_draw: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rt": self.rt,
                "response": self.response,
                "conf": self.conf,
                "evidence_term": self.evidence_term,
                "visibility_term": self.visibility_term,
                "drift_draw": self.drift_draw,
            }
        )


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _as_1d_array(value: ArrayLike, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    arr = arr.reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=float)
    if arr.size != n:
        raise ValueError(f"{name} must be scalar or length n={n}, got length {arr.size}.")
    return arr.astype(float, copy=False)


def _validate_params(
    a: np.ndarray,
    v: np.ndarray,
    t0_mid: np.ndarray,
    z_rel: np.ndarray,
    d: np.ndarray,
    sz_rel: np.ndarray,
    sv: np.ndarray,
    st0: np.ndarray,
    tau: np.ndarray,
    lambda_: np.ndarray,
) -> None:
    if np.any(a <= 0):
        raise ValueError("a must be > 0.")
    if np.any(sz_rel < 0) or np.any(sz_rel > 1):
        raise ValueError("sz must be in [0, 1] on the relative scale.")
    if np.any(st0 < 0):
        raise ValueError("st0 must be >= 0.")
    if np.any(sv < 0):
        raise ValueError("sv must be >= 0.")
    if np.any(t0_mid - np.abs(0.5 * d) - 0.5 * st0 < 0):
        raise ValueError("Invalid combination of t0, d, and st0; lower RT support must be >= 0.")
    if np.any(z_rel - 0.5 * sz_rel <= 0):
        raise ValueError("Invalid z/sz combination; relative start range crosses lower bound.")
    if np.any(z_rel + 0.5 * sz_rel >= 1):
        raise ValueError("Invalid z/sz combination; relative start range crosses upper bound.")
    if np.any(tau < 0):
        raise ValueError("tau must be >= 0.")
    if np.any(lambda_ < 0):
        raise ValueError("lambda must be >= 0.")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def r2dsd(
    n: int,
    *,
    a: ArrayLike,
    v: ArrayLike,
    t0: ArrayLike = 0.0,
    z: ArrayLike = 0.5,
    d: ArrayLike = 0.0,
    sz: ArrayLike = 0.0,
    sv: ArrayLike = 0.0,
    st0: ArrayLike = 0.0,
    tau: ArrayLike = 1.0,
    lambda_: ArrayLike = 0.0,
    s: ArrayLike = 1.0,
    delta: float = 0.01,
    maxrt: float = 15.0,
    z_absolute: bool = False,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> TwoDSDResult:
    """Simulate from the dynConfiR 2DSD special case.

    This ports the package path ``r2DSD() -> r_WEV(..., w=1, muvis=0, sigvis=1, svis=1)``.
    Parameterization follows the *user-facing* R documentation:

    - ``t0`` is the lower bound of the uniform non-decision-time distribution of length ``st0``.
    - ``z`` and ``sz`` are relative to ``a`` unless ``z_absolute=True``.
    - ``s`` rescales ``a``, ``v``, and ``sv`` internally, matching the R wrapper.

    Returns the continuous confidence variable ``conf``; it is **not** discretized.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if delta <= 0:
        raise ValueError("delta must be > 0.")
    if maxrt <= 0:
        raise ValueError("maxrt must be > 0.")

    if rng is None:
        rng = np.random.default_rng(seed)

    a = _as_1d_array(a, n, "a")
    v = _as_1d_array(v, n, "v")
    t0 = _as_1d_array(t0, n, "t0")
    z = _as_1d_array(z, n, "z")
    d = _as_1d_array(d, n, "d")
    sz = _as_1d_array(sz, n, "sz")
    sv = _as_1d_array(sv, n, "sv")
    st0 = _as_1d_array(st0, n, "st0")
    tau = _as_1d_array(tau, n, "tau")
    lambda_ = _as_1d_array(lambda_, n, "lambda_")
    s = _as_1d_array(s, n, "s")

    # Match the R wrapper: internally scale a, v, sv by s.
    a_scaled = a / s
    v_scaled = v / s
    sv_scaled = sv / s

    # Match the R wrapper: documentation exposes t0 as lower bound, C engine expects midpoint.
    t0_mid = t0 + 0.5 * st0

    if z_absolute:
        z_rel = z / a
        sz_rel = sz / a
    else:
        z_rel = z.copy()
        sz_rel = sz.copy()

    _validate_params(a_scaled, v_scaled, t0_mid, z_rel, d, sz_rel, sv_scaled, st0, tau, lambda_)

    rt = np.empty(n, dtype=float)
    response = np.empty(n, dtype=int)
    conf = np.empty(n, dtype=float)
    evidence_term = np.empty(n, dtype=float)
    visibility_term = np.empty(n, dtype=float)
    drift_draw = np.empty(n, dtype=float)

    for i in range(n):
        mu = rng.normal(v_scaled[i], sv_scaled[i])
        drift_draw[i] = mu

        x = a_scaled[i] * rng.uniform(z_rel[i] - sz_rel[i] / 2.0, z_rel[i] + sz_rel[i] / 2.0)
        t = 0.0

        while (0.0 < x < a_scaled[i]) and (t < maxrt):
            x = x + rng.normal(delta * mu, np.sqrt(delta))
            t += delta

        if x >= a_scaled[i]:
            resp = 1
        elif x <= 0.0:
            resp = -1
        else:
            resp = 0

        response[i] = resp

        if tau[i] > 0:
            ev = resp * (x + rng.normal(tau[i] * mu, np.sqrt(tau[i])) - a_scaled[i] * z_rel[i])
        else:
            ev = resp * (x - a_scaled[i] * z_rel[i])
        evidence_term[i] = ev

        vis = rng.normal(0.0, np.sqrt((tau[i] + t) + (tau[i] + t) ** 2))
        visibility_term[i] = vis

        if lambda_[i] > 0:
            conf_val = ev / ((t + tau[i]) ** lambda_[i])
        else:
            conf_val = ev
        conf[i] = conf_val

        ndt = rng.uniform(t0_mid[i] - st0[i] / 2.0, t0_mid[i] + st0[i] / 2.0)
        rt[i] = max(0.0, t - resp * d[i] / 2.0) + ndt

    return TwoDSDResult(
        rt=rt,
        response=response,
        conf=conf,
        evidence_term=evidence_term,
        visibility_term=visibility_term,
        drift_draw=drift_draw,
    )


def simulate_trials(
    stimulus: ArrayLike,
    *,
    a: float,
    slope: float = 1.0,
    intercept: float = 0.0,
    t0: float = 0.0,
    z: float = 0.5,
    d: float = 0.0,
    sz: float = 0.0,
    sv: float = 0.0,
    st0: float = 0.0,
    tau: float = 1.0,
    lambda_: float = 0.0,
    s: float = 1.0,
    delta: float = 0.01,
    maxrt: float = 15.0,
    z_absolute: bool = False,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Convenience wrapper for psychophysics-style simulation.

    Parameters
    ----------
    stimulus:
        Signed stimulus values. These are mapped to drift by ``v = intercept + slope * stimulus``.
    """
    stim = np.asarray(stimulus, dtype=float).reshape(-1)
    drift = intercept + slope * stim
    res = r2dsd(
        len(stim),
        a=a,
        v=drift,
        t0=t0,
        z=z,
        d=d,
        sz=sz,
        sv=sv,
        st0=st0,
        tau=tau,
        lambda_=lambda_,
        s=s,
        delta=delta,
        maxrt=maxrt,
        z_absolute=z_absolute,
        seed=seed,
        rng=rng,
    )
    df = res.to_frame()
    df.insert(0, "stimulus", stim)
    # upper response (1) is correct for positive stimulus, lower (-1) for negative stimulus
    correct = np.where(stim > 0, res.response == 1, np.where(stim < 0, res.response == -1, np.nan))
    df["correct"] = correct.astype(float)
    return df
