import numpy as np

from dynconfir_2dsd import r2dsd, simulate_trials


def test_r2dsd_shapes():
    out = r2dsd(128, a=2.0, v=0.5, seed=1)
    assert out.rt.shape == (128,)
    assert out.response.shape == (128,)
    assert out.conf.shape == (128,)
    assert set(np.unique(out.response)).issubset({-1, 0, 1})


def test_simulate_trials_dataframe():
    stim = np.array([-0.4, -0.2, 0.2, 0.4] * 10)
    df = simulate_trials(stim, a=2.0, slope=1.0, seed=2)
    assert len(df) == len(stim)
    assert {"stimulus", "rt", "response", "conf", "correct"}.issubset(df.columns)
