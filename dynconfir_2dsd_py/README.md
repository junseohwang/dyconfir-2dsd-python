# dynconfir-2dsd-py

Python port of the **`dynConfiR::r2DSD()`** simulator path for the 2DSD model setting.

This package is intended to reproduce the **simulation behavior** of the R package implementation used by `dynConfiR` for:

- response
- RT
- continuous confidence

It ports the source path:

- `R/d2DSD.R` → `r2DSD()`
- `src/RNG_WEV.h` → `r_WEV(...)`

for the 2DSD special case used by the R wrapper (`w=1`, `muvis=0`, `sigvis=1`, `svis=1`).

## Scope

This repository currently implements the **simulator** only. It does **not** yet port the full `d2DSD()` likelihood.

## Installation

```bash
pip install -e .
```

## Example

```python
import numpy as np
from dynconfir_2dsd import simulate_trials

stim = np.repeat(np.array([-0.4, -0.2, 0.2, 0.4]), 200)
df = simulate_trials(
    stimulus=stim,
    a=2.0,
    slope=1.0,
    t0=0.2,
    z=0.5,
    sv=0.5,
    sz=0.1,
    tau=1.0,
    lambda_=0.0,
    seed=123,
)
print(df.head())
```

## Notes on parameterization

This package follows the **user-facing** `dynConfiR` parameterization:

- `t0` is the **lower bound** of the uniform NDT distribution of width `st0`.
- `z` and `sz` are **relative to `a`** unless `z_absolute=True`.
- `s` rescales `a`, `v`, and `sv` internally, just as in the R wrapper.

## Outputs

`r2dsd()` returns a `TwoDSDResult` with:

- `rt`
- `response`
- `conf`
- `evidence_term`
- `visibility_term`
- `drift_draw`

`simulate_trials()` returns a pandas `DataFrame` ready for downstream psychometric analysis.

## License

This port is derived from GPL-licensed source in the `dynConfiR` project. If you distribute derivative code, keep it under a GPL-compatible license and retain attribution to the original authors.
