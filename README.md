# Petal-loaded hexagonal metamaterial patch antenna

Independent re-simulation and machine-learning surrogate for a 12.2 × 12.2 mm
tri-band microstrip patch antenna on FR4, intended for vehicular (V2X) links.

The original study was done in ANSYS HFSS. Everything here runs on free
software — **openEMS** for the electromagnetics, driven from MATLAB or Octave,
and PyTorch for the surrogate — so the results can be reproduced without a
commercial licence.

## How the work flows

```
   Figure 1 of the paper
            |
            v
  [1] geometry reconstruction        pixel measurement of the published
      GEOMETRY_NOTES.md              drawing -> dimensions + petal.csv
            |
            v
  [2] full-wave model                antenna_model.m (openEMS / FDTD)
            |
            +--> [3] toolchain validation      validate_ref_patch.m
            |         known 2.40 GHz patch -> 2.436 GHz  (solver is sane)
            |
            +--> [4] mesh convergence          0.10 mm -> 0.05 mm
            |         confirms which resonances are real
            |
            v
  [5] design sweep                   ml/make_designs.py -> ml/run_sweep.m
      300 geometries sampled over 6 dimensions, each solved full-wave
            |
            v
  [6] surrogate training             ml/train_surrogate.py
      dimensions -> full S11 curve, via a PCA-compressed MLP
            |
            v
  [7] inverse design                 ml/inverse_design.py
      "resonate at 5.9 GHz" -> dimensions, by gradient descent
            |
            v
  [8] full-wave verification         re-simulate the proposal and compare
```

Steps 1–4 answer *does the published antenna behave as claimed?*
Steps 5–8 answer *what geometry would behave the way I want?*

## Layout

| Path | What it is |
|---|---|
| `sim/antenna_model.m` | the antenna model — geometry, mesh, excitation, S11 |
| `sim/ringpoly.m`, `envopt.m` | helpers: split-ring polygons, environment overrides |
| `sim/petal.csv` | petal outline traced from Figure 1 |
| `sim/validate_ref_patch.m` | solver sanity check against a known result |
| `sim/GEOMETRY_NOTES.md` | every measurement, assumption and discrepancy found |
| `sim/REVISION_PLAN.md` | prioritised list of what the paper needs before submission |
| `sim/s11*.csv`, `sim/*.png` | results and figures |
| `sim/ml/` | the surrogate pipeline (sampling, sweep, training, inversion) |

## Running it

Install openEMS from <https://openems.de>, point the two `addpath` lines at the
top of `sim/antenna_model.m` at your install, then from `sim/`:

```matlab
validate_ref_patch     % ~1 min, expect 2.436 GHz
antenna_model          % ~7 min, writes s11.csv and prints the resonances
```

Discretisation can be changed without editing the file:

```matlab
setenv('ANT_RES','0.05'); setenv('ANT_OUT','s11_fine.csv'); antenna_model
```

For the surrogate, from `sim/ml/`:

```bash
python3 make_designs.py 300     # sample the design space
octave -q run_sweep.m           # solve them (shardable, resumable)
python3 train_surrogate.py      # train
python3 inverse_design.py 5.9   # ask for a geometry that resonates at 5.9 GHz
```

## Results so far

Two independent solvers agree on the antenna's upper two bands:

| Band | Paper (HFSS) | openEMS, converged |
|---|---|---|
| 1 | 5.74 GHz | 5.18 GHz |
| 2 | 10.10 GHz | 10.00 GHz |
| 3 | 11.45 GHz | 11.44 GHz |

Bands 2 and 3 match to within 1%. Band 1 remains about 10% low and is still
moving with mesh refinement. Measured −10 dB bandwidths are 84 / 241 / 130 MHz
(1.6 / 2.4 / 1.1 %) — narrow, and not reported in the paper.

See `sim/REVISION_PLAN.md` for what this implies for the manuscript.

## Caveats

- The geometry was reconstructed from a published figure, not from the authors'
  model. `sim/GEOMETRY_NOTES.md` lists what was measured and what was assumed.
- The surrogate training set is generated on a deliberately coarse mesh, which
  sits 5–12% below the converged reference (`sim/ml/calibrate.py` quantifies
  this). The surrogate is a design-space navigator; anything it proposes must
  be confirmed with a full-wave run at reference fidelity.
- No fabricated prototype or measurement exists. Every number here is simulated.
