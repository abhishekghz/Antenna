# Independent re-simulation of the petal-loaded hexagonal patch antenna

This directory reproduces the antenna from *"Petal-Loaded Hexagonal Metamaterial
Patch Antenna for Connected Vehicles"* in an open-source solver, so the
published results can be checked without an ANSYS HFSS licence.

The original study used HFSS (frequency-domain FEM). This model uses
**openEMS** (time-domain FDTD) — a different numerical method, which makes it a
genuine cross-check rather than a repeat of the same computation. openEMS is
free, and it is driven entirely from **MATLAB or Octave** scripts, so it runs on
a MATLAB-only setup.

## Files

| File | Purpose |
|---|---|
| `antenna_model.m` | the model: geometry, mesh, excitation, solver run, S11 extraction |
| `ringpoly.m` | helper — builds a split-ring polygon (circular or hexagonal) |
| `petal.csv` | petal outline traced from Figure 1, 25-point polygon in mm |
| `preview_geometry.py` | draws the reconstructed layout for visual comparison with Figure 1 |
| `plot_results.py` | plots S11, with optional reference markers from a local file |
| `validate_ref_patch.m` | toolchain check — a patch with a known 2.4 GHz resonance |

## Running it

**On your machine (MATLAB, Windows or Linux):**

1. Install openEMS from <https://openems.de> (Windows: unpack the release zip;
   Linux: build from source or use your distribution's package).
2. In MATLAB, point the two `addpath` calls at the top of `antenna_model.m` to
   your install:
   ```matlab
   addpath('C:\openEMS\matlab');
   addpath('C:\openEMS\CSXCAD\matlab');
   ```
3. `cd` to this directory and run `antenna_model`.

Nothing else in the script is platform- or version-specific. Octave works
equally well and needs no licence at all.

## Method notes

- FR4 is modelled as `εr = 4.4` with conductivity fixed from `tanδ = 0.02` at the
  excitation centre frequency (7.5 GHz), i.e. a constant-conductivity
  approximation rather than a frequency-dependent loss tangent. HFSS's default
  constant-`tanδ` model differs slightly; the effect on resonant frequency is
  small, on `|S11|` depth less so.
- Conductors are zero-thickness PEC. Copper loss is therefore not included, so
  simulated resonances are, if anything, deeper than reality.
- The feed is a 50 Ω lumped port across the substrate thickness at the board
  edge. The paper does not state its port type; a different port reference
  changes `|S11|` depth substantially even when the resonant frequencies agree.
- Mesh: uniform 0.15 mm over the board, graded out to 1.2 mm in the air box,
  first-order Mur absorbing boundaries with ~16 mm (≈λ/4 at 5 GHz) of padding.

## Toolchain validation

`validate_ref_patch.m` runs the standard openEMS reference patch, whose
resonance is known to be ≈2.40 GHz. This build returns **2.436 GHz**, confirming
the solver, mesh settings, and S-parameter extraction are behaving before any
conclusion is drawn about the antenna under study.
