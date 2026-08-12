"""Quantify the fidelity offset of the sweep mesh.

The training set is generated at 0.2 mm / 20k timesteps so that 300 designs fit
in under an hour. That is coarser than the converged reference in ../s11_fine.csv
(0.05 mm / 80k timesteps). This script measures the resulting bias on the
nominal design -- which appears as row 1 of the sweep -- so the surrogate's
accuracy can be reported against a stated offset rather than an assumed one.
"""
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FLO, FHI, NF = 3.0, 13.0, 201
fsweep = np.linspace(FLO, FHI, NF)


def resonances(f, s, kmax=3):
    out = []
    for i in range(1, len(s) - 1):
        if s[i] < -10 and s[i] <= s[i - 1] and s[i] <= s[i + 1]:
            out.append((f[i], s[i]))
    out.sort(key=lambda t: t[1])
    return sorted(out[:kmax])


rows = [np.loadtxt(fn, delimiter=',', ndmin=2) for fn in sorted(glob.glob('sweep_s11_*.csv'))]
D = np.vstack([r for r in rows if r.size])
nom = D[D[:, 0] == 1]
if not len(nom):
    raise SystemExit('nominal design (row 1) not in the sweep yet')
s_sweep = nom[0, 7:]

ref = {}
for tag, fn in [('0.10 mm', '../s11.csv'), ('0.05 mm', '../s11_fine.csv')]:
    d = np.loadtxt(fn, delimiter=',')
    ref[tag] = (d[:, 0], d[:, 1])

print('Nominal design, resonances by mesh fidelity')
print('%-22s %s' % ('model', 'bands [GHz] (depth dB)'))
rs = resonances(fsweep, s_sweep)
print('%-22s %s' % ('sweep  0.2 mm/20k', '  '.join('%.2f (%.1f)' % r for r in rs)))
for tag, (f, s) in ref.items():
    rr = resonances(f, s)
    print('%-22s %s' % ('reference %s' % tag, '  '.join('%.2f (%.1f)' % r for r in rr)))

rr = resonances(*ref['0.05 mm'])
if rs and rr:
    k = min(len(rs), len(rr))
    d = [(rs[i][0] - rr[i][0]) / rr[i][0] * 100 for i in range(k)]
    print('\noffset of sweep mesh vs converged reference: ' +
          ', '.join('%+.1f%%' % x for x in d))
    print('mean absolute offset: %.1f%%' % np.mean(np.abs(d)))

fig, ax = plt.subplots(figsize=(9, 4.5))
for tag, (f, s) in ref.items():
    ax.plot(f, s, lw=1.4, label='reference %s' % tag)
ax.plot(fsweep, s_sweep, lw=2, color='#c00000', label='sweep mesh 0.2 mm / 20k')
ax.axhline(-10, ls=':', color='0.5')
ax.set_xlim(3, 13); ax.grid(alpha=.3); ax.legend(fontsize=9)
ax.set_xlabel('frequency [GHz]'); ax.set_ylabel('$|S_{11}|$ [dB]')
ax.set_title('Training-set fidelity vs. the converged reference (nominal design)')
fig.tight_layout(); fig.savefig('calibration.png', dpi=140)
print('wrote calibration.png')
