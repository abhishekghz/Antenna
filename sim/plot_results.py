"""Plot the openEMS S11 result.

Reference markers to overlay (e.g. values from another solver) are read from an
optional local file `paper_markers.csv`, one `frequency_GHz,S11_dB` per line.
The file is deliberately not tracked, so unpublished data stays out of the
repository; without it the script simply plots the simulated response.
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.loadtxt('s11.csv', delimiter=',')
f, s11, rz, iz = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
import os
paper = []
if os.path.exists('paper_markers.csv'):
    paper = [tuple(r) for r in np.loadtxt('paper_markers.csv', delimiter=',', ndmin=2)]

fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                       gridspec_kw={'height_ratios': [2, 1]})
ax[0].plot(f, s11, lw=1.8, color='#1f4e79', label='openEMS (this reconstruction)')
for fp, sp in paper:
    ax[0].plot(fp, sp, 'v', ms=9, color='#c00000')
if paper:
    ax[0].plot([], [], 'v', color='#c00000', label='reference markers')
ax[0].axhline(-10, ls='--', lw=1, color='0.5')
ax[0].text(f[-1], -10.4, '-10 dB', ha='right', va='top', fontsize=8, color='0.4')
ax[0].set_ylabel('$|S_{11}|$  [dB]'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=9)
ax[0].set_title('Reflection coefficient — independent FDTD re-simulation')

ax[1].plot(f, rz, lw=1.5, color='k', label='Re $Z_{in}$')
ax[1].plot(f, iz, lw=1.5, ls='--', color='#c00000', label='Im $Z_{in}$')
ax[1].axhline(50, ls=':', lw=1, color='0.5')
ax[1].set_ylim(-200, 300); ax[1].set_xlabel('frequency [GHz]')
ax[1].set_ylabel('$Z_{in}$  [$\\Omega$]'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=9)

fig.tight_layout(); fig.savefig('s11_comparison.png', dpi=140)
print('wrote s11_comparison.png')

def bands(f, s, thr=-10.0):
    """-10 dB bands with interpolated edges.

    Using the nearest sample instead biases the bandwidth by up to two
    frequency steps, which here is the same order as the bandwidths themselves.
    """
    out, i = [], 0
    while i < len(f) - 1:
        if (s[i] - thr) * (s[i + 1] - thr) < 0 and s[i + 1] < s[i]:
            lo = f[i] + (f[i + 1] - f[i]) * (thr - s[i]) / (s[i + 1] - s[i])
            for j in range(i + 1, len(f) - 1):
                if (s[j] - thr) * (s[j + 1] - thr) < 0 and s[j + 1] > s[j]:
                    hi = f[j] + (f[j + 1] - f[j]) * (thr - s[j]) / (s[j + 1] - s[j])
                    k = i + int(np.argmin(s[i:j + 2]))
                    out.append((f[k], s[k], lo, hi))
                    i = j
                    break
            else:
                break
        i += 1
    return out


bb = bands(f, s11)
if bb:
    print('\n-10 dB bands (interpolated edges):')
    for c, dep, lo, hi in bb:
        print('  %5.2f - %5.2f GHz   min %6.2f dB at %5.2f GHz   BW %5.1f MHz (%.2f%%)'
              % (lo, hi, dep, c, (hi - lo) * 1000, 100 * (hi - lo) / c))
else:
    print('\nno -10 dB band found')
