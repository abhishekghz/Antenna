"""Plot the openEMS S11 result and overlay the three resonances the paper reports."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.loadtxt('s11.csv', delimiter=',')
f, s11, rz, iz = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
paper = [(5.74, -25.0), (10.10, -28.1), (11.45, -35.1)]   # HFSS markers, Fig. 3(a)

fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                       gridspec_kw={'height_ratios': [2, 1]})
ax[0].plot(f, s11, lw=1.8, color='#1f4e79', label='openEMS (this reconstruction)')
for fp, sp in paper:
    ax[0].plot(fp, sp, 'v', ms=9, color='#c00000')
ax[0].plot([], [], 'v', color='#c00000', label='HFSS markers reported in the paper')
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

below = s11 < -10
if below.any():
    print('\n-10 dB bands:')
    edges = np.diff(below.astype(int))
    starts = list(np.where(edges == 1)[0] + 1); stops = list(np.where(edges == -1)[0])
    if below[0]: starts = [0] + starts
    if below[-1]: stops = stops + [len(f) - 1]
    for a, b in zip(starts, stops):
        k = a + int(np.argmin(s11[a:b + 1]))
        print('  %5.2f - %5.2f GHz   min %6.2f dB at %5.2f GHz   BW %4.0f MHz'
              % (f[a], f[b], s11[k], f[k], (f[b] - f[a]) * 1000))
else:
    print('\nno -10 dB band found')
