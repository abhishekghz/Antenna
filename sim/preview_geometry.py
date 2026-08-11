"""Draw the reconstructed antenna layout (top and bottom face) for visual
comparison against Figure 1 of the paper."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

A, H = 12.2, 1.5
def ring(cx, cy, Rout, w, gap, ang, n=0):
    Rm = Rout - w/2
    d = np.degrees(np.arcsin(min(0.999, (gap/2)/Rm)))
    th = np.radians(np.linspace(ang+d, ang+360-d, 400))
    if n:
        pr = lambda R, t: R*np.cos(np.pi/n)/np.cos((t+np.pi/n) % (2*np.pi/n) - np.pi/n)
        ro, ri = pr(Rout, th), pr(Rout - w/np.cos(np.pi/n), th)
    else:
        ro, ri = np.full_like(th, Rout), np.full_like(th, Rout-w)
    return np.c_[np.r_[cx+ro*np.cos(th), cx+ri[::-1]*np.cos(th[::-1])],
                 np.r_[cy+ro*np.sin(th), cy+ri[::-1]*np.sin(th[::-1])]]

petal = np.loadtxt('petal.csv', delimiter=',')
fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
for a, t in zip(ax, ['top face (radiator)', 'bottom face (partial ground + SRR pair)']):
    a.add_patch(plt.Rectangle((-A/2, -A/2), A, A, fc='#f2ead9', ec='0.4', lw=1))
    a.set_xlim(-7, 7); a.set_ylim(-7, 7); a.set_aspect(1); a.set_title(t, fontsize=10)
    a.set_xlabel('x [mm]'); a.set_ylabel('y [mm]')

ax[0].add_patch(Polygon(ring(0, 0, 4.96, 0.60, 1.00, 180, 0), fc='k'))
ax[0].add_patch(Polygon(ring(0, 0, 3.55, 0.57, 1.00,   0, 6), fc='k'))
ax[0].add_patch(Polygon(petal, fc='k'))
ax[0].add_patch(plt.Rectangle((-0.5, -A/2), 1.0, 1.40, fc='k'))

ax[1].add_patch(plt.Rectangle((-A/2, -A/2), A, 6.10, fc='k'))
for s in (-1, 1):
    ax[1].add_patch(Polygon(ring(s*3.0, 2.9, 2.53, 0.55, 0.80,  90), fc='k'))
    ax[1].add_patch(Polygon(ring(s*3.0, 2.9, 1.53, 0.55, 0.35, 270), fc='k'))

fig.suptitle('Reconstructed geometry from Figure 1  (FR4, $\\epsilon_r$=4.4, h=1.5 mm, 12.2 $\\times$ 12.2 mm)',
             fontsize=11)
fig.tight_layout(); fig.savefig('geometry_reconstruction.png', dpi=140)
print('wrote geometry_reconstruction.png')
