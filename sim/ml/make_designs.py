"""Latin-hypercube sample of the 6-dimensional design space.

Only geometrically valid designs are kept: the rings must not overlap each
other, the petal must fit inside the hexagonal ring, and the outer ring must
fit on the board. Rejected samples are replaced so the final set is the
requested size.

Ground height is quantised to the FDTD grid (0.2 mm) so that the ground edge
always lands exactly on a mesh line — otherwise the label would not correspond
to the geometry actually simulated.
"""
import numpy as np, csv, sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
APPEND = '--append' in sys.argv      # keep existing rows, top up to N
SEED = 20260812 if not APPEND else 20260813
RES = 0.2                      # FDTD in-plane cell size used for the sweep
A = 12.2                       # substrate side
PETAL_R = 2.04                 # max radius of the traced petal at scale 1.0

# name, low, high, nominal (the published design)
VARS = [
    ('out_R',       4.20, 5.60, 4.96),   # outer split ring, outer radius
    ('ring_w',      0.40, 1.00, 0.60),   # trace width, both top-face rings
    ('hex_R',       2.60, 4.30, 3.55),   # hexagonal ring circumradius
    ('gap_l1',      0.40, 2.00, 1.00),   # split width of both top rings
    ('petal_s',     0.60, 1.40, 1.00),   # petal scale factor
    ('gnd_h',       3.60, 9.00, 6.10),   # partial ground height
]


def valid(d):
    out_R, w, hex_R, gap, ps, gh = d
    if out_R + 0.3 > A / 2:                    # ring must fit on the board
        return False
    if hex_R + w > out_R - w - 0.25:           # rings must not touch
        return False
    if PETAL_R * ps > hex_R - w - 0.20:        # petal must fit inside hexagon
        return False
    if gap > 2 * np.pi * (hex_R - w / 2) / 3:  # split cannot swallow the ring
        return False
    return True


def lhs(n, k, rng):
    """Latin hypercube on the unit cube.

    Each column gets its own independent permutation -- shuffling the array as
    a whole would move every row as a unit and leave the columns perfectly
    rank-correlated, collapsing the sample onto a diagonal of the cube.
    """
    u = np.empty((n, k))
    for j in range(k):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


existing = None
if APPEND:
    import os
    if os.path.exists('designs.csv'):
        existing = np.loadtxt('designs.csv', delimiter=',', skiprows=1, ndmin=2)
        print('keeping %d already-simulated designs' % len(existing))

rng = np.random.default_rng(SEED)
lo = np.array([v[1] for v in VARS])
hi = np.array([v[2] for v in VARS])

kept = []
tries = 0
while len(kept) < N and tries < 200:
    u = lhs(max(N, 4 * (N - len(kept))), len(VARS), rng)
    cand = lo + u * (hi - lo)
    cand[:, 5] = np.round(cand[:, 5] / RES) * RES        # snap ground to grid
    for d in cand:
        if valid(d):
            kept.append(d)
            if len(kept) == N:
                break
    tries += 1

D = np.array(kept[:N])
if existing is not None:
    # Existing rows keep their indices: run_sweep.m addresses designs by row
    # number and skips those already present in the results.
    need = max(0, N - len(existing))
    D = np.vstack([existing, D[:need]])
else:
    nom = np.array([v[3] for v in VARS]); nom[5] = round(nom[5] / RES) * RES
    D = np.vstack([nom, D[:N - 1]])

with open('designs.csv', 'w', newline='') as fh:
    wcsv = csv.writer(fh)
    wcsv.writerow([v[0] for v in VARS])
    wcsv.writerows(np.round(D, 4))

print('wrote designs.csv:', D.shape)
print('acceptance after filtering: %d designs' % len(D))
for i, v in enumerate(VARS):
    print('  %-8s %.2f .. %.2f   (nominal %.2f)' % (v[0], D[:, i].min(), D[:, i].max(), v[3]))
