"""Inverse design: ask the surrogate for dimensions that resonate where you want.

Forward use of a surrogate ("what does this geometry do?") saves time. Inverse
use ("what geometry does what I need?") is the part worth publishing, because it
turns hours of parameter sweeping into seconds of gradient descent through a
differentiable model.

The default target is 5.9 GHz -- the DSRC / C-V2X allocation that the published
design misses.

Two corrections matter and are applied explicitly:

  * The surrogate is trained on the coarse sweep mesh, which sits ~5.5% below
    the converged reference at the lowest band (see calibrate.py). To land a
    resonance at 5.9 GHz in reality, the surrogate must be asked for one at
    about 5.58 GHz. Pass --corr 0 to disable.
  * Nothing here is believable until a full-wave run at reference fidelity
    confirms it. The script writes candidates.csv for exactly that.

Usage:
    python3 inverse_design.py                 # target 5.9 GHz
    python3 inverse_design.py 5.9 10.5        # two target bands
    python3 inverse_design.py 5.9 --corr 0    # no fidelity correction
"""
import sys
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

ck = torch.load('surrogate.pt', weights_only=False)
NAMES, freq, FF = ck['names'], ck['freq'], ck['FF']
NF = len(freq)

LO = np.array([4.20, 0.40, 2.60, 0.40, 0.60, 3.60])
HI = np.array([5.60, 1.00, 4.30, 2.00, 1.40, 9.00])
PETAL_R = 2.04


class Net(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(nin, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 1))

    def forward(self, x):
        return self.f(x)


nets = []
for st in ck['states']:
    n = Net(ck['nin']); n.load_state_dict(st); n.eval()
    for p_ in n.parameters():
        p_.requires_grad_(False)
    nets.append(n)

xmu = torch.tensor(ck['xmu'], dtype=torch.float32)
xsd = torch.tensor(ck['xsd'], dtype=torch.float32)
ff = torch.tensor(FF, dtype=torch.float32)
lo = torch.tensor(LO, dtype=torch.float32)
hi = torch.tensor(HI, dtype=torch.float32)


def s11_of(p, cols=None):
    """Differentiable S11, ensemble mean, for a batch of designs.

    cols selects which frequency samples to evaluate. The optimiser only looks
    at a handful of frequencies around the target, and evaluating all 201 there
    costs ~40x more for no benefit.
    """
    sel = ff if cols is None else ff[cols]
    m, b = sel.shape[0], p.shape[0]
    xn = (p - xmu) / xsd
    x = torch.cat([xn.repeat_interleave(m, 0), sel.repeat(b, 1)], 1)
    y = torch.stack([n(x) for n in nets]).mean(0)
    return y.reshape(b, m) * ck['ysd'] + ck['ymu']


def penalty(p):
    out_R, w, hex_R, gap, ps, _ = p.unbind(-1)
    r = torch.relu
    return (r(out_R + 0.3 - 6.1)
            + r(hex_R + w - (out_R - w - 0.25))
            + r(PETAL_R * ps - (hex_R - w - 0.20))
            + r(gap - 2 * np.pi * (hex_R - w / 2) / 3)) * 40.0


args = [a for a in sys.argv[1:]]
corr = 0.055
if '--corr' in args:
    i = args.index('--corr'); corr = float(args[i + 1]); del args[i:i + 2]
targets = [float(a) for a in args] or [5.9]
shifted = [t * (1 - corr) for t in targets]
tidx = [int(np.argmin(np.abs(freq - t))) for t in shifted]
print('physical targets : ' + ', '.join('%.2f GHz' % t for t in targets))
print('surrogate targets: ' + ', '.join('%.2f GHz' % freq[i] for i in tidx)
      + ('   (%.1f%% fidelity correction)' % (100 * corr) if corr else ''))

NSTART = 256
g = torch.Generator().manual_seed(3)
raw = torch.logit(torch.rand(NSTART, 6, generator=g).clamp(0.02, 0.98)).requires_grad_(True)
opt = torch.optim.Adam([raw], lr=0.05)

cols = sorted({c for i in tidx for c in range(max(0, i - 2), min(NF, i + 3))})
colmap = {c: j for j, c in enumerate(cols)}
colsel = torch.tensor(cols)

for step in range(400):
    opt.zero_grad()
    p = lo + (hi - lo) * torch.sigmoid(raw)
    s = s11_of(p, colsel)
    obj = 0
    for i in tidx:
        j = [colmap[c] for c in range(max(0, i - 2), min(NF, i + 3))]
        obj = obj + s[:, j].mean(-1)
    (obj + penalty(p)).sum().backward()
    opt.step()

with torch.no_grad():
    p = lo + (hi - lo) * torch.sigmoid(raw)
    s = s11_of(p)
    score = sum(s[:, i] for i in tidx) + penalty(p)
    order = torch.argsort(score)

print('\ntop candidates (surrogate prediction)')
print('  ' + '  '.join('%8s' % n for n in NAMES) + '  ' +
      '  '.join('S11@%.2f' % freq[i] for i in tidx))
best, seen = [], []
for r in order:
    row = p[r].numpy()
    if penalty(p[r]).item() > 1e-6:
        continue
    if any(np.linalg.norm((row - q) / (HI - LO)) < 0.12 for q in seen):
        continue          # keep the candidates genuinely distinct
    seen.append(row); best.append(row)
    print('  ' + '  '.join('%8.3f' % v for v in row) + '  ' +
          '  '.join('%8.2f' % s[r, i].item() for i in tidx))
    if len(best) == 6:
        break

if best:
    np.savetxt('candidates.csv', np.array(best), delimiter=',',
               header=','.join(NAMES), comments='', fmt='%.4f')
    print('\nwrote candidates.csv')
    print('verify before believing any of it:')
    print('  SWEEP_DESIGNS=candidates.csv SWEEP_TAG=verify SWEEP_RES=0.1 \\')
    print('  SWEEP_NRTS=40000 SWEEP_AIR=16 SWEEP_SUBZ=9 octave -q run_sweep.m')
