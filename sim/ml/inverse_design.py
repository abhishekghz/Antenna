"""Inverse design: ask the surrogate for dimensions that resonate where you want.

Forward use of a surrogate ("what does this geometry do?") is convenient.
Inverse use ("what geometry does what I need?") is the part that is actually
worth publishing, because it turns hours of parameter sweeping into seconds of
gradient descent through a differentiable model.

The default target is 5.9 GHz -- the DSRC / C-V2X allocation that the published
design misses.

Usage:
    python3 inverse_design.py            # target 5.9 GHz
    python3 inverse_design.py 5.9 10.5   # two target bands
"""
import sys
import numpy as np
import torch
import torch.nn as nn

ck = torch.load('surrogate.pt', weights_only=False)
NAMES = ck['names']
freq = ck['freq']
NC = ck['nc']

# design-space box, must match make_designs.py
LO = np.array([4.20, 0.40, 2.60, 0.40, 0.60, 3.60])
HI = np.array([5.60, 1.00, 4.30, 2.00, 1.40, 9.00])
PETAL_R = 2.04


class Net(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(nin, 128), nn.SiLU(),
            nn.Linear(128, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, nout))

    def forward(self, x):
        return self.f(x)


net = Net(6, NC)
net.load_state_dict(ck['state'])
net.eval()
for p in net.parameters():
    p.requires_grad_(False)

xmu = torch.tensor(ck['xmu'], dtype=torch.float32)
xsd = torch.tensor(ck['xsd'], dtype=torch.float32)
zsd = torch.tensor(ck['zsd'], dtype=torch.float32)
basis = torch.tensor(ck['basis'], dtype=torch.float32)
ymu = torch.tensor(ck['ymu'], dtype=torch.float32)
lo = torch.tensor(LO, dtype=torch.float32)
hi = torch.tensor(HI, dtype=torch.float32)


def s11_of(p):
    """Differentiable S11 curve for a batch of designs p (physical units)."""
    z = net((p - xmu) / xsd) * zsd
    return z @ basis + ymu


def penalty(p):
    """Soft version of the geometric validity constraints in make_designs.py."""
    out_R, w, hex_R, gap, ps, _ = p.unbind(-1)
    relu = torch.relu
    return (relu(out_R + 0.3 - 6.1)
            + relu(hex_R + w - (out_R - w - 0.25))
            + relu(PETAL_R * ps - (hex_R - w - 0.20))
            + relu(gap - 2 * np.pi * (hex_R - w / 2) / 3)) * 40.0


targets = [float(a) for a in sys.argv[1:]] or [5.9]
tidx = [int(np.argmin(np.abs(freq - t))) for t in targets]
print('targets: ' + ', '.join('%.2f GHz' % freq[i] for i in tidx))

NSTART = 512
g = torch.Generator().manual_seed(3)
u = torch.rand(NSTART, 6, generator=g)
raw = torch.logit(u.clamp(0.02, 0.98)).requires_grad_(True)
opt = torch.optim.Adam([raw], lr=0.05)

for step in range(900):
    opt.zero_grad()
    p = lo + (hi - lo) * torch.sigmoid(raw)
    s = s11_of(p)
    # push the response down at each target frequency, and a little either side
    # so the match is a band rather than a knife-edge null
    obj = 0
    for i in tidx:
        w = slice(max(0, i - 1), min(len(freq), i + 2))
        obj = obj + s[:, w].mean(-1)
    loss = (obj + penalty(p)).sum()
    loss.backward()
    opt.step()

with torch.no_grad():
    p = lo + (hi - lo) * torch.sigmoid(raw)
    s = s11_of(p)
    score = sum(s[:, i] for i in tidx) + penalty(p)
    order = torch.argsort(score)

print('\ntop candidate designs')
print('  ' + '  '.join('%8s' % n for n in NAMES) + '   ' +
      '  '.join('S11@%.1f' % freq[i] for i in tidx))
best = []
for r in order[:8]:
    row = p[r].numpy()
    if penalty(p[r]).item() > 1e-6:
        continue
    best.append(row)
    print('  ' + '  '.join('%8.3f' % v for v in row) + '   ' +
          '  '.join('%8.2f' % s[r, i].item() for i in tidx))

if best:
    np.savetxt('candidates.csv', np.array(best), delimiter=',',
               header=','.join(NAMES), comments='', fmt='%.4f')
    print('\nwrote candidates.csv -- verify with a full-wave run before believing it')
