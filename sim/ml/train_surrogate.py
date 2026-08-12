"""Train a neural surrogate for the antenna's reflection response.

The network maps (6 design dimensions, frequency) -> S11 at that frequency,
rather than dimensions -> the whole 201-point curve.

That choice matters. The obvious formulation -- compress the curve with PCA and
regress the coefficients -- fails badly here, and the dataset shows why:

  * The resonances are narrow and they *slide* with geometry. A moving sharp
    feature is not a low-rank phenomenon, so PCA needs ~60 components for 98.6%
    of the variance instead of the handful one might expect.
  * The number of resonances is not constant over the design space: of 300
    designs, 12 have none, 87 have one, 119 two, 56 three, and 26 have four or
    more. There is no fixed-length target that describes every design.
  * Only 4.8% of each curve lies below -10 dB, so a model that predicts "flat"
    scores well on curve error while being useless for design.

Treating frequency as an input solves all three: arbitrary band structure is
representable, every design contributes 201 training rows instead of one, and
Fourier features give the network the spatial frequency it needs to place sharp
nulls. The model stays differentiable in the design variables, so it can still
be inverted (see inverse_design.py).

Outputs
  surrogate.pt          weights + normalisation constants
  surrogate_fit.png     predicted vs simulated curves, held-out designs
  surrogate_parity.png  parity plot of held-out resonant frequencies
"""
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 7
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

NAMES = ['out_R', 'ring_w', 'hex_R', 'gap_l1', 'petal_s', 'gnd_h']
FLO, FHI, NF = 3.0, 13.0, 201
NFOURIER = 48
freq = np.linspace(FLO, FHI, NF)

# ---------------------------------------------------------------- load data
D = np.vstack([np.loadtxt(f, delimiter=',', ndmin=2)
               for f in sorted(glob.glob('sweep_s11_*.csv'))])
D = D[np.argsort(D[:, 0])]
_, uniq = np.unique(D[:, 0], return_index=True)
D = D[uniq]
X, Y = D[:, 1:7], np.clip(D[:, 7:], -45.0, 0.0)
ndes = len(X)
print('dataset: %d designs x %d frequencies = %d samples' % (ndes, NF, ndes * NF))

# Split by DESIGN, never by sample: rows from one design are near-duplicates,
# so a random row split would leak the answer into the test set.
perm = rng.permutation(ndes)
ntest = nval = int(0.15 * ndes)
itest, ival, itrain = perm[:ntest], perm[ntest:ntest + nval], perm[ntest + nval:]
print('split by design: %d train / %d val / %d test' % (len(itrain), len(ival), len(itest)))

xmu, xsd = X[itrain].mean(0), X[itrain].std(0) + 1e-9
ymu, ysd = Y[itrain].mean(), Y[itrain].std()

fn = (freq - FLO) / (FHI - FLO)
k = np.arange(1, NFOURIER + 1)[:, None]
FF = np.concatenate([fn[None, :] * 0 + fn[None, :],
                     np.sin(2 * np.pi * k * fn),
                     np.cos(2 * np.pi * k * fn)], 0).T          # (NF, 1+2K)
print('frequency encoding: %d features' % FF.shape[1])


def weights(yv):
    # Plain MSE spends its capacity on the ~95% of each curve that sits near
    # 0 dB, and smooths away the narrow nulls that are the whole point. Weight
    # the matched region so the resonances actually drive the fit.
    return 1.0 + 6.0 / (1.0 + np.exp((yv + 5.0) / 1.5))


def build(idx):
    xn = (X[idx] - xmu) / xsd                                    # (n, 6)
    n = len(idx)
    a = np.repeat(xn, NF, axis=0)
    b = np.tile(FF, (n, 1))
    yraw = Y[idx].reshape(-1, 1)
    y = ((yraw - ymu) / ysd)
    w = weights(yraw)
    return (torch.tensor(np.hstack([a, b]), dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(w, dtype=torch.float32))


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


xtr, ytr, wtr = build(itrain)
xva, yva, wva = build(ival)
net = Net(xtr.shape[1])
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
EPOCHS, BS = 400, 1024
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

best, beststate, bad = 1e9, None, 0
for ep in range(EPOCHS):
    net.train()
    p = torch.randperm(len(xtr))
    for i in range(0, len(p), BS):
        j = p[i:i + BS]
        opt.zero_grad()
        (wtr[j] * (net(xtr[j]) - ytr[j]) ** 2).mean().backward()
        opt.step()
    sched.step()
    net.eval()
    with torch.no_grad():
        vl = (wva * (net(xva) - yva) ** 2).mean().item()
    if vl < best - 1e-6:
        best, beststate, bad = vl, {k_: v.clone() for k_, v in net.state_dict().items()}, 0
    else:
        bad += 1
    if ep % 40 == 0:
        print('  epoch %3d  val MSE %.4f  (val MAE %.2f dB)' % (ep, vl, np.sqrt(vl) * ysd))
    if bad > 80:
        print('  early stop at epoch %d' % ep)
        break
net.load_state_dict(beststate)
net.eval()


def predict(idx):
    x = build(idx)[0]
    with torch.no_grad():
        return (net(x).numpy().reshape(len(idx), NF) * ysd) + ymu


print()
for lab, idx in [('train', itrain), ('val', ival), ('test', itest)]:
    P = predict(idx)
    print('%-5s  curve MAE %5.2f dB   RMSE %5.2f dB' %
          (lab, np.abs(P - Y[idx]).mean(), np.sqrt(((P - Y[idx]) ** 2).mean())))


def resonances(s, kmax=4):
    out = [(s[i], freq[i]) for i in range(1, NF - 1)
           if s[i] < -10 and s[i] <= s[i - 1] and s[i] <= s[i + 1]]
    out.sort()
    return sorted(f for _, f in out[:kmax])


Pte = predict(itest)
ft, fp, missed, spurious = [], [], 0, 0
for r, i in enumerate(itest):
    a, b = resonances(Y[i]), resonances(Pte[r])
    used = set()
    for fa in a:
        if not b:
            missed += 1
            continue
        j = int(np.argmin([abs(fb - fa) for fb in b]))
        if abs(b[j] - fa) < 0.6 and j not in used:
            ft.append(fa); fp.append(b[j]); used.add(j)
        else:
            missed += 1
    spurious += max(0, len(b) - len(used))
ft, fp = np.array(ft), np.array(fp)
nband = sum(len(resonances(Y[i])) for i in itest)
print('\nheld-out band recovery: %d/%d matched within 600 MHz, %d missed, %d spurious'
      % (len(ft), nband, missed, spurious))
if len(ft):
    print('matched-band frequency MAE: %.3f GHz (%.1f%% of centre)'
          % (np.abs(fp - ft).mean(), 100 * np.abs(fp - ft).mean() / ft.mean()))

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)
for ax, r in zip(axes.ravel(), range(6)):
    i = itest[r]
    ax.plot(freq, Y[i], color='#1f4e79', lw=1.6, label='openEMS')
    ax.plot(freq, Pte[r], color='#c00000', lw=1.4, ls='--', label='surrogate')
    ax.axhline(-10, color='0.6', lw=.8, ls=':')
    ax.set_title(', '.join('%s=%.2f' % (n, v) for n, v in zip(NAMES, X[i]))[:58], fontsize=6.5)
    ax.grid(alpha=.3)
axes[0, 0].legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel('frequency [GHz]')
for ax in axes[:, 0]:
    ax.set_ylabel('$|S_{11}|$ [dB]')
fig.suptitle('Neural surrogate vs. full-wave simulation, held-out designs', fontsize=12)
fig.tight_layout(); fig.savefig('surrogate_fit.png', dpi=140)

if len(ft):
    fig2, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([FLO, FHI], [FLO, FHI], color='0.6', lw=1)
    ax.scatter(ft, fp, s=28, color='#1f4e79', alpha=.75)
    ax.set_xlabel('simulated resonance [GHz]'); ax.set_ylabel('predicted resonance [GHz]')
    ax.set_title('Held-out resonant frequencies'); ax.grid(alpha=.3); ax.set_aspect(1)
    fig2.tight_layout(); fig2.savefig('surrogate_parity.png', dpi=140)

torch.save({'state': net.state_dict(), 'xmu': xmu, 'xsd': xsd, 'ymu': ymu, 'ysd': ysd,
            'FF': FF, 'freq': freq, 'names': NAMES, 'nin': xtr.shape[1]}, 'surrogate.pt')
print('saved surrogate.pt')
