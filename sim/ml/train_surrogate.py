"""Train a neural surrogate: 6 antenna dimensions -> full S11(f) response.

The network predicts a PCA-compressed representation of the S11 curve rather
than all 201 frequency samples directly. With a few hundred training designs
that is the difference between a model that generalises and one that memorises:
the curves live on a low-dimensional manifold (a handful of resonances sliding
around), so ~20 components capture essentially all the variance while cutting
the output dimension by an order of magnitude.

Outputs
  surrogate.pt        trained weights + normalisation constants
  surrogate_fit.png   predicted vs simulated curves on held-out designs
  surrogate_parity.png parity plot of resonant frequency and depth
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
freq = np.linspace(FLO, FHI, NF)

# ---------------------------------------------------------------- load data
rows = []
for fn in sorted(glob.glob('sweep_s11_*.csv')):
    d = np.loadtxt(fn, delimiter=',', ndmin=2)
    if d.size:
        rows.append(d)
D = np.vstack(rows)
D = D[np.argsort(D[:, 0])]
_, uniq = np.unique(D[:, 0], return_index=True)
D = D[uniq]
X = D[:, 1:7]
Y = D[:, 7:]
print('dataset: %d designs, %d frequency samples' % X.shape[0:1] + (Y.shape[1],))

# Clip the extreme tail: values below -45 dB are numerically noisy nulls and
# would otherwise dominate the loss without carrying design information.
Y = np.clip(Y, -45.0, 0.0)

# ------------------------------------------------------------------ splits
n = len(X)
perm = rng.permutation(n)
ntest = max(12, int(0.15 * n))
nval = max(12, int(0.15 * n))
itest, ival, itrain = perm[:ntest], perm[ntest:ntest + nval], perm[ntest + nval:]
print('split: %d train / %d val / %d test' % (len(itrain), len(ival), len(itest)))

# --------------------------------------------------- normalisation and PCA
xmu, xsd = X[itrain].mean(0), X[itrain].std(0) + 1e-9
Xn = (X - xmu) / xsd

ymu = Y[itrain].mean(0)
Yc = Y - ymu
U, S, Vt = np.linalg.svd(Yc[itrain], full_matrices=False)
NC = int(np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.995) + 1)
NC = max(8, min(NC, 30))
basis = Vt[:NC]                       # (NC, NF)
Z = Yc @ basis.T
zsd = Z[itrain].std(0) + 1e-9
print('PCA: %d components, %.2f%% of variance'
      % (NC, 100 * np.sum(S[:NC]**2) / np.sum(S**2)))


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


def tt(a):
    return torch.tensor(a, dtype=torch.float32)


net = Net(6, NC)
opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000)
xtr, ztr = tt(Xn[itrain]), tt(Z[itrain] / zsd)
xva, zva = tt(Xn[ival]), tt(Z[ival] / zsd)

best, beststate, bad = 1e9, None, 0
for ep in range(4000):
    net.train(); opt.zero_grad()
    loss = nn.functional.mse_loss(net(xtr), ztr)
    loss.backward(); opt.step(); sched.step()
    if ep % 20 == 0:
        net.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(net(xva), zva).item()
        if vl < best - 1e-5:
            best, beststate, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 400 == 0:
            print('  epoch %4d  train %.4f  val %.4f' % (ep, loss.item(), vl))
        if bad > 40:
            print('  early stop at epoch %d' % ep)
            break
net.load_state_dict(beststate)
net.eval()


def predict(Xraw):
    with torch.no_grad():
        z = net(tt((Xraw - xmu) / xsd)).numpy() * zsd
    return z @ basis + ymu


# ------------------------------------------------------------- evaluation
Yp = predict(X)
err = np.abs(Yp - Y)
for lab, idx in [('train', itrain), ('val', ival), ('test', itest)]:
    print('%-5s  curve MAE %5.2f dB   RMSE %5.2f dB'
          % (lab, err[idx].mean(), np.sqrt(((Yp[idx] - Y[idx])**2).mean())))


def resonances(s11, kmax=3):
    """Return the kmax deepest local minima below -10 dB, sorted by frequency."""
    out = []
    for i in range(1, len(s11) - 1):
        if s11[i] < -10 and s11[i] <= s11[i - 1] and s11[i] <= s11[i + 1]:
            out.append((s11[i], freq[i]))
    out.sort()
    return sorted(f for _, f in out[:kmax])


ft, fp = [], []
for i in itest:
    a, b = resonances(Y[i]), resonances(Yp[i])
    for k in range(min(len(a), len(b))):
        ft.append(a[k]); fp.append(b[k])
ft, fp = np.array(ft), np.array(fp)
if len(ft):
    print('resonant frequency: MAE %.3f GHz on %d held-out bands (%.2f%% of centre)'
          % (np.abs(fp - ft).mean(), len(ft), 100 * np.abs(fp - ft).mean() / ft.mean()))

# ----------------------------------------------------------------- figures
show = itest[:6]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)
for ax, i in zip(axes.ravel(), show):
    ax.plot(freq, Y[i], color='#1f4e79', lw=1.6, label='openEMS')
    ax.plot(freq, Yp[i], color='#c00000', lw=1.4, ls='--', label='surrogate')
    ax.axhline(-10, color='0.6', lw=.8, ls=':')
    ax.set_title(', '.join('%s=%.2f' % (n, v) for n, v in zip(NAMES[:3], X[i][:3])), fontsize=7)
    ax.grid(alpha=.3)
axes[0, 0].legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel('frequency [GHz]')
for ax in axes[:, 0]:
    ax.set_ylabel('$|S_{11}|$ [dB]')
fig.suptitle('Neural surrogate vs. full-wave simulation on held-out designs', fontsize=12)
fig.tight_layout(); fig.savefig('surrogate_fit.png', dpi=140)

if len(ft):
    fig2, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([FLO, FHI], [FLO, FHI], color='0.6', lw=1)
    ax.scatter(ft, fp, s=28, color='#1f4e79', alpha=.75)
    ax.set_xlabel('simulated resonance [GHz]'); ax.set_ylabel('predicted resonance [GHz]')
    ax.set_title('Held-out resonant frequencies'); ax.grid(alpha=.3); ax.set_aspect(1)
    fig2.tight_layout(); fig2.savefig('surrogate_parity.png', dpi=140)

torch.save({'state': net.state_dict(), 'xmu': xmu, 'xsd': xsd, 'ymu': ymu,
            'basis': basis, 'zsd': zsd, 'nc': NC, 'freq': freq, 'names': NAMES},
           'surrogate.pt')
print('saved surrogate.pt')
