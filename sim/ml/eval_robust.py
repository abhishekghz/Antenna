"""Repeated-split evaluation of the surrogate.

The headline metrics in train_surrogate.py come from a single split, and the
model's design (weighted loss, ensembling, Fourier width) was chosen while
those test numbers were visible. That biases them optimistically. This script
retrains from scratch on several independent splits and reports the spread, so
the quoted accuracy carries an honest error bar rather than a best case.

Architecture and hyper-parameters are frozen here; only the split changes.
"""
import glob
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)          # leave cores for a concurrent solver run

NSPLIT, NENS, NFOURIER = 5, 5, 32
FLO, FHI, NF = 3.0, 13.0, 201
freq = np.linspace(FLO, FHI, NF)

D = np.vstack([np.loadtxt(f, delimiter=',', ndmin=2)
               for f in sorted(glob.glob('sweep_s11_*.csv'))])
D = D[np.argsort(D[:, 0])]
_, u = np.unique(D[:, 0], return_index=True)
D = D[u]
X, Y = D[:, 1:7], np.clip(D[:, 7:], -45.0, 0.0)
ndes = len(X)

fn = (freq - FLO) / (FHI - FLO)
kk = np.arange(1, NFOURIER + 1)[:, None]
FF = np.concatenate([fn[None, :], np.sin(2 * np.pi * kk * fn), np.cos(2 * np.pi * kk * fn)], 0).T


class Net(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nin, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                               nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.f(x)


def resonances(s, kmax=4):
    out = [(s[i], freq[i]) for i in range(1, NF - 1)
           if s[i] < -10 and s[i] <= s[i - 1] and s[i] <= s[i + 1]]
    out.sort()
    return sorted(f for _, f in out[:kmax])


res = []
for sp in range(NSPLIT):
    rng = np.random.default_rng(1000 + sp)
    perm = rng.permutation(ndes)
    nt = nv = int(0.15 * ndes)
    itest, ival, itrain = perm[:nt], perm[nt:nt + nv], perm[nt + nv:]

    xmu, xsd = X[itrain].mean(0), X[itrain].std(0) + 1e-9
    ymu, ysd = Y[itrain].mean(), Y[itrain].std()

    def build(idx):
        a = np.repeat((X[idx] - xmu) / xsd, NF, axis=0)
        b = np.tile(FF, (len(idx), 1))
        yr = Y[idx].reshape(-1, 1)
        w = 1.0 + 6.0 / (1.0 + np.exp((yr + 5.0) / 1.5))
        return (torch.tensor(np.hstack([a, b]), dtype=torch.float32),
                torch.tensor((yr - ymu) / ysd, dtype=torch.float32),
                torch.tensor(w, dtype=torch.float32))

    xtr, ytr, wtr = build(itrain)
    xva, yva, wva = build(ival)
    nets = []
    for m in range(NENS):
        torch.manual_seed(7 + 100 * m + 13 * sp)
        net = Net(xtr.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
        best, bs, bad = 1e9, None, 0
        for ep in range(400):
            net.train()
            p = torch.randperm(len(xtr))
            for i in range(0, len(p), 1024):
                j = p[i:i + 1024]
                opt.zero_grad()
                (wtr[j] * (net(xtr[j]) - ytr[j]) ** 2).mean().backward()
                opt.step()
            sch.step()
            net.eval()
            with torch.no_grad():
                vl = (wva * (net(xva) - yva) ** 2).mean().item()
            if vl < best - 1e-6:
                best, bs, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
            if bad > 80:
                break
        net.load_state_dict(bs); net.eval(); nets.append(net)

    xte = build(itest)[0]
    with torch.no_grad():
        P = np.mean([n(xte).numpy() for n in nets], 0).reshape(len(itest), NF) * ysd + ymu

    mae = np.abs(P - Y[itest]).mean()
    ft, fp, spur, tot = [], [], 0, 0
    for r, i in enumerate(itest):
        a, b = resonances(Y[i]), resonances(P[r])
        tot += len(a); used = set()
        for fa in a:
            if not b:
                continue
            j = int(np.argmin([abs(x - fa) for x in b]))
            if abs(b[j] - fa) < 0.6 and j not in used:
                ft.append(fa); fp.append(b[j]); used.add(j)
        spur += max(0, len(b) - len(used))
    ft, fp = np.array(ft), np.array(fp)
    rec = 100 * len(ft) / tot
    fmae = np.abs(fp - ft).mean() if len(ft) else np.nan
    res.append((mae, rec, fmae, spur / len(itest)))
    print('split %d: curve MAE %.2f dB | recovery %.0f%% | freq MAE %.0f MHz | spurious/design %.2f'
          % (sp + 1, mae, rec, 1000 * fmae, spur / len(itest)))

R = np.array(res)
print('\nover %d independent splits (mean +/- sd):' % NSPLIT)
for name, col, fmt in [('curve MAE', 0, '%.2f dB'), ('band recovery', 1, '%.0f%%'),
                       ('matched freq MAE', 2, '%.0f MHz'), ('spurious/design', 3, '%.2f')]:
    v = R[:, col] * (1000 if col == 2 else 1)
    print(('  %-18s ' + fmt + '  +/- ' + fmt.replace('%.0f', '%.0f').replace('%.2f', '%.2f'))
          % (name, v.mean(), v.std()))
