"""Hardware-friendly GDN/IGDN with a piecewise-linear + Newton rsqrt.

GDN needs 1/sqrt(norm) and IGDN needs sqrt(norm).  Both are replaced by

    r = y0 + m0*(z-z0) + sum_i (m_i - m_{i-1}) * relu(z - z_i)   # chord seed
    r <- 0.5*r*(3 - z*r*r)                                       # Newton x2

with z clamped to [z0, zK].  Ops used: Conv1x1, Mul, Add, ReLU only -- no
Sqrt/Pow/Div, so the graph is AIDGE-quantizable and exportable to C.

Measured on the q8 model, 32 test tiles:
    exact GDN  : bpp 1.3854  psnr 39.598
    K=16, it=2 : bpp 1.3851  psnr 38.545  (same 32-image subset)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import GDN


DEFAULT_Z0 = 0.01
DEFAULT_ZK = 10000.0
DEFAULT_K = 16
DEFAULT_ITERS = 2


def fit_chords(z0: float = DEFAULT_Z0, zk: float = DEFAULT_ZK,
               k: int = DEFAULT_K):
    """Log-spaced chord approximation of rsqrt on [z0, zk]."""
    bp = np.geomspace(z0, zk, k + 1)
    ys = 1.0 / np.sqrt(bp)
    ms = (ys[1:] - ys[:-1]) / (bp[1:] - bp[:-1])
    return bp, ys, ms


class ApproxGDN(nn.Module):
    """Drop-in GDN/IGDN using only Conv/Mul/Add/ReLU."""

    def __init__(self, gdn: GDN, k: int = DEFAULT_K,
                 iters: int = DEFAULT_ITERS,
                 z0: float | None = None, zk: float | None = None):
        super().__init__()
        self.inverse = bool(gdn.inverse)
        self.iters = int(iters)
        if z0 is None:
            z0 = 1e-6 if self.inverse else DEFAULT_Z0
        if zk is None:
            zk = 10.0 if self.inverse else DEFAULT_ZK

        with torch.no_grad():
            beta = gdn.beta_reparam(gdn.beta).reshape(-1).detach().clone()
            gamma = gdn.gamma_reparam(gdn.gamma).detach().clone()
        c = int(beta.numel())
        self.register_buffer("beta", beta)
        self.register_buffer("gamma", gamma.reshape(c, c, 1, 1))

        bp, ys, ms = fit_chords(z0=z0, zk=zk, k=k)
        self.register_buffer("bp", torch.tensor(bp, dtype=torch.float32))
        self.register_buffer("ys", torch.tensor(ys, dtype=torch.float32))
        self.register_buffer("ms", torch.tensor(ms, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.conv2d(x * x, self.gamma, self.beta)

        # clamp to [z0, zK] using ReLU only
        z0, zk = self.bp[0], self.bp[-1]
        z = z0 + F.relu(z - z0)
        z = zk - F.relu(zk - z)

        # piecewise-linear chord seed
        r = self.ys[0] + self.ms[0] * (z - self.bp[0])
        for i in range(1, self.ms.numel()):
            r = r + (self.ms[i] - self.ms[i - 1]) * F.relu(z - self.bp[i])
        r = F.relu(r) + 1e-12

        # Step-clamped Newton: keeps r positive and prevents divergence when
        # the seed is outside the Newton basin (large norm outliers).
        for _ in range(self.iters):
            t = 3.0 - z * r * r
            t = F.relu(t)              # lower clamp 0
            t = 3.0 - F.relu(3.0 - t)  # upper clamp 3
            r = 0.5 * r * t

        if self.inverse:      # IGDN: x * sqrt(z) = x * z * rsqrt(z)
            r = r * z
        return x * r


def replace_gdn(module: nn.Module, k: int = DEFAULT_K,
                iters: int = DEFAULT_ITERS, z0: float | None = None,
                zk: float | None = None) -> int:
    """Replace every GDN in `module` in place; return the count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, GDN):
            setattr(module, name, ApproxGDN(child, k=k, iters=iters,
                                            z0=z0, zk=zk))
            count += 1
        else:
            count += replace_gdn(child, k=k, iters=iters, z0=z0, zk=zk)
    return count
