"""Hardware-friendly surgery on CompressAI's bmshj2018 factorized model.

Two changes, both optional:

  * GDN -> PReLU / LeakyReLU / ReLU
  * selected 5x5 strided convs -> two 3x3 convs (3x3 s1 + 3x3 s2)

The decoder (g_s) is left untouched so the optimized encoder can be compared
against the original reference decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from compressai.layers import GDN
from compressai.models import FactorizedPrior

from approx_gdn import ApproxGDN


class FactorizedConv2d(nn.Module):
    """A 5x5/stride-2 conv expressed as two 3x3 convs."""

    def __init__(self, cin: int, cout: int, mid: int, stride: int = 2,
                 bias: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, mid, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(mid, cout, 3, stride=stride, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class ScaledPReLU(nn.Module):
    """Per-channel gain * PReLU: y = scale_c * (x if x>0 else slope_c*x).

    Initialized from GDN's small-signal gain 1/sqrt(beta_c) so the latent keeps
    its original scale when GDN is removed.
    """

    def __init__(self, channels: int, scale: torch.Tensor | None = None,
                 slope: float = 0.1):
        super().__init__()
        if scale is None:
            scale = torch.ones(channels)
        self.scale = nn.Parameter(scale.clone().view(1, channels, 1, 1))
        self.slope = nn.Parameter(torch.full((channels,), slope))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slope = self.slope.view(1, -1, 1, 1)
        return self.scale * torch.where(x > 0, x, slope * x)


def _activation(kind: str, channels: int, gdn: GDN | None = None) -> nn.Module:
    if kind == "prelu":
        return nn.PReLU(num_parameters=channels, init=0.1)
    if kind == "leaky":
        return nn.LeakyReLU(0.1, inplace=False)
    if kind == "relu":
        return nn.ReLU(inplace=False)
    if kind == "scaled_prelu":
        scale = None
        if gdn is not None:
            with torch.no_grad():
                beta = gdn.beta_reparam(gdn.beta).reshape(-1)
                scale = 1.0 / torch.sqrt(beta.clamp_min(1e-8))
        return ScaledPReLU(channels, scale=scale)
    raise ValueError(f"unknown activation {kind!r}")


def build_model(quality: int = 8, act: str = "prelu",
                factor_layers: tuple[int, ...] = (2, 4, 6),
                mids: dict[int, int] | None = None,
                widths: tuple[int, int] = (192, 320),
                pretrained: bool = True) -> FactorizedPrior:
    """Return a FactorizedPrior with the requested encoder surgery."""
    model = FactorizedPrior(N=widths[0], M=widths[1])

    if pretrained and widths == (192, 320):
        from compressai.zoo import bmshj2018_factorized
        ref = bmshj2018_factorized(quality=quality, pretrained=True)
        model.load_state_dict(ref.state_dict())

    g_a = model.g_a
    for i, module in enumerate(list(g_a)):
        if isinstance(module, GDN):
            channels = int(module.gamma.shape[0])
            if act == "keep":
                continue
            if act == "approx":
                setattr(g_a, str(i), ApproxGDN(module))
            else:
                setattr(g_a, str(i), _activation(act, channels, module))

    mids = mids or {}
    for i in factor_layers:
        conv = g_a[i]
        if not isinstance(conv, nn.Conv2d):
            continue
        mid = int(mids.get(i, conv.out_channels))
        setattr(g_a, str(i), FactorizedConv2d(
            conv.in_channels, conv.out_channels, mid,
            stride=int(conv.stride[0]), bias=conv.bias is not None))

    return model


def describe(model: FactorizedPrior) -> dict:
    convs = 0
    params = 0
    for m in model.g_a.modules():
        if isinstance(m, (nn.Conv2d, FactorizedConv2d)):
            convs += 1
    for p in model.g_a.parameters():
        params += p.numel()
    return {"encoder_convs": convs, "encoder_params": params}
