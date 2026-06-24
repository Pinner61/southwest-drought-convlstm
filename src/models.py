"""
Models.

ConvLSTMForecaster  - spatiotemporal model that sees the whole drought field.
PixelLSTM           - the key baseline: an LSTM applied independently to every
                      pixel's time series, with NO spatial coupling. The whole
                      scientific question is whether the ConvLSTM's spatial
                      awareness buys real forecast skill over this.

Persistence and climatology are computed directly in evaluate.py.
"""
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def init_state(self, b, h, w, device):
        z = torch.zeros(b, self.hid_ch, h, w, device=device)
        return z, z.clone()


class ConvLSTMForecaster(nn.Module):
    """Input (B, T, C, H, W) -> next-step field (B, C, H, W)."""

    def __init__(self, in_ch, hid_ch=32, layers=1, k=3):
        super().__init__()
        self.cells = nn.ModuleList(
            ConvLSTMCell(in_ch if i == 0 else hid_ch, hid_ch, k) for i in range(layers)
        )
        self.head = nn.Conv2d(hid_ch, in_ch, 1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        states = [cell.init_state(b, h, w, x.device) for cell in self.cells]
        for ti in range(t):
            inp = x[:, ti]
            for li, cell in enumerate(self.cells):
                states[li] = cell(inp, states[li])
                inp = states[li][0]
        return self.head(states[-1][0])


class PixelLSTM(nn.Module):
    """Shared LSTM applied per-pixel. No spatial information by construction."""

    def __init__(self, in_ch, hid=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hid, layers, batch_first=True)
        self.head = nn.Linear(hid, in_ch)

    def forward(self, x):
        b, t, c, h, w = x.shape
        seq = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        out, _ = self.lstm(seq)
        out = self.head(out[:, -1])
        return out.reshape(b, h, w, c).permute(0, 3, 1, 2)


def build_model(name, in_ch, cfg):
    if name == "convlstm":
        return ConvLSTMForecaster(in_ch, cfg.get("hidden", 32), cfg.get("layers", 1))
    if name == "pixel_lstm":
        return PixelLSTM(in_ch, cfg.get("pixel_hidden", 64), cfg.get("layers", 1))
    raise ValueError(name)
