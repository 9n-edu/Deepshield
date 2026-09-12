"""
model_arch.py — Demo 用 Autoencoder 架構（僅推論，不含訓練依賴）

對齊 bottleneck256_batch32 / retrain_fusion2 的 bn256 結構。
"""

from __future__ import annotations

import torch.nn as nn

LATENT_DIM = 256
IMG_SIZE = 128
OUT_CHANNELS = 8
IN_CHANNELS = 1


def _spatial_after_pools(img_size: int, n_pools: int = 4) -> int:
    s = img_size
    for _ in range(n_pools):
        s //= 2
    return s


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        act_fn: nn.Module,
        spatial: int,
    ):
        super().__init__()
        self.spatial = spatial
        flat_in = spatial * spatial * out_channels * 8
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(out_channels, 2 * out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.BatchNorm2d(2 * out_channels),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2 * out_channels, 4 * out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.BatchNorm2d(4 * out_channels),
            nn.Conv2d(4 * out_channels, 4 * out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.BatchNorm2d(4 * out_channels),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4 * out_channels, 8 * out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.BatchNorm2d(8 * out_channels),
            nn.Conv2d(8 * out_channels, 8 * out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.BatchNorm2d(8 * out_channels),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(flat_in, out_channels * latent_dim),
            act_fn,
            nn.Linear(out_channels * latent_dim, latent_dim),
            act_fn,
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        act_fn: nn.Module,
        spatial: int,
    ):
        super().__init__()
        self.spatial = spatial
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * out_channels),
            act_fn,
            nn.Linear(latent_dim * out_channels, 8 * out_channels * spatial * spatial),
            act_fn,
            nn.Unflatten(1, (8 * out_channels, spatial, spatial)),
        )
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.ConvTranspose2d(8 * out_channels, 8 * out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.BatchNorm2d(8 * out_channels),
            nn.ConvTranspose2d(8 * out_channels, 4 * out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.BatchNorm2d(4 * out_channels),
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.ConvTranspose2d(4 * out_channels, 4 * out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.BatchNorm2d(4 * out_channels),
            nn.ConvTranspose2d(4 * out_channels, 2 * out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.BatchNorm2d(2 * out_channels),
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.ConvTranspose2d(2 * out_channels, 1 * out_channels, kernel_size=3, padding=1),
            act_fn,
            nn.BatchNorm2d(1 * out_channels),
            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.ConvTranspose2d(1 * out_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(self.linear(x))


class Autoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build_model(
    *,
    in_channels: int = IN_CHANNELS,
    out_channels: int = OUT_CHANNELS,
    latent_dim: int = LATENT_DIM,
    img_size: int = IMG_SIZE,
) -> Autoencoder:
    spatial = _spatial_after_pools(img_size)
    if spatial < 1:
        raise ValueError(f"img_size={img_size} 太小，無法經過 4 次 MaxPool")
    enc = Encoder(in_channels, out_channels, latent_dim, nn.ReLU(inplace=True), spatial)
    dec = Decoder(in_channels, out_channels, latent_dim, nn.ReLU(inplace=True), spatial)
    return Autoencoder(enc, dec)
