"""Geo-STN architecture used in the formal cross-event experiments."""

from __future__ import annotations

import torch
from torch import nn


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, token_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, token_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, token_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class GroupTokenEncoder(nn.Module):
    def __init__(self, n_features: int, token_dim: int, emb: int, dropout: float):
        super().__init__()
        self.n_features = int(n_features)
        effective_features = max(1, self.n_features)
        self.tokenizer = FeatureTokenizer(effective_features, token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=2, dim_feedforward=token_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.feature_interactor = nn.TransformerEncoder(layer, num_layers=1)
        self.raw_encoder = nn.Sequential(
            nn.Linear(effective_features, emb), nn.LayerNorm(emb), nn.GELU(),
            nn.Dropout(dropout), ResidualMLPBlock(emb, dropout * 0.7),
        )
        self.fusion = nn.Sequential(
            nn.Linear(token_dim + emb, emb), nn.LayerNorm(emb), nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_features == 0:
            x = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        tokens = self.feature_interactor(self.tokenizer(x))
        raw = self.raw_encoder(x)
        return self.fusion(torch.cat([tokens[:, 0], raw], dim=1))


class PositiveLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.raw_weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.08 - 2.0)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, nn.functional.softplus(self.raw_weight), self.bias)


class MonotonicPGABranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PositiveLinear(1, 12), nn.Softplus(),
            PositiveLinear(12, 12), nn.Softplus(), PositiveLinear(12, 1),
        )

    def forward(self, pga: torch.Tensor) -> torch.Tensor:
        return self.net(pga).squeeze(-1)


class UnconstrainedPGABranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 12), nn.GELU(), nn.Linear(12, 12), nn.GELU(),
            nn.Linear(12, 1),
        )

    def forward(self, pga: torch.Tensor) -> torch.Tensor:
        return self.net(pga).squeeze(-1)


class GeoTabSTNV2(nn.Module):
    """Additive static, PGA and spatial components; named Geo-STN in the paper."""

    def __init__(self, feature_names: list[str], monotonic_pga: bool):
        super().__init__()
        self.feature_names = list(feature_names)
        pga_indices = [i for i, name in enumerate(self.feature_names)
                       if str(name).lower() == "usgs_pga"]
        if len(pga_indices) != 1:
            raise ValueError("Exactly one USGS_PGA feature is required")
        self.pga_idx = int(pga_indices[0])
        static_idx = [i for i in range(len(self.feature_names)) if i != self.pga_idx]
        self.register_buffer("static_idx", torch.tensor(static_idx, dtype=torch.long),
                             persistent=False)
        token_dim, emb = 24, 72
        self.static_encoder = GroupTokenEncoder(len(static_idx), token_dim, emb, 0.18)
        self.static_head = nn.Linear(emb, 1)
        self.pga_branch = MonotonicPGABranch() if monotonic_pga else UnconstrainedPGABranch()
        self.spatial_encoder = nn.Sequential(
            nn.Linear(len(static_idx) * 2, emb), nn.LayerNorm(emb), nn.GELU(),
            nn.Dropout(0.18), ResidualMLPBlock(emb, 0.12),
        )
        self.spatial_head = nn.Linear(emb, 1)
        nn.init.zeros_(self.spatial_head.weight)
        nn.init.zeros_(self.spatial_head.bias)

    def forward(self, center: torch.Tensor, neigh_mean: torch.Tensor,
                return_parts: bool = False):
        static_center = center.index_select(1, self.static_idx.to(center.device))
        static_neigh = neigh_mean.index_select(1, self.static_idx.to(center.device))
        static_logit = self.static_head(self.static_encoder(static_center)).squeeze(-1)
        pga_delta = self.pga_branch(center[:, self.pga_idx:self.pga_idx + 1])
        spatial_input = torch.cat([static_neigh, static_neigh - static_center], dim=1)
        spatial_delta = self.spatial_head(self.spatial_encoder(spatial_input)).squeeze(-1)
        logit = static_logit + pga_delta + spatial_delta
        if return_parts:
            return {"logit": logit, "static_logit": static_logit,
                    "pga_delta": pga_delta, "spatial_delta": spatial_delta}
        return logit
