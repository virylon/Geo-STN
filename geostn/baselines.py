"""Public comparison architectures and fixed tree hyperparameters."""

from __future__ import annotations

import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from torch import nn


class RegularizedMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Dropout(0.38), nn.Linear(64, 24), nn.GELU(),
            nn.Dropout(0.25), nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TabularTransformerStudent(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        dim = 48
        self.value_weight = nn.Parameter(torch.empty(n_features, dim))
        self.value_bias = nn.Parameter(torch.zeros(n_features, dim))
        self.feature_embedding = nn.Parameter(torch.empty(n_features, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 4,
            dropout=0.15, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(dim)
        self.prob_head = nn.Linear(dim, 1)
        self.gain_head = nn.Linear(dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.value_weight)
        nn.init.normal_(self.feature_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, return_gain: bool = False):
        tokens = x.unsqueeze(-1) * self.value_weight.unsqueeze(0) + self.value_bias.unsqueeze(0)
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        pooled = self.norm(self.encoder(torch.cat([cls, tokens], dim=1))[:, 0])
        prob_logit = self.prob_head(pooled).squeeze(-1)
        if return_gain:
            return prob_logit, self.gain_head(pooled).squeeze(-1)
        return prob_logit


class PatchCNN(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, padding=1, groups=n_features),
            nn.Conv2d(n_features, 48, 1), nn.BatchNorm2d(48), nn.GELU(),
            nn.Dropout2d(0.16), nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.GELU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Dropout(0.28), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).squeeze(-1)


class GraphSAGE8(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features * 2, 72), nn.BatchNorm1d(72), nn.GELU(),
            nn.Dropout(0.34), nn.Linear(72, 28), nn.GELU(),
            nn.Dropout(0.22), nn.Linear(28, 1),
        )

    def forward(self, center: torch.Tensor, neigh_mean: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([center, neigh_mean], dim=1)).squeeze(-1)


NEURAL_BASELINES = {
    "MLP": RegularizedMLP,
    "FTTrans": TabularTransformerStudent,
    "PatchCNN": PatchCNN,
    "GraphSAGE": GraphSAGE8,
}


def tree_baseline(name: str, seed: int) -> Pipeline:
    if name == "RF":
        model = RandomForestRegressor(
            n_estimators=260, max_depth=18, min_samples_leaf=3,
            n_jobs=-1, random_state=seed,
        )
    elif name == "HGB":
        model = HistGradientBoostingRegressor(
            max_iter=220, learning_rate=0.045, max_leaf_nodes=31,
            l2_regularization=0.02, random_state=seed,
        )
    else:
        raise ValueError(name)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
