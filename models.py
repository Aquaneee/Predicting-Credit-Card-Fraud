import numpy as np
import torch
import torch.nn as nn

class PeriodicNumericalEmbedding(nn.Module):
    def __init__(self, num_features, n_frequencies=8, sigma=0.1):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_features, n_frequencies) * sigma)

    def forward(self, x):
        v = 2 * np.pi * x.unsqueeze(-1) * self.weights.unsqueeze(0)
        return torch.cat([torch.sin(v), torch.cos(v)], dim=-1).flatten(start_dim=1)

class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class TabularResNet(nn.Module):
    def __init__(self, num_features, cat_dims, hidden_dim=384, num_blocks=3, dropout=0.25):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(c, min(32, (c + 1) // 2)) for c in cat_dims
        ])
        emb_dim = sum(e.embedding_dim for e in self.embeddings)

        self.has_num = num_features > 0
        if self.has_num:
            self.periodic = PeriodicNumericalEmbedding(num_features, n_frequencies=8)
            num_in_dim = num_features + (num_features * 16)
        else:
            num_in_dim = 0

        self.in_proj = nn.Sequential(
            nn.Linear(emb_dim + num_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(nn.BatchNorm1d(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, num_x, cat_x):
        feats = []
        if self.has_num:
            feats.extend([num_x, self.periodic(num_x)])
        if self.embeddings:
            feats.extend([emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)])
        x = self.in_proj(torch.cat(feats, dim=1))
        for b in self.blocks:
            x = b(x)
        return self.head(x).squeeze(-1)

class CrossLayerV2(nn.Module):
    def __init__(self, dim, rank=64):
        super().__init__()
        self.u = nn.Linear(dim, rank, bias=False)
        self.v = nn.Linear(rank, dim, bias=True)
        self.act = nn.GELU()

    def forward(self, x0, xl):
        return x0 * self.v(self.act(self.u(xl))) + xl

class DCNv2(nn.Module):
    def __init__(self, num_features, cat_dims, hidden_dim=384, num_cross=3, num_deep=3, dropout=0.25):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(c, min(32, (c + 1) // 2)) for c in cat_dims
        ])
        emb_dim = sum(e.embedding_dim for e in self.embeddings)
        in_dim = num_features + emb_dim

        self.in_norm = nn.BatchNorm1d(in_dim)
        self.cross_layers = nn.ModuleList([CrossLayerV2(in_dim, rank=64) for _ in range(num_cross)])

        deep_layers = []
        curr_dim = in_dim
        for _ in range(num_deep):
            deep_layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            curr_dim = hidden_dim
        self.deep_net = nn.Sequential(*deep_layers)
        self.head = nn.Linear(in_dim + hidden_dim, 1)

    def forward(self, num_x, cat_x):
        feats = [num_x] if num_x.shape[1] > 0 else []
        if self.embeddings:
            feats.extend([emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)])
        x0 = self.in_norm(torch.cat(feats, dim=1))

        xc = x0
        for layer in self.cross_layers:
            xc = layer(x0, xc)
        xd = self.deep_net(x0)

        return self.head(torch.cat([xc, xd], dim=1)).squeeze(-1)

class FTTransformer(nn.Module):
    def __init__(self, num_features, cat_dims, d_token=64, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        self.num_features = num_features
        if num_features > 0:
            self.num_weight = nn.Parameter(torch.randn(num_features, d_token) * 0.02)
            self.num_bias = nn.Parameter(torch.zeros(num_features, d_token))

        self.cat_embeddings = nn.ModuleList([nn.Embedding(c, d_token) for c in cat_dims])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.GELU(), nn.Linear(d_token, 1))

    def forward(self, num_x, cat_x):
        B = num_x.shape[0]
        tokens = [self.cls_token.expand(B, -1, -1)]

        if self.num_features > 0:
            num_tokens = num_x.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            tokens.append(num_tokens)

        if self.cat_embeddings:
            cat_tokens = torch.stack([emb(cat_x[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
            tokens.append(cat_tokens)

        x = torch.cat(tokens, dim=1)
        x = self.transformer(x)
        return self.head(x[:, 0, :]).squeeze(-1)

def build_model(model_name, num_features, cat_dims):
    if model_name == "resnet":
        return TabularResNet(num_features, cat_dims)
    elif model_name == "dcnv2":
        return DCNv2(num_features, cat_dims)
    elif model_name == "fttransformer":
        return FTTransformer(num_features, cat_dims)
    else:
        raise ValueError(f"Không hỗ trợ kiến trúc: {model_name}")