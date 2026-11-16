"""Taxonomic Ornstein-Uhlenbeck prior on the AVES shape codes of a batch.

Inter-species shape covariance is modelled as

    Cov(beta_i, beta_j) = sigma^2 / (2 alpha) * exp(-alpha * d_tax(i, j))

with learnable scalars alpha (selection strength) and sigma (drift). The
negative log-likelihood treats the D shape dimensions as independent draws
from the same B x B Gaussian:

    L = (D/2) log|Sigma| + (1/2) sum_d (beta_d - mu)^T Sigma^{-1} (beta_d - mu)

evaluated with a Cholesky factorisation plus diagonal jitter.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    """Return x such that softplus(x) = y (y > 0)."""
    if y <= 0:
        return 0.0
    return math.log(math.exp(y) - 1.0)


class OUPhyloPrior(nn.Module):
    """OU process prior over the shape codes of one mini-batch.

    Args:
        shape_dim:       D, dimension of the shape code (15 for AVES beta)
        alpha_init:      initial selection strength (> 0)
        sigma_init:      initial drift magnitude (> 0)
        learnable_alpha: make alpha a parameter
        learnable_sigma: make sigma a parameter
        learnable_mu:    make the ancestral mean mu (R^D) a parameter
        jitter:          diagonal jitter added to the covariance
        alpha_min:       hard floor on alpha
    """

    def __init__(
        self,
        shape_dim: int = 15,
        alpha_init: float = 1.0,
        sigma_init: float = 1.0,
        learnable_alpha: bool = True,
        learnable_sigma: bool = True,
        learnable_mu: bool = True,
        jitter: float = 1e-6,
        alpha_min: float = 1e-3,
    ):
        super().__init__()
        self.shape_dim = shape_dim
        self.jitter = jitter
        self.alpha_min = alpha_min

        alpha_raw = torch.tensor(_inv_softplus(alpha_init))
        sigma_raw = torch.tensor(_inv_softplus(sigma_init))

        if learnable_alpha:
            self.alpha_raw = nn.Parameter(alpha_raw)
        else:
            self.register_buffer("alpha_raw", alpha_raw)

        if learnable_sigma:
            self.sigma_raw = nn.Parameter(sigma_raw)
        else:
            self.register_buffer("sigma_raw", sigma_raw)

        if learnable_mu:
            self.mu = nn.Parameter(torch.zeros(shape_dim))
        else:
            self.register_buffer("mu", torch.zeros(shape_dim))

    # positive parameters via softplus
    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_raw) + self.alpha_min

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.sigma_raw)

    def build_cov(self, dist: torch.Tensor) -> torch.Tensor:
        """(N, N) pairwise taxonomic distances -> (N, N) OU covariance."""
        variance = self.sigma ** 2 / (2.0 * self.alpha)
        cov = variance * torch.exp(-self.alpha * dist)
        eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        return cov + self.jitter * eye

    def nll(self, beta: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of beta (B, D) under the OU prior with
        distances dist (B, B); returned per sample (divided by B)."""
        B, D = beta.shape
        assert dist.shape == (B, B), f"dist shape {dist.shape} != ({B},{B})"

        cov = self.build_cov(dist)
        L = torch.linalg.cholesky(cov)
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        centered = beta - self.mu.unsqueeze(0)
        z = torch.linalg.solve_triangular(L, centered, upper=False)   # L z = centered
        quad = (z ** 2).sum()
        nll = 0.5 * D * logdet + 0.5 * quad                            # constant term dropped
        return nll / B

    def forward(self, beta: torch.Tensor, folder_idx: torch.Tensor,
                phylo_matrix: torch.Tensor) -> torch.Tensor:
        """Slice the full species distance matrix by folder_idx and return the NLL.
        A batch with fewer than two samples contributes zero."""
        if folder_idx.shape[0] < 2:
            return beta.new_zeros(())
        dist = phylo_matrix[folder_idx][:, folder_idx]
        return self.nll(beta, dist)



class TaxonomicOULoss(nn.Module):
    """L_OU on AniMer+ batches.

    Picks the bird samples of a batch, maps their CUB-200 class ids to the rows of the
    17 x 17 taxonomic distance matrix D_tax, and returns the OU negative log-likelihood
    of their shape codes. Batches with fewer than two mapped birds contribute zero.

    Args:
        phylo_matrix_path: D_tax, (17, 17) .npy; row i is species i of data/phylogeny/build_distance_matrix.py
        cub_mapping_path:  json {"cub_category_to_folder_idx": {CUB-200 class id: D_tax row}}
    """

    BIRD_SUPERCATEGORY = 6

    def __init__(self, phylo_matrix_path: str, cub_mapping_path: str, shape_dim: int = 15,
                 alpha_init: float = 1.0, sigma_init: float = 1.0, jitter: float = 1e-6):
        super().__init__()
        d = np.load(phylo_matrix_path).astype(np.float32)
        assert d.ndim == 2 and d.shape[0] == d.shape[1], f"D_tax must be square, got {d.shape}"
        self.register_buffer("phylo_distance", torch.from_numpy(d))
        self.ou = OUPhyloPrior(shape_dim=shape_dim, alpha_init=alpha_init, sigma_init=sigma_init, jitter=jitter)

        cub_to_row = {int(k): int(v) for k, v in json.load(open(cub_mapping_path))["cub_category_to_folder_idx"].items()}
        lut = torch.full((max(cub_to_row) + 1,), -1, dtype=torch.long)    # CUB class id -> D_tax row, -1 if absent
        for cub_id, row in cub_to_row.items():
            lut[cub_id] = row
        self.cub_to_row = lut

    def species_rows(self, category: torch.Tensor, supercategory: torch.Tensor | None):
        """Return (D_tax rows of the mapped birds, boolean mask of those samples in the batch)."""
        if supercategory is not None:
            bird_mask = supercategory == self.BIRD_SUPERCATEGORY
        else:
            bird_mask = torch.ones_like(category, dtype=torch.bool)
        cats = category[bird_mask]
        lut = self.cub_to_row.to(cats.device)
        in_range = (cats >= 0) & (cats < lut.numel())
        rows = lut[cats.clamp(0, lut.numel() - 1)]
        rows = torch.where(in_range, rows, torch.full_like(rows, -1))
        valid = rows >= 0
        if not valid.all():
            bird_mask = bird_mask.clone()
            bird_mask[bird_mask.nonzero(as_tuple=False).squeeze(-1)[~valid]] = False
            rows = rows[valid]
        return rows, bird_mask

    def forward(self, pred_betas: torch.Tensor, category: torch.Tensor,
                supercategory: torch.Tensor | None = None) -> torch.Tensor:
        rows, bird_mask = self.species_rows(category, supercategory)
        if rows.numel() < 2:
            return pred_betas.new_zeros(())
        betas = pred_betas[bird_mask]
        return self.ou(betas, rows.to(betas.device), self.phylo_distance.to(betas.device))

    def alpha(self) -> torch.Tensor:
        return self.ou.alpha

    def sigma(self) -> torch.Tensor:
        return self.ou.sigma
