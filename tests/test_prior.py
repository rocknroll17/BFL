"""CPU checks of the OU prior (python -m pytest tests)."""
import math
from pathlib import Path

import torch

from bfl.prior import OUPhyloPrior, TaxonomicOULoss

ROOT = Path(__file__).resolve().parents[1]
D_TAX = ROOT / "data/phylogeny/phylo_distance_matrix.npy"
CUB_MAP = ROOT / "data/phylogeny/cub_category_mapping.json"


def test_covariance_matches_formula():
    prior = OUPhyloPrior(alpha_init=0.5, sigma_init=0.5, alpha_min=0.0, learnable_alpha=False, learnable_sigma=False)
    cov = prior.build_cov(torch.tensor([[0.0, 2.0], [2.0, 0.0]]))
    assert abs(cov[0, 1].item() - 0.5 ** 2 / (2 * 0.5) * math.exp(-0.5 * 2.0)) < 1e-6


def test_nll_is_finite_and_differentiable():
    torch.manual_seed(0)
    prior = OUPhyloPrior()
    d = torch.rand(17, 17) * 5
    d = (d + d.T) / 2
    d.fill_diagonal_(0)
    beta = torch.randn(8, 15, requires_grad=True)
    loss = prior(beta, torch.randint(0, 17, (8,)), d)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(beta.grad).all()
    assert prior.alpha_raw.grad is not None and prior.sigma_raw.grad is not None
    far = torch.full((4, 4), 100.0).fill_diagonal_(0)            # nearly uncorrelated species stay positive definite
    assert torch.isfinite(prior(torch.randn(4, 15), torch.arange(4), far))


def test_species_mapping_and_bird_selection():
    loss = TaxonomicOULoss(str(D_TAX), str(CUB_MAP))
    rows, mask = loss.species_rows(torch.tensor([2, 16, 188, 17, 999]), torch.tensor([6, 6, 6, 6, 6]))
    assert rows.tolist() == [0, 1, 16, 2] and mask.tolist() == [True, True, True, True, False]
    _, mask = loss.species_rows(torch.tensor([2, 16, 17]), torch.tensor([6, 1, 6]))   # supercategory 1: not a bird
    assert mask.tolist() == [True, False, True]


def test_fewer_than_two_birds_contribute_zero():
    loss = TaxonomicOULoss(str(D_TAX), str(CUB_MAP))
    assert loss(torch.randn(1, 15), torch.tensor([2]), torch.tensor([6])).item() == 0.0
    assert loss(torch.randn(3, 15), torch.tensor([999] * 3), torch.tensor([1] * 3)).item() == 0.0
