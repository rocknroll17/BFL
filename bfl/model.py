"""AniMer+ with the taxonomic OU prior in its training loss.

AniMer+ (third_party/AniMerPlus, amr/models/animerpp.py) trains with

    loss = loss_aves + loss_smal + w_con * L_con

where L_con is its family-contrastive loss. BFLModel adds the OU prior,

    loss = loss_aves + loss_smal + w_con * L_con + lambda_OU * L_OU,

and the paper's runs set w_con = 0 (cfg.LOSS_WEIGHTS.SUPCON), so that L_OU replaces L_con.
Inference is unchanged: the prior only enters the loss.
"""
from __future__ import annotations

from amr.models.animerpp import AniMerPlusPlus

from .prior import TaxonomicOULoss


class BFLModel(AniMerPlusPlus):
    def __init__(self, cfg, init_renderer: bool = True, ou_weight: float = 0.0,
                 phylo_matrix_path: str | None = None, cub_mapping_path: str | None = None):
        super().__init__(cfg, init_renderer=init_renderer)
        self.ou_weight = float(ou_weight)
        self.ou_prior = TaxonomicOULoss(phylo_matrix_path, cub_mapping_path) if phylo_matrix_path else None

    def compute_loss(self, batch, output, train: bool = True):
        loss = super().compute_loss(batch, output, train=train)
        if self.ou_weight > 0 and "aves_output" in output:
            ou = self.ou_prior(output["aves_output"]["pred_params"]["betas"], batch["category"], batch.get("supercategory"))
            loss = loss + self.ou_weight * ou
            if isinstance(output.get("losses"), dict):
                output["losses"]["loss_ou"] = ou.detach()
                output["losses"]["loss"] = loss.detach()
        return loss
