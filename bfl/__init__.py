"""Beyond Family Labels: a taxonomic Ornstein-Uhlenbeck prior for avian 3D shape recovery (NeurIPS 2026).

    prior.py     OU covariance over the shape codes of a batch and its Gaussian NLL
    model.py     BFLModel: AniMer+ with the OU prior added to its training loss
    datasets.py  passes the species id from AniMer+'s data loader to the loss
"""
