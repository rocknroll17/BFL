"""Pass the species id from AniMer+'s training data loader to the loss.

In third_party/AniMerPlus/amr/datasets/animal3d_dataset.py, Train3DDataset.__getitem__ fills both
batch['category'] and batch['supercategory'] with the record's supercategory (6 for every bird), so
the CUB-200 class id stored in the training json never reaches the loss. Without the species id the
OU prior cannot look up D_tax. use_species_labels() makes __getitem__ return the record's own
'category' and 'supercategory'. It must be called before the data module is set up.
"""
from __future__ import annotations

import numpy as np


def use_species_labels() -> None:
    from amr.datasets.animal3d_dataset import Train3DDataset

    original = Train3DDataset.__getitem__
    if getattr(original, "_bfl_species_labels", False):
        return

    def __getitem__(self, item):
        out = original(self, item)
        record = self.data["data"][item]
        if "category" in record and "supercategory" in record:
            out["category"] = np.array(record["category"], dtype=np.int32)
            out["supercategory"] = np.array(record["supercategory"], dtype=np.int32)
        return out

    __getitem__._bfl_species_labels = True
    Train3DDataset.__getitem__ = __getitem__
