"""Build the 17 x 17 taxonomic distance matrix D_tax of the CtrlAVES3D species.

Distance = Linnaean rank of the lowest shared taxon (IOC / Clements):
same species 0, same genus 0.5, same family 1, same order 2, sister orders 4,
other orders 5, outgroup (Procellariiformes) 6.

Output: phylo_distance_matrix.npy (17, 17) float32, symmetric, zero diagonal; row i is SPECIES[i].
"""

import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# Taxonomic tree for the 17 CtrlAVES3D species
# Structure: (class -> order -> family -> genus -> species)
# Each species stored as dict with index, folder, common, scientific
# -----------------------------------------------------------------------------

SPECIES = [
    # index, folder_name, scientific_name, genus, family, order
    (0, "Laysan_Albatross",          "Phoebastria_immutabilis",  "Phoebastria", "Diomedeidae",   "Procellariiformes"),
    (1, "Painted_Bunting",           "Passerina_ciris",           "Passerina",   "Cardinalidae",  "Passeriformes"),
    (2, "Cardinal",                  "Cardinalis_cardinalis",     "Cardinalis",  "Cardinalidae",  "Passeriformes"),
    (3, "American_Crow",             "Corvus_brachyrhynchos",     "Corvus",      "Corvidae",      "Passeriformes"),
    (4, "Northern_Flicker",          "Colaptes_auratus",          "Colaptes",    "Picidae",       "Piciformes"),
    (5, "Scissor_tailed_Flycatcher", "Tyrannus_forficatus",       "Tyrannus",    "Tyrannidae",    "Passeriformes"),
    (6, "Boat_tailed_Grackle",       "Quiscalus_major",           "Quiscalus",   "Icteridae",     "Passeriformes"),
    (7, "Evening_Grosbeak",          "Hesperiphona_vespertina",   "Hesperiphona","Fringillidae",  "Passeriformes"),
    (8, "California_Gull",           "Larus_californicus",        "Larus",       "Laridae",       "Charadriiformes"),
    (9, "Blue_Jay",                  "Cyanocitta_cristata",       "Cyanocitta",  "Corvidae",      "Passeriformes"),
    (10, "Green_Kingfisher",         "Chloroceryle_americana",    "Chloroceryle","Alcedinidae",   "Coraciiformes"),
    (11, "White_breasted_Kingfisher","Halcyon_smyrnensis",        "Halcyon",     "Alcedinidae",   "Coraciiformes"),
    (12, "Mallard",                  "Anas_platyrhynchos",        "Anas",        "Anatidae",      "Anseriformes"),
    (13, "Horned_Puffin",            "Fratercula_corniculata",    "Fratercula",  "Alcidae",       "Charadriiformes"),
    (14, "Geococcyx",                "Geococcyx_californianus",   "Geococcyx",   "Cuculidae",     "Cuculiformes"),
    (15, "Cedar_Waxwing",            "Bombycilla_cedrorum",       "Bombycilla",  "Bombycillidae", "Passeriformes"),
    (16, "Pileated_Woodpecker",      "Dryocopus_pileatus",        "Dryocopus",   "Picidae",       "Piciformes"),
]

# -----------------------------------------------------------------------------
# Phylogenetic distance via taxonomic hierarchy
# -----------------------------------------------------------------------------
# Levels (from shallow to deep shared ancestor):
#   same species -> 0.0
#   same genus   -> 0.5
#   same family  -> 1.0
#   same order   -> 2.0
#   different order (but sister orders, e.g. Passeriformes clades) -> 4.0
#   distant orders -> 5.0  (generic)
#   very distant (e.g., Procellariiformes vs Passerines) -> 6.0
# -----------------------------------------------------------------------------

# Sister orders (Jarvis et al. 2014, Prum et al. 2015) get distance 4.0 instead of 5.0.
SISTER_ORDER_GROUPS = [
    # Land bird radiation (Telluraves): Coraciiformes, Piciformes close
    {"Coraciiformes", "Piciformes"},
    # Charadriiformes is water bird clade, closer to gulls/puffins -> self-grouped
    # Anseriformes is basal to Galloanseres
    # Cuculiformes is within Neoaves landbird clade
]

def sister_orders(o1: str, o2: str) -> bool:
    for group in SISTER_ORDER_GROUPS:
        if o1 in group and o2 in group:
            return True
    return False

def pairwise_distance(sp_i, sp_j) -> float:
    """Hierarchical path-length distance between two species records."""
    _, _, sci_i, gen_i, fam_i, ord_i = sp_i
    _, _, sci_j, gen_j, fam_j, ord_j = sp_j
    if sci_i == sci_j:  return 0.0
    if gen_i == gen_j:  return 0.5
    if fam_i == fam_j:  return 1.0
    if ord_i == ord_j:  return 2.0
    if sister_orders(ord_i, ord_j):  return 4.0
    # Special: Procellariiformes is a deeply branching seabird order
    if "Procellariiformes" in (ord_i, ord_j):
        return 6.0
    return 5.0

def build_matrix():
    n = len(SPECIES)
    d = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i, j] = pairwise_distance(SPECIES[i], SPECIES[j])
    return d

def validate(d: np.ndarray):
    n = d.shape[0]
    assert d.shape == (n, n)
    # Symmetric
    assert np.allclose(d, d.T), "Matrix not symmetric"
    # Zero diagonal
    assert np.all(np.diag(d) == 0), "Diagonal not zero"
    # Triangle inequality (weak form): d(i,j) <= d(i,k) + d(k,j)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if d[i, j] > d[i, k] + d[k, j] + 1e-5:
                    raise AssertionError(f"Triangle violated: {i},{j},{k}")
    # Within-genus distance < within-family < within-order
    cardinal_idx = [idx for idx, *_ in SPECIES if SPECIES[idx][4] == "Cardinalidae"]
    if len(cardinal_idx) >= 2:
        # Painted Bunting (1) & Cardinal (2) are both Cardinalidae but different genera
        i, j = 1, 2
        assert d[i, j] == 1.0, f"Expected same-family=1.0, got {d[i,j]}"
    # Two Corvidae (Am. Crow 3 & Blue Jay 9) -> same family
    assert d[3, 9] == 1.0, f"Corvidae same-family expected 1.0, got {d[3,9]}"
    # Two Alcedinidae (10 & 11) -> same family
    assert d[10, 11] == 1.0
    # Two Picidae (4 & 16) -> same family
    assert d[4, 16] == 1.0
    # Coraciiformes & Piciformes sister -> 4.0
    assert d[10, 4] == 4.0
    # Procellariiformes vs Passeriformes -> 6.0
    assert d[0, 3] == 6.0, f"Albatross vs Crow expected 6.0, got {d[0, 3]}"
    print("[OK] Matrix passes all validation checks.")

def main():
    out_dir = Path(__file__).parent
    d = build_matrix()
    validate(d)

    np.save(out_dir / "phylo_distance_matrix.npy", d)

    print(f"[saved] {out_dir}/phylo_distance_matrix.npy  shape={d.shape}")
    print("\nDistance matrix (first 8x8):")
    for i in range(min(8, d.shape[0])):
        print(" ".join(f"{d[i, j]:.1f}" for j in range(min(8, d.shape[1]))))

    # Basic stats
    off_diag = d[~np.eye(d.shape[0], dtype=bool)]
    print(f"\nDistance stats: min={off_diag.min():.1f}, max={off_diag.max():.1f}, mean={off_diag.mean():.2f}")

if __name__ == "__main__":
    main()
