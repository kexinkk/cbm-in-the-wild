"""Post-hoc Concept Bottleneck Model (PCBM) genericity point.

Second architecture for the scope-v2 §3.4 genericity test (replaces UCBM,
2026-08-09). PCBM = Yuksekgonul et al., ICLR 2023. Deployed forward mirrors
DCBM/LF-CBM shape but the concept bank is built from CAVs:

    z(CLIP ViT-B/16) → raw = (z @ cavs.T + intercepts) / norms
                     → standardize (train mean/std)
                     → sparse linear head            [NO input-dependent gate]

Two concept sources (paper's two methods):
  * cav  — per-concept linear SVM on Broden concept images (primary, faithful)
  * mm   — CLIP text-encoder embeddings of concept names (cheap secondary)

Both live in the 512-d CLIP projection = the cached ImageNet + ImageNet-C
`clip_emb` space, so the 50 cached corruption cells are reused verbatim.
`concept_shift`/`ablation`/`propagation`/`cells` are copied from `lfcbm/`
(bank-agnostic; operate only on the [N,K] score matrix) and repointed here.
"""
