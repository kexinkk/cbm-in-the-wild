# Concept Bottleneck Models in the Wild: Assessing Interpretability under Image Corruptions

Kexin Zhang, Katharina Prasse, Mishal Fatima, Margret Keuper (University of Mannheim)

Trustworthy AI for Good Workshop @ NeurIPS 2026
([OpenReview](https://openreview.net/forum?id=OxAL5DSut0) · [NeurIPS page](https://neurips.cc/virtual/2026/165351)) ·
Self-Evolving Diversity-Driven Search for Robust AI Systems Workshop @ NeurIPS 2026
([OpenReview](https://openreview.net/forum?id=fXMXuVdcHP) · [NeurIPS page](https://neurips.cc/virtual/2026/175729))

We follow image corruptions through three concept bottleneck models (DCBM, Label-free CBM,
Post-hoc CBM) on a frozen CLIP ViT-B/16. The CLIP embedding moves little, yet the concept
ranking the head reads is reordered for most images. A per-concept shift score $d_k$ exposes
which concepts are stable, and keeping only the stable ones beats a size-matched random
subset under corruption for all three models.

This repository holds the code, configs, result files (`summary.json`, CSV, NPZ) and
figures behind every number in the paper. Checkpoints are attached to the
[GitHub release](https://github.com/kexinkk/cbm-in-the-wild/releases).

## Layout

```
src/cbm_in_the_wild/   importable package: corruptions, CLIP embedding, DCBM bridge,
                       LF-CBM / PCBM implementations, stage metrics, d_k
experiments/<name>/    runner + config.yaml + result files for one experiment
scripts/data_prep/     dataset preparation, CLIP embedding, SAM2 crops, concept bank
scripts/slurm/         the SLURM jobs we ran (bwHPC partitions; adapt to your cluster)
figures/               the paper figures
third_party/           DCBM and Label-free CBM upstream code (git submodules)
```

## Paper → experiment

| Paper | Experiment directory | Result files |
|---|---|---|
| Corruption severity calibration (App. B, PSNR/SSIM) | `corruption_baselines` | `summary.json` |
| DCBM reproduction and K sweep (App. C) | `imagenet_baseline`, `cifar10_baseline` | `summary.json`, `seeds.jsonl` |
| Stage-wise propagation, DCBM (§3, App. D) | `imagenet_c_propagation`, `cifar10_c_propagation` | `<corruption>_s<sev>/summary.json`, `stage_metrics.jsonl` |
| Stage-wise propagation, LF-CBM / PCBM (§3, App. D) | `{imagenet,cifar10}_{lfcbm,pcbm}_genericity` | `*stage_metrics.jsonl` |
| Per-concept shift $d_k$ (§4.1) | `imagenet_concept_shift_robust_set` (DCBM), `imagenet_{lfcbm,pcbm}_genericity` | `*concept_shift_table.csv`, `*shift_grids.npz` |
| Capacity-matched selection, Table 2 (§4.2) | `imagenet_concept_shift_robust_set_full_val` (DCBM), `imagenet_{lfcbm,pcbm}_genericity_full_val` | `*ablation_capacity_matched_summary.json` |
| $d_k$ on a disjoint image pool, unseen corruption types (App. E) | `imagenet_dk_disjoint_pool` | `summary.json`, `loco/loco_summary.json` |
| Crop geometry, DCBM (§4.3) | `imagenet_crop_quality_dk` | `summary*.json`, `*_table.csv` |

The medoid-versus-mean bank control (App. C) is not part of this release.

## Setup

```bash
git clone --recursive https://github.com/kexinkk/cbm-in-the-wild.git
cd cbm-in-the-wild
conda create -n cbm-wild python=3.11 -y && conda activate cbm-wild
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install git+https://github.com/openai/CLIP.git
# SAM2, only to rebuild the DCBM concept bank (installed without deps, it runs on torch 2.1.2)
pip install "hydra-core>=1.3.2" "iopath>=0.1.10"
pip install -e third_party/DCBM/data/Segments/scripts/utilities/segment_anything_2 --no-deps --no-build-isolation
```

Environment variables (defaults in `src/cbm_in_the_wild/paths.py`):

| Variable | Default | Points to |
|---|---|---|
| `DCBM_REPO` | `~/DCBM` | the DCBM checkout; use `third_party/DCBM` |
| `LFCBM_REPO` | `~/Label-free-CBM` | the Label-free CBM checkout; use `third_party/Label-free-CBM` |
| `DCBM_DATA_DIR` | `./data` | datasets and caches |
| `DCBM_IMAGENET_DIR`, `DCBM_IMGNET_C_DIR` | under `DCBM_DATA_DIR` | ImageNet image folder, ImageNet-C |
| `PCBM_BRODEN_DIR` | none | Broden concept images, only to rebuild the PCBM CAV bank |

```bash
export DCBM_REPO=$PWD/third_party/DCBM LFCBM_REPO=$PWD/third_party/Label-free-CBM
```

## Checkpoints

```bash
bash scripts/download_checkpoints.sh   # ~30 MB; checks sha256 and unpacks in place
```

| Family | Files | Clean top-1 (ImageNet) |
|---|---|---|
| DCBM, K=2048 | `data/checkpoints/imagenet_baseline/` (bank `centroids.pt`, standardizer `scaler.pkl`, head `linear_head.pt`, medoid crop ids `centroid_crops.json`) | 69.77 |
| Label-free CBM | `experiments/imagenet_lfcbm_genericity/model/` | 75.48 |
| Post-hoc CBM (multimodal bank) | `experiments/imagenet_pcbm_genericity/model_mmL/` | 72.04 |

CIFAR-10 versions of all three are in the same release. The concept bank is a set of SAM2
crop medoids; we release their CLIP embeddings and crop ids, not the crops, which are
pieces of ImageNet images. Rebuild the crops from your own ImageNet copy (below).

## Data

- **ImageNet-1k** (ILSVRC2012) and **CIFAR-10** as image folders: `scripts/data_prep/{imagenet,cifar10}_to_imagefolder.py`, then `*_split_val.py` (90/10 train/val split, seed 0).
- **ImageNet-C / CIFAR-10-C**: the distributed files of Hendrycks & Dietterich (2019) for motion blur, Gaussian noise, JPEG compression and fog (`scripts/data_prep/imagenet_c_download.py`). We never regenerate them.
- **Six sensor corruptions** (pixel dropout, salt & pepper, sparse occlusion, patch masking, row/column dropout, structured dropout) are generated on the fly by `cbm_in_the_wild.corruptions.sensor_c`. Their per-image masks are seeded by a Python string hash, so launch every corruption job with `PYTHONHASHSEED=0` (ImageNet) or `PYTHONHASHSEED=42` (CIFAR-10, as in the sbatch files).

## Reproducing

All jobs ran under SLURM. `scripts/slurm/<experiment>/` has the exact chain for each
experiment; the steps below give the order and the entry points.

1. **DCBM bank and baseline** (`scripts/slurm/imagenet_baseline/`, steps 01–07):
   split val → `imagenet_build_subsets.py` (50 images/class, seed 42) → `imagenet_clip_extract.py` →
   `imagenet_sam2_extract.py` → `imagenet_segcrop_embed.py` → `experiments/imagenet_baseline/run.py`
   (sweep, then 3-seed replicate). CIFAR-10 is the same under `cifar10_baseline/`.
2. **Stage-wise propagation**: `experiments/imagenet_c_propagation/run_cell.py --task-id {0..49}`
   (one cell = corruption × severity; it also caches the corrupted CLIP embeddings that every later
   step reuses), then `aggregate.py`. LF-CBM and PCBM: `train_lfcbm.py --stage a|b`,
   `train_pcbm.py --stage a|aL|b --bank mmL`, then `run_propagation.py`.
3. **$d_k$**: `experiments/imagenet_concept_shift_robust_set/run.py` (DCBM) and
   `run_concept_shift.py` in the LF-CBM / PCBM dirs.
4. **Capacity-matched selection** (Table 2): `scripts/slurm/concept_shift_robust_set_full_val/submit.sh`
   (DCBM) and `run_ablation.py` in `imagenet_{lfcbm,pcbm}_genericity_full_val` (`--bank mmL` for PCBM).
   Every arm refits a fresh head on the 128,118-image held-out pool under seeds 0, 1, 2.
5. **Disjoint pool and leave-one-corruption-out**:
   `PYTHONHASHSEED=0 python experiments/imagenet_dk_disjoint_pool/embed_train_pool.py` →
   `compute_dk.py` → `bash scripts/slurm/dk_disjoint_loco/submit.sh` → `aggregate_loco.py`.
6. **Crop geometry**: `experiments/imagenet_crop_quality_dk/run.py`, `run_causal.py --phase embed|dk`.

Each runner skips cells whose output exists, so interrupted jobs resume.

## Notes on the upstream DCBM code

We use DCBM unmodified and patch around these issues from this repository:

- `utilities/clip.py::compute_embeddings_batched` is broken; we use `cbm_in_the_wild.clip_embed`.
- `utilities/SAM2.py::resize_image` is a no-op for 32×32 images; `scripts/data_prep/cifar10_sam2_extract.py` upsamples to 1024 px in memory.
- The concept subsets (`Embeddings/subsets/<Dataset>/<Dataset>_rand_50.txt`) must exist before SAM2 runs.
- `create_segments_SAM2.py` hardcodes `configs/sam2_1/...`; the real hydra path is `configs/sam2.1/...`.
- `utils/dcbm.py` calls `torch.manual_seed(5)` at import; build models through `cbm_in_the_wild.dcbm_bridge.build_cbm`.
- File-name casing differs per dataset: `ImageNet` (embeddings, segcrops, subsets) vs `cifar10`.
- For ImageNet, `CBM.__init__` caches per-class concept subsets in `./processed_data/` relative to the working directory. If the subset file was missing, an empty cache is written and silently reused, so delete `processed_data/` after fixing a path.
- Upstream writes the SAM2 crops with `cv2.imwrite` on an RGB array, so the crops on disk have red and blue swapped. The released bank was embedded from these crops; flip the channels only for display.

## Citation

```bibtex
@inproceedings{zhang2026cbmwild,
  title     = {Concept Bottleneck Models in the Wild: Assessing Interpretability under Image Corruptions},
  author    = {Zhang, Kexin and Prasse, Katharina and Fatima, Mishal and Keuper, Margret},
  booktitle = {Trustworthy AI for Good Workshop at the Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). The submodules keep their own terms: DCBM (MIT, Prasse et al.) and
Label-free CBM (Oikarinen et al.). The Post-hoc CBM code here is our reimplementation of
Yuksekgonul et al. (ICLR 2023). The released Label-free CBM concept list (`concepts.txt`) is
the filtered ImageNet concept set from the Label-free CBM repository.
