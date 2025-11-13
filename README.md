TAO-Net Research Snapshot
=========================

This repository hosts the in-progress implementation of **TAO-Net**, our two-stage adaptive OOD classification pipeline for encrypted traffic. The current snapshot keeps all research code, scripts, and experiments needed for the paper under review.

> ⚠️ Configuration availability  
> Final configuration files (hyperparameters, dataset manifests, and deployment scripts) will be published once the paper successfully completes peer review. Until then, only the minimal wiring required to reproduce internal experiments is provided.

Key directories
---------------

- `stage1_ood_detection/`: BLOOD + PCA hybrid scoring, calibration utilities, and visualization helpers.
- `stage1_data_processing/`: tooling that reconciles Stage-1 scores with raw packets and emits ID/OOD splits.
- `stage2_pacrep_* / stage2_id_branch.py`: PacRep/BERT ID branches for Tinghua, VPN, and Nontor datasets.
- `stage2_ood_llm/`: SPK templates, prompt builder, and LLM driver for the OOD labeling branch.
- `TAO-NET_STAGE_GUIDE.md`: end-to-end instructions for running both stages with the currently released assets.

Status
------

The codebase mirrors the version used in our submission. Once the review cycle finishes, the repo will be updated with:

1. Full configuration bundles (YAML/JSON) for every experiment.
2. Preprocessed data references and checksum manifests.
3. Reproducibility scripts (Docker/environment files) for public release.

Please watch this repository or reach out via the paper’s contact email if you need early access for artifact evaluation.
