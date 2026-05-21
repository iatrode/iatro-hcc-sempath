# HCC-SemPath TODO

## Data Contract

- Keep training and evaluation IAC-only:
  - `image_tile_package_path`
  - `teacher_feature_package_path`
- Keep loose PNG tiles and `.npy` feature files as build/debug intermediates only.
- Move data/preprocessing modules into clearer subpackages after the current pipeline stabilizes:
  - `data/iatrocache.py`
  - `data/image_tile_cache.py`
  - `data/feature_cache.py`
  - `preprocessing/tiling.py`

## Compression

- Use real WSI package QC contact sheets in smoke runs.
- Current TCGA-MR-A520 result:
  - lossless JXL IAC: 255.47 MB
  - JXL distance 1.0: 41.91 MB
  - JXL distance 2.0: 30.86 MB
  - JXL distance 3.0: 22.74 MB
- GigaPath local 256-tile MPS drift confirms GigaPath is compression-sensitive:
  - distance 1.0: feature cosine mean 0.992857
  - distance 2.0: feature cosine mean 0.975004
  - distance 3.0: feature cosine mean 0.929425
- Provisional policy:
  - GigaPath teacher cache generation: use distance 1.0.
  - distance 2.0 and 3.0 are compression-experiment controls only, not formal data delivery formats.
  - Publication freeze: include lossless or distance 1.0 packages for benchmark-critical subsets.

## Teacher Targets

- Short term:
  - Use local GigaPath weights at `/Volumes/MacDataHD/DevX/2024-CT-WSI/model/prov-gigapath`.
  - Built `data/test_svs_full_mtf03/teacher_features_gigapath_d1.iac` from d1.0 tiles:
    - records: 3207
    - feature_dim: 1536
    - generation device: MPS outside Codex sandbox
    - generation time: 9m34s
- H-optimus:
  - Requires Hugging Face gated repo approval/authentication.
  - After access, rerun compression drift and build `teacher_features_hoptimus.iac`.
- Multi-teacher training:
  - Do not force GigaPath and H-optimus into one shared output vector.
  - Use shared student backbone with teacher-specific heads.
  - Compare GigaPath-only, H-optimus-only, and multi-head student.

## Environment

- `hcc-sempath` conda env is intentionally minimal: conda manages Python, pip manages runtime wheels.
- `PYTHONNOUSERSITE=1` is set to avoid importing user-site packages from `~/.local`.
- MPS validation must run outside the Codex sandbox; sandboxed checks can report false-negative MPS availability.
- Verified outside sandbox: `torch.backends.mps.is_available() == True`, `torch.mps.device_count() == 1`.
