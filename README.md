# PENGWIN 2026 — Task 1 Baseline

A reference baseline for **Task 1** of the
[PENGWIN 2026 Challenge](https://pengwin2026.grand-challenge.org/):
automatic segmentation of pelvic bone fragments in CT.

Training data is the official release on Zenodo:
**[zenodo.org/records/19732767](https://zenodo.org/records/19732767)**.

The baseline trains two complementary [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet)
models that can be combined to produce per-fragment instance segmentation.

---

## Approach

The raw label volume uses an integer encoding that packs both anatomy and instance:

| Range | Anatomical class | Meaning |
|-------|------------------|---------|
| 0          | —          | background |
| 1 – 50     | sacrum     | up to 50 sacrum fragments |
| 51 – 100   | left hip   | up to 50 left hipbone fragments |
| 101 – 150  | right hip  | up to 50 right hipbone fragments |
| 151 – 200  | femur      | up to 50 femur fragments |

We factor the problem into two segmentation stages plus a deterministic
post-processing step:

1. **Anatomical 5-class segmentation** (`Dataset001_PENGWIN_Anatomical`)
   `0=bg, 1=sacrum, 2=leftHip, 3=rightHip, 4=femur`
   A single nnUNetv2 model maps the whole CT volume to anatomy.

2. **CSM (Contact-Surface Map) 3-class segmentation** (`Dataset002_PENGWIN_Frac`)
   `0=bg, 1=foreground, 2=contact`
   Each source case is split into up to four per-bone cases. For each per-bone
   case, the CT is masked to that bone's voxels (so nnUNet auto-crops to a tight
   bounding box) and the contact-surface map is computed from instance labels.
   A second nnUNetv2 model then learns to separate touching fragments by
   predicting the contact ribbon between them.

3. **CSM → instance post-processing** (`inference/frac_to_instance.py`)
   Connected-component analysis on the CSM prediction, plus KD-Tree
   nearest-neighbour assignment, turns the 3-class semantic mask into per-fragment
   instance IDs.

At inference time, stage (1) yields per-bone masks; stage (2) labels each masked
bone with its CSM; stage (3) reads the CSM and emits per-fragment instances.
A final user-side step merges per-bone results and offsets the instance IDs into
the official label ranges (sacrum 1–50, leftHip 51–100, rightHip 101–150,
femur 151–200).

---

## Repository layout

```
AutoSeg-Baseline/
├── install.sh                       # conda env + dependency installer
├── preprocessing/
│   ├── gen_nnunet_dataset.py        # raw .mha  -> two nnUNetv2 datasets
│   └── gen_CSM_dataset.py           # instance labels -> 3-class CSM labels
├── training/
│   ├── train_anatomical.md          # nnUNetv2 training/inference recipe (anatomical)
│   └── train_csm.md                 # nnUNetv2 training/inference recipe (CSM)
├── inference/
│   ├── frac_to_instance.py          # CSM 3-class predictions -> instance IDs
│   └── README.md                    # end-to-end inference pipeline doc
├── data/                            # generated; .gitignored
│   ├── Dataset001_PENGWIN_Anatomical/
│   └── Dataset002_PENGWIN_Frac/
└── raw_data/                        # downloaded; .gitignored
    └── PENGWIN_train/<case_id>/{image.mha,label.mha}
```

---

## Quick start

```bash
# 0. Clone
git clone <repo-url> AutoSeg-Baseline
cd AutoSeg-Baseline

# 1. Install the conda env + nnUNetv2 + helpers
bash install.sh
conda activate nnunet

# 2. Download the training data from Zenodo and place it under raw_data/
#    https://zenodo.org/records/19732767
#    Expected layout:
#      raw_data/PENGWIN_train/<case_id>/{image.mha,label.mha}

# 3. Convert raw .mha into two nnUNetv2-style datasets under data/
python preprocessing/gen_nnunet_dataset.py \
    --src raw_data/PENGWIN_train \
    --out data

# 4. Generate the CSM labels for the per-bone dataset
python preprocessing/gen_CSM_dataset.py \
    --input  data/Dataset002_PENGWIN_Frac/labelsTr_instance \
    --output data/Dataset002_PENGWIN_Frac/labelsTr \
    --kernel 7

# 5. Point nnUNetv2 at storage you control (adjust paths)
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

# 6. Train both models  (see the docs below for full options)
#    Anatomical:
cp -r data/Dataset001_PENGWIN_Anatomical "${nnUNet_raw}/"
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
nnUNetv2_train 001 3d_fullres 0 --npz

#    CSM:
cp -r data/Dataset002_PENGWIN_Frac "${nnUNet_raw}/"
nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity
nnUNetv2_train 002 3d_fullres 0 --npz

# 7. Inference  (after training each model with 5 folds, omit -f for just fold 0)
nnUNetv2_predict -i <test_CTs>           -o ./pred/anatomical \
    -d 001 -c 3d_fullres -f 0 1 2 3 4
nnUNetv2_predict -i <test_per_bone_CTs>  -o ./pred/csm \
    -d 002 -c 3d_fullres -f 0 1 2 3 4

# 8. CSM semantic predictions -> per-fragment instance IDs
python inference/frac_to_instance.py \
    -i ./pred/csm -o ./pred/instance \
    -k 5 -c 100 --device cuda
```

Full recipes (5-fold, alternate configurations, full inference pipeline) are in
[`training/train_anatomical.md`](training/train_anatomical.md),
[`training/train_csm.md`](training/train_csm.md), and
[`inference/README.md`](inference/README.md).

---

## Data

The Zenodo archive contains the training set in `.mha` format:

```
PENGWIN_train/
├── 001/{image.mha, label.mha}
├── 002/{image.mha, label.mha}
└── ...
```

`image.mha` is a CT volume; `label.mha` is an integer volume using the
anatomy + instance encoding described in the table above.

`preprocessing/gen_nnunet_dataset.py` converts the archive into:

- `data/Dataset001_PENGWIN_Anatomical/` — full CT + 5-class anatomical labels
- `data/Dataset002_PENGWIN_Frac/` — per-bone masked CT + per-instance labels
  (sequential numbering, 1..N, within each bone)

---

## Preprocessing scripts — CLI reference

```text
python preprocessing/gen_nnunet_dataset.py [--src DIR] [--out DIR] [--overwrite]

  --src        Root holding <case_id>/{image.mha,label.mha}
               (default: ./raw_data/PENGWIN_train)
  --out        Output root where the two Dataset folders are created
               (default: ./data)
  --overwrite  Re-write outputs even if they already exist
```

```text
python preprocessing/gen_CSM_dataset.py [--input DIR] [--output DIR] [--kernel N] [--bg V]

  --input   Per-instance .nii.gz labels
            (default: data/Dataset002_PENGWIN_Frac/labelsTr_instance)
  --output  3-class CSM labels
            (default: data/Dataset002_PENGWIN_Frac/labelsTr)
  --kernel  Odd kernel size for contact detection (3/5/7/9; default 7)
  --bg      Background label value (default 0)
```

```text
python inference/frac_to_instance.py -i DIR -o DIR [-k N] [-c N] [--device cuda|cpu]

  -i, --input_dir     Directory of CSM (3-class) prediction .nii.gz files (required)
  -o, --output_dir    Directory for per-fragment instance .nii.gz files (required)
  -k, --kernel_size   3D dilation kernel for CSM absorption (default 5)
  -c, --ccf_threshold Minimum voxels for a core to survive (default 100)
  --device            cuda or cpu (default cuda, falls back to cpu if unavailable)
```

---

## Hardware

- nnUNetv2 `3d_fullres` typically needs a GPU with **≥ 12 GB** of memory.
  For smaller GPUs, swap in `3d_lowres` or `2d` as documented in the training recipes.
- `gen_CSM_dataset.py` will use GPU if available (CUDA) and otherwise fall back to CPU.

---

## Citation

If you use this baseline, please cite the PENGWIN 2026 challenge and the Zenodo
training set linked above, in addition to nnU-Net:

> Isensee, F., Jaeger, P.F., Kohl, S.A.A. *et al.* nnU-Net: a self-configuring
> method for deep learning-based biomedical image segmentation.
> *Nat Methods* **18**, 203–211 (2021).
