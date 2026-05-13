# Inference

Post-processing step that turns the CSM model's **semantic** predictions into per-fragment **instance** segmentations. Combined with the Anatomical model's per-bone masks, this completes the full Task 1 pipeline (CT → fragment instance labels with anatomical bone class).

---

## Where this fits in the pipeline

```
       ┌────────────────────────────┐
CT ──► │ Anatomical model (Dataset001)│ ──► 5-class anatomy
       │   0=bg, 1=sacrum, 2=leftHip,│      mask per voxel
       │   3=rightHip, 4=femur       │
       └─────────────┬──────────────┘
                     │ for each bone class
                     ▼
       ┌────────────────────────────┐
       │  mask the CT to that bone  │
       │  (zero out other voxels)   │
       └─────────────┬──────────────┘
                     ▼
       ┌────────────────────────────┐
       │ CSM model (Dataset002)      │ ──► 3-class CSM
       │   0=bg, 1=foreground,       │      .nii.gz per bone
       │   2=contact                 │
       └─────────────┬──────────────┘
                     ▼
       ┌────────────────────────────┐
       │  inference/frac_to_instance │ ──► instance IDs 1..N
       │  (this script)              │      .nii.gz per bone
       └─────────────┬──────────────┘
                     ▼
       ┌────────────────────────────┐
       │  merge bones, offset IDs    │ ──► challenge submission
       │  into PENGWIN ranges        │      (sacrum 1-50, leftHip 51-100,
       │  (user / external script)   │       rightHip 101-150, femur 151-200)
       └────────────────────────────┘
```

`frac_to_instance.py` covers the boxed step. The final ID-range offset stage is dataset-specific and is left to the user.

---

## What `frac_to_instance.py` does

**Input:** A directory of CSM predictions saved by `nnUNetv2_predict`. Each `.nii.gz` is an integer volume with values:

| value | meaning |
|-------|---------|
| 0     | background |
| 1     | fragment core / boundary (foreground) |
| 2     | contact surface between two fragments (CSM) |

**Output:** A directory of the same filenames, where each voxel carries a positive integer **fragment instance ID** (background remains 0). IDs are renumbered `1..N` per file, ordered by descending fragment volume.

**Algorithm (4 stages):**

1. **Core extraction.** Connected components on `label == 1` give candidate fragment cores. Components smaller than `--ccf_threshold` voxels are discarded as noise. The survivors are renumbered by descending volume.
2. **Unambiguous CSM absorption.** Each connected CSM region (`label == 2`) is dilated by `--kernel_size`. If the dilated region touches *exactly one* core, that CSM strip is merged into the core (resolves the trivial boundary between a fragment and background).
3. **Renumber.** Cores left after step 2 are re-thresholded and renumbered.
4. **KD-Tree nearest-neighbour fill.** Any remaining `1` (boundary) and `2` (true inter-fragment contact) voxels are assigned to their nearest core via a KD-Tree on the core voxel coordinates. This produces dense fragment volumes touching each other along the CSM line, exactly as required for the instance segmentation output.

The script falls back to CPU if `--device cuda` is requested but unavailable.

---

## Usage

```bash
python inference/frac_to_instance.py \
    -i  <csm_predictions_dir> \
    -o  <instance_output_dir> \
    -k  5 \
    -c  100 \
    --device cuda
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-i, --input_dir`    | *required* | Directory of CSM `.nii.gz` predictions (3-class). |
| `-o, --output_dir`   | *required* | Where to write instance `.nii.gz` files. |
| `-k, --kernel_size`  | `5`        | Odd 3-D dilation kernel used in step 2. Larger ⇒ more aggressive CSM absorption. |
| `-c, --ccf_threshold`| `100`      | Minimum voxel count for a core to survive (steps 1 & 3). |
| `--device`           | `cuda`     | `cuda` or `cpu` for the dilation step. |

---

## End-to-end example

Assuming both models are trained (see [`../training/`](../training/)) and `${nnUNet_results}` is set:

```bash
# 1) Anatomical predictions
nnUNetv2_predict \
    -i  /path/to/test_CTs \
    -o  ./predictions/anatomical \
    -d  001 -c 3d_fullres -f 0 1 2 3 4

# 2) For each test CT, mask to a single bone (loop in your own script).
#    The simplest case: assume the test set is already split per-bone
#    using preprocessing/gen_nnunet_dataset.py (in which case the CSM
#    predictions in step 3 are directly per-bone).

# 3) CSM predictions on the per-bone masked CT volumes
nnUNetv2_predict \
    -i  /path/to/test_per_bone_CTs \
    -o  ./predictions/csm \
    -d  002 -c 3d_fullres -f 0 1 2 3 4

# 4) Semantic CSM -> instance IDs
python inference/frac_to_instance.py \
    -i ./predictions/csm \
    -o ./predictions/instance \
    -k 5 -c 100 --device cuda

# 5) (User) Combine per-bone instance maps into a single volume per case
#    and offset IDs into the PENGWIN anatomical ranges:
#      sacrum   1..50
#      leftHip  51..100
#      rightHip 101..150
#      femur    151..200
```

---

## Tuning notes

- **`kernel_size` (default 5).** Controls how far the CSM strip is allowed to "reach back" to a single core in step 2. Increase to merge thinner contact lines into cores; decrease if cores are leaking into neighbours.
- **`ccf_threshold` (default 100).** Small enough to keep distal fragments visible, large enough to reject speckle. The right value depends on voxel spacing — for ~0.5 mm CT, 100 voxels ≈ 12 mm³.
- The script is **deterministic** for a given input and parameters.
- GPU acceleration only speeds up step 2; steps 1, 3, 4 run on CPU. For typical pelvic CTs the whole script finishes in seconds per volume.
