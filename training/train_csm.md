# Training — CSM (Contact Surface Map) segmentation

Dataset: `Dataset002_PENGWIN_Frac`
Labels: `0=background, 1=foreground, 2=contact`
Each source case is split into per-bone cases: `PENGWIN_<pid>_{sacrum,leftHip,rightHip,femur}`

---

## Prerequisites

1. **Generate the per-instance dataset**

   ```bash
   python preprocessing/gen_nnunet_dataset.py
   ```

   Produces `data/Dataset002_PENGWIN_Frac/{imagesTr, labelsTr_instance, dataset.json}`. The `imagesTr` volumes are already masked to the foreground voxels of the corresponding bone.

2. **Generate CSM labels**

   ```bash
   python preprocessing/gen_CSM_dataset.py
   ```

   Converts `labelsTr_instance/` (multi-instance integers) into `labelsTr/` (3-class CSM).

3. **Environment**

   ```bash
   conda activate nnunet
   export nnUNet_raw=/path/to/nnUNet_raw
   export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
   export nnUNet_results=/path/to/nnUNet_results
   ```

---

## Step 1 — Copy the dataset into `nnUNet_raw`

```bash
cp -r data/Dataset002_PENGWIN_Frac "${nnUNet_raw}/Dataset002_PENGWIN_Frac"
```

Direct physical copy; nnUNet operations will not affect the original files in `data/`. If disk space is tight, you may omit `labelsTr_instance/` (only `imagesTr/`, `labelsTr/`, and `dataset.json` are required for training).

## Step 2 — Plan & preprocess

```bash
nnUNetv2_plan_and_preprocess -d 002 --verify_dataset_integrity
```

Integrity checks compare spacing/shape between every `imagesTr/<case>_0000.nii.gz` and `labelsTr/<case>.nii.gz`. Geometric metadata is preserved by `SimpleITK.Image.CopyInformation` in both preprocessing scripts, so this step is expected to pass.

## Step 3 — Train

```bash
nnUNetv2_train 002 3d_fullres 0 --npz
```

5-fold:

```bash
for f in 0 1 2 3 4; do
    nnUNetv2_train 002 3d_fullres ${f} --npz
done
```

Common variants (same as Anatomical):

| Goal | Command |
|------|---------|
| 2D model              | `nnUNetv2_train 002 2d 0 --npz` |
| Low resolution        | `nnUNetv2_train 002 3d_lowres 0 --npz` |
| Shorter (250 epochs)  | `nnUNetv2_train 002 3d_fullres 0 -tr nnUNetTrainer_250epochs --npz` |
| Resume training       | append `--c` |

## Step 4 — Best configuration + inference

```bash
nnUNetv2_find_best_configuration 002 -c 3d_fullres

nnUNetv2_predict \
    -i <input_dir> \
    -o <output_dir> \
    -d 002 \
    -c 3d_fullres \
    -f 0 1 2 3 4
```

---

## Notes

- Each source case is split into up to 4 per-bone cases, so `numTraining` is much larger than the number of source patients. nnUNet's default 5-fold split treats those per-bone cases independently. If you want all 4 bones of a patient to fall into the same fold, write a custom `splits_final.json` under `${nnUNet_preprocessed}/Dataset002_PENGWIN_Frac/`.
- To collapse CSM to a plain binary (background vs any-foreground) task: remove the contact-assignment line in `preprocessing/gen_CSM_dataset.py` (`out[contact_voxels] = 2`) and update `dataset.json` to `"labels": {"background": 0, "foreground": 1}` before preprocessing.
