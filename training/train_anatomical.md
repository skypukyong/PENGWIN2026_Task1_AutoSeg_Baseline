# Training — Anatomical 5-class semantic segmentation

Dataset: `Dataset001_PENGWIN_Anatomical`
Labels: `0=background, 1=sacrum, 2=leftHip, 3=rightHip, 4=femur`

---

## Prerequisites

- `python preprocessing/gen_nnunet_dataset.py` has been run, producing
  ```
  data/Dataset001_PENGWIN_Anatomical/
  ├── imagesTr/      PENGWIN_<pid>_0000.nii.gz
  ├── labelsTr/      PENGWIN_<pid>.nii.gz       (values 0..4)
  └── dataset.json
  ```
- The conda env created by `install.sh` is activated:
  ```bash
  conda activate nnunet
  ```
- The three nnUNetv2 path variables are exported. Adjust the paths to your machine:
  ```bash
  export nnUNet_raw=/path/to/nnUNet_raw
  export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
  export nnUNet_results=/path/to/nnUNet_results
  ```

---

## Step 1 — Copy the dataset into `nnUNet_raw`

```bash
cp -r data/Dataset001_PENGWIN_Anatomical "${nnUNet_raw}/Dataset001_PENGWIN_Anatomical"
```

nnUNetv2 identifies datasets by the integer ID embedded in the folder name (`Dataset<ID>_<Name>`); here we use `001`. A direct copy is recommended so that any internal changes made by nnUNet do not affect the original files under `data/`.

## Step 2 — Plan & preprocess

```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

This writes `nnUNetPlans.json` and preprocessed tensors for each configuration (`2d`, `3d_lowres`, `3d_fullres`) into `${nnUNet_preprocessed}/Dataset001_PENGWIN_Anatomical/`.

## Step 3 — Train

Single fold (fold 0):

```bash
nnUNetv2_train 001 3d_fullres 0 --npz
```

Full 5-fold cross-validation:

```bash
for f in 0 1 2 3 4; do
    nnUNetv2_train 001 3d_fullres ${f} --npz
done
```

Common variants:

| Goal | Command |
|------|---------|
| Lower memory footprint | `nnUNetv2_train 001 3d_lowres 0 --npz` |
| 2D model               | `nnUNetv2_train 001 2d 0 --npz` |
| Shorter (250 epochs)   | `nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainer_250epochs --npz` |
| Resume training        | append `--c` |

Checkpoints live under `${nnUNet_results}/Dataset001_PENGWIN_Anatomical/`.

## Step 4 — Pick the best configuration

```bash
nnUNetv2_find_best_configuration 001 -c 3d_fullres
```

## Step 5 — Inference

```bash
nnUNetv2_predict \
    -i <input_dir> \
    -o <output_dir> \
    -d 001 \
    -c 3d_fullres \
    -f 0 1 2 3 4
```
