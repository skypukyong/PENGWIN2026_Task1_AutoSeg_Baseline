"""Convert raw PENGWIN data into two nnUNetv2-style datasets.

Source label semantics:
    0          = background
    1   - 50   = sacrum fragments
    51  - 100  = left hipbone fragments
    101 - 150  = right hipbone fragments
    151 - 200  = femur fragments

Output 1: Dataset001_PENGWIN_Anatomical  (5-class semantic segmentation)
    labels: 0=bg, 1=sacrum, 2=leftHip, 3=rightHip, 4=femur
    Can be fed directly to nnUNetv2.

Output 2: Dataset002_PENGWIN_Frac        (per-bone instance segmentation prep)
    Each source case is split into up to 4 cases:
        <pid>_sacrum / _leftHip / _rightHip / _femur
    For each per-bone case, the image is masked to keep only voxels of that bone
    (other voxels filled with FRAC_BG_VALUE so nnUNet auto-crops to the bone),
    and instance labels within the bone are re-numbered from 1 (0, 1, 2, ...).
    Labels are written to ``labelsTr_instance/``; run ``gen_CSM_dataset.py``
    afterwards to produce the 3-class CSM labels in ``labelsTr/``.

Usage:
    python preprocessing/gen_nnunet_dataset.py \
        --src ./raw_data/PENGWIN_train \
        --out ./data
"""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


# ================= Default configuration =================

DEFAULT_SRC_DIR  = "./raw_data/PENGWIN_train"
DEFAULT_OUT_ROOT = "./data"
CASE_PREFIX      = "PENGWIN"
CHANNEL_NAMES    = {"0": "CT"}
SKIP_EXISTING    = True

# Value used to fill non-bone voxels in the Frac dataset images
# (CT background; nnUNet will auto-crop to the nonzero bone region)
FRAC_BG_VALUE = 0

# (bone_name, label_lo, label_hi, anatomical_class_id)
BONE_GROUPS = [
    ("sacrum",   1,   50,  1),
    ("leftHip",  51,  100, 2),
    ("rightHip", 101, 150, 3),
    ("femur",    151, 200, 4),
]

ANATOMICAL_NAME = "Dataset001_PENGWIN_Anatomical"
FRAC_NAME       = "Dataset002_PENGWIN_Frac"

# =========================================================


def write_image_like(arr: np.ndarray, ref: sitk.Image, out_path: Path, dtype):
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ref)
    img = sitk.Cast(img, dtype)
    sitk.WriteImage(img, str(out_path), useCompression=True)


def write_image_passthrough(img: sitk.Image, out_path: Path):
    sitk.WriteImage(img, str(out_path), useCompression=True)


def to_anatomical(label_arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(label_arr, dtype=np.uint8)
    for _name, lo, hi, cls in BONE_GROUPS:
        out[(label_arr >= lo) & (label_arr <= hi)] = cls
    return out


def to_frac_instance(label_arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Extract instances within [lo, hi] and renumber sequentially from 1.

    Returns an all-zero array if no voxel falls in the range.
    """
    mask = (label_arr >= lo) & (label_arr <= hi)
    out = np.zeros_like(label_arr, dtype=np.uint8)
    if not mask.any():
        return out
    for new_id, old_id in enumerate(np.unique(label_arr[mask]), start=1):
        out[label_arr == old_id] = new_id
    return out


def ensure_dirs(root: Path, label_subdir: str):
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / label_subdir).mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", default=DEFAULT_SRC_DIR,
                   help=f"Directory holding <case_id>/{{image.mha,label.mha}} "
                        f"(default: {DEFAULT_SRC_DIR})")
    p.add_argument("--out", default=DEFAULT_OUT_ROOT,
                   help=f"Output root where the two Dataset folders are created "
                        f"(default: {DEFAULT_OUT_ROOT})")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-write outputs even if they already exist")
    return p.parse_args()


def main():
    args = parse_args()
    skip_existing = SKIP_EXISTING and not args.overwrite

    src = Path(args.src)
    out_root = Path(args.out)
    anatomical_dir = out_root / ANATOMICAL_NAME
    frac_dir       = out_root / FRAC_NAME

    cases = sorted(p for p in src.iterdir() if p.is_dir())
    if not cases:
        raise RuntimeError(f"No case sub-folder found under {src}")

    ensure_dirs(anatomical_dir, "labelsTr")
    ensure_dirs(frac_dir, "labelsTr_instance")

    ana_count = 0
    frac_count = 0

    for case in tqdm(cases, desc="Converting"):
        cid = case.name
        image_src = case / "image.mha"
        label_src = case / "label.mha"
        if not image_src.exists() or not label_src.exists():
            print(f"WARN  skipping {cid}: image.mha or label.mha is missing")
            continue

        image_sitk = sitk.ReadImage(str(image_src))
        label_sitk = sitk.ReadImage(str(label_src))
        image_arr  = sitk.GetArrayFromImage(image_sitk)
        label_arr  = sitk.GetArrayFromImage(label_sitk).astype(np.int32)

        # ---- Anatomical dataset (5 classes) ----
        ana_image_path = anatomical_dir / "imagesTr" / f"{CASE_PREFIX}_{cid}_0000.nii.gz"
        ana_label_path = anatomical_dir / "labelsTr" / f"{CASE_PREFIX}_{cid}.nii.gz"

        if not (skip_existing and ana_image_path.exists()):
            write_image_passthrough(image_sitk, ana_image_path)
        if not (skip_existing and ana_label_path.exists()):
            write_image_like(to_anatomical(label_arr), label_sitk,
                             ana_label_path, dtype=sitk.sitkUInt8)
        ana_count += 1

        # ---- Frac dataset (per-bone, instance labels) ----
        for name, lo, hi, _cls in BONE_GROUPS:
            frac_arr = to_frac_instance(label_arr, lo, hi)
            if not frac_arr.any():
                continue  # this bone is absent in this case

            stem = f"{CASE_PREFIX}_{cid}_{name}"
            img_path = frac_dir / "imagesTr" / f"{stem}_0000.nii.gz"
            lbl_path = frac_dir / "labelsTr_instance" / f"{stem}.nii.gz"

            if not (skip_existing and img_path.exists()):
                # Keep only this bone's voxels; fill the rest with FRAC_BG_VALUE
                masked = np.where(frac_arr > 0, image_arr, FRAC_BG_VALUE).astype(image_arr.dtype)
                masked_sitk = sitk.GetImageFromArray(masked)
                masked_sitk.CopyInformation(image_sitk)
                sitk.WriteImage(masked_sitk, str(img_path), useCompression=True)
            if not (skip_existing and lbl_path.exists()):
                write_image_like(frac_arr, label_sitk, lbl_path, dtype=sitk.sitkUInt8)
            frac_count += 1

    # ---- dataset.json ----
    ana_json = {
        "channel_names": CHANNEL_NAMES,
        "labels": {
            "background": 0,
            "sacrum": 1,
            "leftHip": 2,
            "rightHip": 3,
            "femur": 4,
        },
        "numTraining": ana_count,
        "file_ending": ".nii.gz",
        "name": anatomical_dir.name,
        "description": "PENGWIN anatomical 5-class semantic segmentation",
        "source_dir": str(src),
    }
    with open(anatomical_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(ana_json, f, indent=2, ensure_ascii=False)

    # The Frac dataset's final training labels are produced by gen_CSM_dataset.py.
    # The labels block here describes the post-CSM semantics.
    frac_json = {
        "channel_names": CHANNEL_NAMES,
        "labels": {
            "background": 0,
            "foreground": 1,
            "contact": 2,
        },
        "numTraining": frac_count,
        "file_ending": ".nii.gz",
        "name": frac_dir.name,
        "description": "PENGWIN per-bone instance labels (run gen_CSM_dataset.py to produce CSM)",
        "source_dir": str(src),
        "bone_groups": [name for name, *_ in BONE_GROUPS],
    }
    with open(frac_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(frac_json, f, indent=2, ensure_ascii=False)

    print(f"\nAnatomical: {ana_count} cases  -> {anatomical_dir}")
    print(f"Frac:       {frac_count} cases  -> {frac_dir}")
    print("Next (Frac): run gen_CSM_dataset.py to convert labelsTr_instance -> labelsTr (CSM).")


if __name__ == "__main__":
    main()
