"""Generate Contact-Surface-Map (CSM) labels from per-instance segmentation.

Input: an nnUNet-style ``labelsTr_instance/`` folder where each ``.nii.gz``
contains integer instance IDs (0 = background, 1..N = distinct instances).

Output: an ``labelsTr/`` folder where each label is remapped to 3 classes:
    0 = background
    1 = foreground (non-contact instance interior)
    2 = contact surface between any two different instances

The contact mask is computed with a 3D neighbourhood shift on the GPU when
CUDA is available; falls back to CPU otherwise.

Usage:
    python preprocessing/gen_CSM_dataset.py \
        --input  data/Dataset002_PENGWIN_Frac/labelsTr_instance \
        --output data/Dataset002_PENGWIN_Frac/labelsTr \
        --kernel 7
"""

import argparse
import os

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm


def calculate_contact_voxels(array, bg_val=0, kernel_size=3, device="cuda"):
    """Detect voxels that lie on the contact surface between different instances.

    Implementation: shift the volume within the kernel neighbourhood and mark a
    voxel as a contact voxel if any neighbour belongs to a *different*
    non-background instance. Runs as 3D tensor ops on GPU for speed.
    """
    if array.ndim != 3:
        raise ValueError("input array must be 3-D")

    tensor = torch.from_numpy(array.astype(np.int32)).to(device=device)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    pad_size = kernel_size // 2
    padded = F.pad(tensor, (pad_size,) * 6, mode="constant", value=bg_val)

    D, H, W = tensor.shape[2:]
    mask = torch.zeros_like(tensor, dtype=torch.bool, device=device)

    for dx in range(-pad_size, pad_size + 1):
        for dy in range(-pad_size, pad_size + 1):
            for dz in range(-pad_size, pad_size + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                shifted = padded[:, :,
                                 pad_size + dx: pad_size + dx + D,
                                 pad_size + dy: pad_size + dy + H,
                                 pad_size + dz: pad_size + dz + W]
                # contact voxel: foreground here, foreground there, different IDs
                mask |= (tensor != bg_val) & (shifted != bg_val) & (shifted != tensor)

    return mask.squeeze().cpu().numpy()


def compute_CSM(seg_label_3d, bg_val=0, kernel_size=7):
    """Build a 3-class CSM label volume from an instance-label volume.

    Label semantics:
        0 = background
        1 = foreground (any non-background instance)
        2 = contact surface
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    contact_voxels = calculate_contact_voxels(
        seg_label_3d,
        bg_val=bg_val,
        kernel_size=kernel_size,
        device=device,
    )

    out = np.full(seg_label_3d.shape, bg_val, dtype=np.uint8)
    out[seg_label_3d > bg_val] = 1
    out[contact_voxels] = 2
    return out


def batch_generate_csm(input_dir, output_dir, kernel_size=7, bg_val=0):
    """Iterate every ``.nii.gz`` in ``input_dir`` and write CSM labels to ``output_dir``."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"created output dir: {output_dir}")

    files = [f for f in os.listdir(input_dir)
             if f.endswith(".nii.gz") and not f.startswith("._")]

    if len(files) == 0:
        print(f"WARN  no .nii.gz files found in {input_dir}")
        return

    print(f"processing {len(files)} files  |  kernel_size={kernel_size}")

    for file_name in tqdm(files, desc="CSM"):
        file_path   = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)  # keep filename (nnUNet convention)

        try:
            itk_img = sitk.ReadImage(file_path)
            seg_label = sitk.GetArrayFromImage(itk_img)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # avoid OOM on large volumes

            seg_csm = compute_CSM(seg_label, bg_val=bg_val, kernel_size=kernel_size)

            out_img = sitk.GetImageFromArray(seg_csm)
            out_img.CopyInformation(itk_img)  # preserve spacing / origin / direction
            sitk.WriteImage(out_img, output_path)

        except Exception as e:
            print(f"ERROR  failed on {file_name}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input",  default="data/Dataset002_PENGWIN_Frac/labelsTr_instance",
                   help="input directory of per-instance .nii.gz labels")
    p.add_argument("--output", default="data/Dataset002_PENGWIN_Frac/labelsTr",
                   help="output directory for 3-class CSM labels")
    p.add_argument("--kernel", type=int, default=7,
                   help="odd kernel size for contact detection (3/5/7/9; default 7)")
    p.add_argument("--bg",     type=int, default=0,
                   help="background label value (default 0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_generate_csm(
        input_dir=args.input,
        output_dir=args.output,
        kernel_size=args.kernel,
        bg_val=args.bg,
    )
    print("\nDone: all CSM labels generated.")
