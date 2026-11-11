# Histology-derived asymmetric FODs

Python tools for estimating **symmetric and asymmetric fibre orientation distributions (FODs)** from 2D microscopy-derived orientation fields.

The method follows the histology-based asymmetric FOD construction described in Section 3.2 of [Bastiani et al., NeuroImage 2017](https://doi.org/10.1016/j.neuroimage.2017.06.050).

Unlike a conventional orientation histogram, the asymmetric FOD uses both the **orientation and spatial location** of vectors within a patch. This allows asymmetric fibre configurations such as bending, fanning and convergence/divergence to be represented without requiring axonal polarity.

## Input

A NIfTI (`.nii`/`.nii.gz`) 2D vector field:

```text
(nx, ny, 2)
```

where the final dimension contains the x/y components of the local fibre orientation.

## Usage

```bash
python histology_afod.py \
    --vectors structure_tensor_vectors.nii.gz \
    --centre 100 120 \
    --patch 21 21 \
    --output hist_afod.npz \
    --plot hist_afod.png
```

The output contains the symmetric FOD (`sfod`), asymmetric FOD (`afod`), their even/odd components, and an asymmetry index (`odd_power_fraction`).

## Dependencies

```bash
pip install numpy nibabel matplotlib
```

## Reference

Bastiani M, et al. **Improved tractography using asymmetric fibre orientation distributions.** *NeuroImage*. 2017;158:205-218. https://doi.org/10.1016/j.neuroimage.2017.06.050
