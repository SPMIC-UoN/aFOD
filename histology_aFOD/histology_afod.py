#!/usr/bin/env python3
"""Estimate 2D histology-derived symmetric and asymmetric FODs from NIfTI.

Based on Bastiani et al. (2017), NeuroImage 158:205-218, Section 3.2,
DOI: 10.1016/j.neuroimage.2017.06.050.

For each angular bin u, unoriented local fibre axes matching u are counted
only if they lie in the corresponding half of the patch:

    (sample_position - patch_centre) . u > 0.

The opposite bin -u gets contributions from the other half. Samples on
that dividing plane contribute half to each bin. The angular histogram is
then Gaussian-smoothed (default: 23 degrees FWHM, as in Bastiani et al.).
No axon polarity is needed: reversing any input vector leaves the output unchanged.


EXAMPLE:
python histology_afod.py --vectors ./structure_tensor_vectors.nii.gz \
                         --centre 300 300 --patch 601 601 \
                         --output ./hist_afod.npz --plot ./hist_afod.png

File arguments:
    --vectors  INPUT  : NIfTI image (.nii or .nii.gz). Accepted NIfTI data shapes: (nx, ny, 2) or (nx, ny, 1, 2). Zero-length vectors are excluded automatically.
    --centre   INPUT  : Choose a zero-based centre pixel (cx,cy) 
    --patch    INPUT  : ODD patch dimensions (xdim,ydim) around the centre pixel
    --output   OUTPUT : NumPy archive (.npz) containing the estimated FODs
    --plot (Optional) OUTPUT : PNG image (.png) showing the FOD glyphs

Outputs in the compressed .npz file:
    directions            (D,2) unit directions sampled on the circle for the histograms, with directions[i+D/2] = -directions[i]
    angles                (D,) angles in rads corresponding to the directions
    afod                  (D,) Gaussian-smoothed asymmetric angular density (integral ~1)
    sfod                  (D,) Gaussian-smoothed symmetric angular density (integral ~1)
    even_fod, odd_fod     (D,) antipodally symmetric/asymmetric (i.e. even/odd) parts of afod
    odd_power_fraction    scalar, sum(odd_fod**2) / sum(afod**2)
    centre, patch, n_valid, fwhm_deg


METHOD
------
1. Treat each structure-tensor vector as an unoriented axis.
2. For each orientation, assign contributions according to which half of the
   patch the pixel lies in; the separating line runs through the patch centre
   and is perpendicular to that orientation (Bastiani et al., Section 3.2).
3. Smooth the 360-bin circular histograms with a 23-degree FWHM Gaussian.
4. Save symmetric/asymmetric FODs and their even/odd decomposition.


DEPENDENCIES
------------
    pip install numpy nibabel matplotlib
    # matplotlib is only needed when using --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_nifti_vectors(filename: str) -> np.ndarray:
    """Load the INPUT structure-tensor vector field from NIfTI (.nii/.nii.gz), not NPZ."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Install NIfTI support with: pip install nibabel") from exc

    if not filename.lower().endswith((".nii", ".nii.gz")):
        raise ValueError("Input vector field must be a .nii or .nii.gz NIfTI file")
    data = np.asarray(nib.load(filename).dataobj, dtype=np.float64)
    if data.ndim == 4 and data.shape[2] == 1 and data.shape[3] == 2:
        data = data[:, :, 0, :]
    elif data.ndim == 5 and data.shape[2:4] == (1, 1) and data.shape[4] == 2:
        data = data[:, :, 0, 0, :]
    if data.ndim != 3 or data.shape[-1] != 2:
        raise ValueError(
            f"Expected NIfTI vector field (nx,ny,2) or (nx,ny,1,2); got {data.shape}"
        )
    return data


def smooth_circular_histogram(hist: np.ndarray, fwhm_deg: float) -> np.ndarray:
    """Convolve a periodic orientation histogram with an angular Gaussian."""
    n_bins = len(hist)
    sigma_deg = fwhm_deg / (2 * np.sqrt(2 * np.log(2)))
    offsets_deg = np.minimum(np.arange(n_bins), n_bins - np.arange(n_bins)) * (360 / n_bins)
    kernel = np.exp(-0.5 * (offsets_deg / sigma_deg) ** 2)
    kernel /= kernel.sum()
    return np.fft.ifft(np.fft.fft(hist) * np.fft.fft(kernel)).real


def estimate_histology_afod(
    vectors: np.ndarray,
    centre: tuple[int, int],
    patch: tuple[int, int],
    n_bins: int = 360,
    fwhm_deg: float = 23.0,
) -> dict[str, np.ndarray]:
    """Estimate 2D symmetric and asymmetric FODs from one rectangular patch.

    vectors: (nx,ny,2) array of unoriented 2D structure-tensor axes.
    centre:  (cx,cy), ZERO-BASED pixel coordinates.
    patch:   (xdim,ydim), positive odd integers; patch must lie in the image.
    """
    vectors = np.asarray(vectors)
    if vectors.ndim != 3 or vectors.shape[-1] != 2:
        raise ValueError("vectors must have shape (nx, ny, 2)")
    c = np.asarray(centre, dtype=int)
    p = np.asarray(patch, dtype=int)
    if c.shape != (2,) or p.shape != (2,):
        raise ValueError("centre and patch must each have two integers")
    if np.any(p <= 0) or np.any(p % 2 == 0):
        raise ValueError("patch dimensions must be positive odd integers")
    if n_bins < 8 or n_bins % 2 != 0:
        raise ValueError("n_bins must be even and >= 8")
    if not np.isfinite(fwhm_deg) or not 0 < fwhm_deg < 180:
        raise ValueError("fwhm_deg must be between 0 and 180")

    lo = c - p // 2
    hi = lo + p
    if np.any(lo < 0) or np.any(hi > vectors.shape[:2]):
        raise ValueError("Selected patch extends outside the NIfTI image")

    patch_vectors = vectors[lo[0]:hi[0], lo[1]:hi[1], :]
    offsets = np.stack(np.meshgrid(
        np.arange(p[0]) - p[0] // 2,
        np.arange(p[1]) - p[1] // 2,
        indexing="ij",
    ), axis=-1)
    # Equivalently: indices -patch//2 through +patch//2 for each axis.
    positions = offsets.reshape(-1, 2).astype(float)
    axes = patch_vectors.reshape(-1, 2).astype(float)
    norms = np.linalg.norm(axes, axis=1)
    valid = np.all(np.isfinite(axes), axis=1) & np.isfinite(norms) & (norms > 0)
    if not np.any(valid):
        raise ValueError("Selected patch contains no valid nonzero vectors")
    axes = axes[valid] / norms[valid, None]
    positions = positions[valid]

    theta = np.arange(n_bins) * (2 * np.pi / n_bins)
    directions = np.column_stack((np.cos(theta), np.sin(theta)))
    half = n_bins // 2

    # Treat each structure-tensor eigenvector as an AXIS (angle modulo pi),
    # independent of its arbitrary sign. Quantise to the nearest axial bin.
    axial_angle = np.mod(np.arctan2(axes[:, 1], axes[:, 0]), np.pi)
    axial_bin = np.rint(axial_angle * n_bins / (2 * np.pi)).astype(int) % half
    base_u = directions[axial_bin]

    # Bastiani Sec 3.2: assign orientations according to their LOCATION
    # in the half-patch defined by a line through the patch centre normal
    # to the sampled direction. Central-line samples split equally.
    side = np.sum(positions * base_u, axis=1)
    tol = 1e-10
    positive = side > tol
    negative = side < -tol
    boundary = ~(positive | negative)

    hist_asym = np.zeros(n_bins, dtype=float)
    hist_sym = np.zeros(n_bins, dtype=float)
    np.add.at(hist_asym, axial_bin[positive], 1.0)
    np.add.at(hist_asym, axial_bin[negative] + half, 1.0)
    np.add.at(hist_asym, axial_bin[boundary], 0.5)
    np.add.at(hist_asym, axial_bin[boundary] + half, 0.5)
    np.add.at(hist_sym, axial_bin, 0.5)
    np.add.at(hist_sym, axial_bin + half, 0.5)

    afod = smooth_circular_histogram(hist_asym, fwhm_deg)
    sfod = smooth_circular_histogram(hist_sym, fwhm_deg)
    dtheta = 2 * np.pi / n_bins
    afod /= afod.sum() * dtheta
    sfod /= sfod.sum() * dtheta

    opposite = np.roll(afod, half)
    even_fod = (afod + opposite) / 2
    odd_fod = (afod - opposite) / 2
    odd_power_fraction = float(np.sum(odd_fod**2) / np.sum(afod**2))

    return {
        "angles_rad": theta,
        "directions": directions,
        "afod": afod,
        "sfod": sfod,
        "even_fod": even_fod,
        "odd_fod": odd_fod,
        "odd_power_fraction": np.array(odd_power_fraction),
        "centre": c,
        "patch": p,
        "n_valid": np.array(len(axes)),
        "fwhm_deg": np.array(fwhm_deg),
    }


# Same FOD glyph colours are used by the CLI plot and the four-example figure.
SYMMETRIC_COLOR = "#3d80b9"
ASYMMETRIC_COLOR = "#d67454"


def fod_outline_xy(
    result: dict[str, np.ndarray],
    component: str,
    radial_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return closed 2D glyph-outline coordinates at a common radial scale."""
    if component not in ("sfod", "afod"):
        raise ValueError("component must be 'sfod' or 'afod'")
    if not np.isfinite(radial_scale) or radial_scale <= 0:
        raise ValueError("radial_scale must be a positive finite number")
    theta = np.asarray(result["angles_rad"], dtype=float)
    radius = np.asarray(result[component], dtype=float) / radial_scale
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.r_[x, x[0]], np.r_[y, y[0]]


def draw_fod_glyph(
    ax,
    result: dict[str, np.ndarray],
    component: str,
    radial_scale: float | None = None,
    *,
    color: str | None = None,
    fill_alpha: float = 0.75,
    axis_limit: float = 1.05,
    grid_color: str = "#d6e0e5",
) -> None:
    """Draw one filled 2D FOD glyph on an existing Matplotlib Axes.

    This is the SHARED plotting primitive used by plot_fods() and by the
    four-example demo. Supply the same radial_scale for symmetric and
    asymmetric glyphs so their amplitudes are directly comparable.
    """
    if radial_scale is None:
        radial_scale = max(float(result["sfod"].max()), float(result["afod"].max()))
    if color is None:
        color = SYMMETRIC_COLOR if component == "sfod" else ASYMMETRIC_COLOR
    x, y = fod_outline_xy(result, component, radial_scale)
    ax.fill(x, y, color=color, alpha=fill_alpha, linewidth=0)
    ax.plot(x, y, color=color, lw=1.8)
    ax.axhline(0, color=grid_color, lw=0.6, zorder=-1)
    ax.axvline(0, color=grid_color, lw=0.6, zorder=-1)
    ax.set(aspect="equal", xlim=(-axis_limit, axis_limit), ylim=(-axis_limit, axis_limit))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_fods(result: dict[str, np.ndarray], filename: str) -> None:
    """Save shared-style filled 2D symmetric/asymmetric glyphs plus an overlay."""
    import matplotlib.pyplot as plt

    scale = max(float(result["sfod"].max()), float(result["afod"].max()))
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.7))
    draw_fod_glyph(axes[0], result, "sfod", scale)
    draw_fod_glyph(axes[1], result, "afod", scale)
    axes[0].set_title("Symmetric FOD")
    axes[1].set_title("Asymmetric FOD")

    # Use the same closed-outline geometry as the filled glyphs.
    for key, color, label in (
        ("sfod", SYMMETRIC_COLOR, "Symmetric"),
        ("afod", ASYMMETRIC_COLOR, "Asymmetric"),
    ):
        x, y = fod_outline_xy(result, key, scale)
        axes[2].plot(x, y, color=color, lw=2, label=label)
    axes[2].set_title("Overlay")
    axes[2].legend(loc="lower center", fontsize=8, frameon=False)
    axes[2].axhline(0, color="#d6e0e5", lw=0.6, zorder=-1)
    axes[2].axvline(0, color="#d6e0e5", lw=0.6, zorder=-1)
    axes[2].set(aspect="equal", xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    for spine in axes[2].spines.values():
        spine.set_visible(False)

    fig.suptitle(f"Histology FODs | asymmetric power fraction = {float(result['odd_power_fraction']):.3f}")
    fig.tight_layout()
    out = Path(filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", required=True, help="INPUT NIfTI (.nii/.nii.gz), 2D structure-tensor vectors, not .npz")
    parser.add_argument("--centre", "--center", nargs=2, type=int, required=True,
                        metavar=("CX", "CY"), help="zero-based centre pixel indices")
    parser.add_argument("--patch", nargs=2, type=int, required=True,
                        metavar=("XDIM", "YDIM"), help="odd-sized rectangular patch")
    parser.add_argument("--n-bins", type=int, default=360, help="number of angular bins (default: 360)")
    parser.add_argument("--fwhm-deg", type=float, default=23.0,
                        help="angular smoothing FWHM (default: 23 degrees)")
    parser.add_argument("--output", default="hist_afod.npz", help="OUTPUT compressed .npz with FOD arrays (not the input vector file)")
    parser.add_argument("--plot", help="optional PNG output with filled 2D FOD glyphs")
    args = parser.parse_args()

    vectors = load_nifti_vectors(args.vectors)
    result = estimate_histology_afod(
        vectors, centre=tuple(args.centre), patch=tuple(args.patch),
        n_bins=args.n_bins, fwhm_deg=args.fwhm_deg,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **result)
    print(f"Saved {target}: {int(result['n_valid'])} vectors, "
          f"odd-power fraction = {float(result['odd_power_fraction']):.5f}")
    if args.plot:
        plot_fods(result, args.plot)
        print(f"Saved {args.plot}")


if __name__ == "__main__":
    main()
