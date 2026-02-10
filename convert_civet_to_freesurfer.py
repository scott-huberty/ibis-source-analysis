#!/usr/bin/env python3
"""Convert CIVET cortical surfaces to FreeSurfer surf files for MNE workflows."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import nrrd
from nibabel.freesurfer.io import read_geometry, write_geometry


@dataclass(frozen=True)
class SurfaceSpec:
    civet_surface: str
    hemisphere: str
    fs_hemi: str
    fs_surface: str


SURFACE_SPECS = (
    SurfaceSpec("gray", "left", "lh", "pial"),
    SurfaceSpec("gray", "right", "rh", "pial"),
    SurfaceSpec("white", "left", "lh", "white"),
    SurfaceSpec("white", "right", "rh", "white"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert CIVET meshes to FreeSurfer surfaces using mris_convert. "
            "Writes {lh,rh}.{pial,white} under SUBJECTS_DIR/<subject>/surf."
        )
    )
    parser.add_argument(
        "--civet-dir",
        type=Path,
        default=Path("CIVETv2.0_Surfaces"),
        help="CIVET surfaces root (searched recursively).",
    )
    parser.add_argument(
        "--subjects-dir",
        type=Path,
        required=True,
        help="FreeSurfer SUBJECTS_DIR root.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        required=True,
        help="FreeSurfer subject folder name (e.g., ibis_210772_V1_anlm0.5r).",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="327680",
        help="Resolution token in CIVET filenames (e.g., 327680 or 81920).",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="vtk",
        choices=("vtk", "obj"),
        help="Input surface extension to search for.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print discovered files and exit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing FreeSurfer surfaces.",
    )
    parser.add_argument(
        "--skip-scanner-to-tkr",
        action="store_true",
        help="Skip post-conversion transform from scanner RAS to FreeSurfer tkr RAS.",
    )
    parser.add_argument(
        "--t1-nrrd",
        type=Path,
        default=None,
        help="Optional T1 NRRD file to convert into mri/T1.mgz before surface conversion.",
    )
    parser.add_argument(
        "--t1-conversion",
        type=str,
        default="fs-via-nifti",
        choices=("fs-via-nifti", "direct-mgz"),
        help=(
            "How to build T1.mgz from --t1-nrrd. "
            "'fs-via-nifti' writes NIfTI then calls FreeSurfer mri_convert. "
            "'direct-mgz' writes MGZ directly via nibabel."
        ),
    )
    parser.add_argument(
        "--write-orig-001",
        action="store_true",
        help="Also write mri/orig/001.mgz from --t1-nrrd.",
    )
    parser.add_argument(
        "--only-t1",
        action="store_true",
        help="Only build MRI files from --t1-nrrd; skip surface conversion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print conversion actions without executing them.",
    )
    return parser.parse_args()


def find_surface_file(
    civet_dir: Path,
    spec: SurfaceSpec,
    resolution: str,
    ext: str,
) -> Path:
    pattern = f"*_{spec.civet_surface}_surface*_rsl_{spec.hemisphere}_{resolution}*.{ext}"
    matches = sorted(civet_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Missing CIVET file for surface={spec.civet_surface}, "
            f"hemi={spec.hemisphere}, resolution={resolution}, ext={ext}, "
            f"pattern={pattern}"
        )

    # Prefer native ITK-space exports when available.
    preferred = [m for m in matches if "native_itkSpace" in m.stem]
    return preferred[0] if preferred else matches[0]


def scanner_to_tkr_matrix(t1_path: Path) -> np.ndarray:
    img = nib.load(str(t1_path))
    return img.header.get_vox2ras_tkr() @ np.linalg.inv(img.header.get_vox2ras())


def apply_affine_to_surface(surface_path: Path, affine: np.ndarray) -> None:
    verts, faces = read_geometry(str(surface_path))
    verts_h = np.c_[verts, np.ones(len(verts))]
    verts_out = (affine @ verts_h.T).T[:, :3]
    write_geometry(str(surface_path), verts_out, faces)


def run_mris_convert(input_path: Path, output_path: Path, dry_run: bool) -> None:
    cmd = ["mris_convert", str(input_path), str(output_path)]
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def nrrd_to_ras_nifti(
    nrrd_path: Path,
    nii_path: Path,
    dry_run: bool,
    force_float32: bool = True,
) -> None:
    print(f"NRRD->NIfTI: {nrrd_path} -> {nii_path}")
    if dry_run:
        return

    data, info = nrrd.read(str(nrrd_path))
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={data.shape}")

    directions = np.asarray(info.get("space directions"), dtype=float)
    if directions.shape != (3, 3):
        raise ValueError(f"Unexpected 'space directions' shape: {directions.shape}")

    origin = np.asarray(info.get("space origin", [0.0, 0.0, 0.0]), dtype=float)
    if origin.shape != (3,):
        raise ValueError(f"Unexpected 'space origin' shape: {origin.shape}")

    # NRRD axis direction vectors are per array axis.
    affine_lps = np.eye(4, dtype=float)
    affine_lps[:3, :3] = directions
    affine_lps[:3, 3] = origin

    # Convert LPS -> RAS
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_ras = lps_to_ras @ affine_lps

    if force_float32:
        data = np.asarray(data, dtype=np.float32)

    img = nib.Nifti1Image(data, affine_ras)

    # Make FS / other tools less ambiguous: set both sform and qform
    img.set_sform(affine_ras, code=1)
    img.set_qform(affine_ras, code=1)

    nii_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(nii_path))


def nrrd_to_ras_mgz(
    nrrd_path: Path,
    output_path: Path,
    dry_run: bool,
    force_float32: bool = True,
) -> None:
    print(f"NRRD->MGZ: {nrrd_path} -> {output_path}")
    if dry_run:
        return

    data, info = nrrd.read(str(nrrd_path))
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={data.shape}")

    if "space directions" not in info:
        raise ValueError("NRRD missing 'space directions'; cannot build affine.")

    directions = np.asarray(info["space directions"], dtype=float)
    if directions.shape != (3, 3):
        raise ValueError(f"Unexpected 'space directions' shape: {directions.shape}")

    origin = np.asarray(info.get("space origin", [0.0, 0.0, 0.0]), dtype=float)
    if origin.shape != (3,):
        raise ValueError(f"Unexpected 'space origin' shape: {origin.shape}")

    affine_lps = np.eye(4, dtype=float)
    affine_lps[:3, :3] = directions
    affine_lps[:3, 3] = origin

    # CIVET/ITK NRRD commonly uses LPS; FreeSurfer uses RAS.
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_ras = lps_to_ras @ affine_lps

    if force_float32:
        data = np.asarray(data, dtype=np.float32)

    img = nib.MGHImage(data, affine=affine_ras)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output_path))


def run_mri_convert(input_path: Path, output_path: Path, extra_args: list[str], dry_run: bool) -> None:
    cmd = ["mri_convert", *extra_args, str(input_path), str(output_path)]
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def build_t1_from_nrrd(
    nrrd_path: Path,
    fs_subject_dir: Path,
    mode: str,
    write_orig_001: bool,
    dry_run: bool,
) -> None:
    mri_dir = fs_subject_dir / "mri"
    t1_path = mri_dir / "T1.mgz"

    if mode == "direct-mgz":
        nrrd_to_ras_mgz(nrrd_path, t1_path, dry_run=dry_run)
        if write_orig_001:
            nrrd_to_ras_mgz(nrrd_path, mri_dir / "orig" / "001.mgz", dry_run=dry_run)
        return

    orig_nii = mri_dir / "orig.nii.gz"
    orig_mgz = mri_dir / "orig.mgz"
    nrrd_to_ras_nifti(nrrd_path, orig_nii, dry_run=dry_run, force_float32=True)
    run_mri_convert(orig_nii, orig_mgz, extra_args=[], dry_run=dry_run)
    # run_mri_convert(orig_mgz, orig_mgz, extra_args=["--conform"], dry_run=dry_run)
    run_mri_convert(orig_mgz, t1_path, extra_args=[], dry_run=dry_run)
    if write_orig_001:
        run_mri_convert(orig_mgz, mri_dir / "orig" / "001.mgz", extra_args=[], dry_run=dry_run)


def main() -> int:
    args = parse_args()

    if not args.only_t1 and not args.civet_dir.exists():
        print(f"CIVET dir not found: {args.civet_dir}", file=sys.stderr)
        return 1
    if not args.only_t1 and shutil.which("mris_convert") is None and not args.list:
        print("mris_convert not found on PATH.", file=sys.stderr)
        return 1
    if args.t1_nrrd is not None and args.t1_conversion == "fs-via-nifti" and shutil.which("mri_convert") is None:
        print("mri_convert not found on PATH; required for --t1-conversion fs-via-nifti.", file=sys.stderr)
        return 1

    fs_subject_dir = args.subjects_dir / args.subject
    surf_dir = fs_subject_dir / "surf"
    t1_path = fs_subject_dir / "mri" / "T1.mgz"

    if args.t1_nrrd is not None:
        if not args.t1_nrrd.exists():
            print(f"T1 NRRD not found: {args.t1_nrrd}", file=sys.stderr)
            return 1
        build_t1_from_nrrd(
            nrrd_path=args.t1_nrrd,
            fs_subject_dir=fs_subject_dir,
            mode=args.t1_conversion,
            write_orig_001=args.write_orig_001,
            dry_run=args.dry_run,
        )
    elif args.only_t1:
        print("--only-t1 requires --t1-nrrd.", file=sys.stderr)
        return 1

    if args.only_t1:
        return 0

    if not args.list and (not t1_path.exists()) and (not args.skip_scanner_to_tkr):
        print(
            f"T1 not found at {t1_path}; required for scanner->tkr transform. "
            "Use --skip-scanner-to-tkr to bypass.",
            file=sys.stderr,
        )
        return 1

    mapping: list[tuple[SurfaceSpec, Path, Path]] = []
    for spec in SURFACE_SPECS:
        input_path = find_surface_file(args.civet_dir, spec, args.resolution, args.ext)
        output_path = surf_dir / f"{spec.fs_hemi}.{spec.fs_surface}"
        mapping.append((spec, input_path, output_path))

    if args.list:
        print("Discovered CIVET inputs:")
        for spec, input_path, output_path in mapping:
            print(
                f"  {spec.civet_surface:>5}/{spec.hemisphere:<5} -> "
                f"{spec.fs_hemi}.{spec.fs_surface:<5} | {input_path} -> {output_path}"
            )
        return 0

    surf_dir.mkdir(parents=True, exist_ok=True)
    affine = None
    if not args.skip_scanner_to_tkr:
        affine = scanner_to_tkr_matrix(t1_path)

    for _, input_path, output_path in mapping:
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing: {output_path}")
            continue

        run_mris_convert(input_path, output_path, dry_run=args.dry_run)

        if affine is not None:
            if args.dry_run:
                print(f"apply_affine(scanner->tkr): {output_path}")
            else:
                apply_affine_to_surface(output_path, affine)

    print("\nNext step for BEM surfaces (watershed):")
    print(
        "mne watershed_bem "
        f"--subject {args.subject} "
        f"--subjects-dir {args.subjects_dir} "
        "--overwrite"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
