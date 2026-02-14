#!/usr/bin/env python3
"""Convert CIVET cortical surfaces to FreeSurfer surf files for MNE workflows."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import signal
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
            "Convert CIVET Volumes/meshes to FreeSurfer format using mris_convert. "
            "Writes mri/T1.mgz and {lh,rh}.{pial,white} under SUBJECTS_DIR/<subject>/surf."
        )
    )
    # Input Volume
    parser.add_argument(
        "--t1-nrrd",
        type=Path,
        required=True,
        help="T1 NRRD file to convert into mri/T1.mgz before surface conversion.",
    )
    # Args for finding input surface files
    parser.add_argument(
        "--civet-dir",
        type=Path,
        default=Path("CIVETv2.0_Surfaces"),
        help="CIVET surfaces root (searched recursively).",
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
    # args for freesurfer output folders
    parser.add_argument(
        "--subjects-dir",
        type=Path,
        required=True,
        help="FreeSurfer SUBJECTS_DIR root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing FreeSurfer surfaces.",
    )
    # util args
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print discovered files and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print conversion actions without executing them.",
    )
    return parser.parse_args()


def run_command(command, dry_run=False):
    print("------------------ RUNNING A COMMAND IN BASH ------------------")
    print(command)
    print("-------------------------- OUTPUT -----------------------------")
    if not dry_run:
        bash(command.replace("\\", "").split())
    print("------------------------ END OUTPUT ---------------------------\n")


def run_mri_convert(input_path: Path, output_path: Path, extra_args: list[str], dry_run: bool) -> None:
    cmd = ["mri_convert", *extra_args, str(input_path), str(output_path)]
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def bash(cmd, print_stdout=True, print_stderr=True):
    sp = subprocess
    proc = sp.Popen(cmd, stderr=sp.PIPE, stdout=sp.PIPE)
    # , shell=True, universal_newlines=True,
    # executable='/bin/bash')

    all_stdout = []
    all_stderr = []
    while proc.poll() is None:
        for stdout_line in proc.stdout:
            if stdout_line != '':
                if print_stdout:
                    print(stdout_line.decode(), end='')
                all_stdout.append(stdout_line)
        for stderr_line in proc.stderr:
            if stderr_line != '':
                if print_stderr:
                    print(stderr_line.decode(), end='', file=sys.stderr)
                all_stderr.append(stderr_line)

    stdout_text = ''.join([x.decode() for x in all_stdout])
    stderr_text = ''.join([x.decode() for x in all_stderr])
    if proc.wait() != 0:
        raise VerboseCalledProcessError(proc.returncode, cmd, stdout_text, stderr_text)


class VerboseCalledProcessError(subprocess.CalledProcessError):
    def __str__(self):
        if self.returncode and self.returncode < 0:
            try:
                msg = "Command '%s' died with %r." % (
                    self.cmd, signal.Signals(-self.returncode))
            except ValueError:
                msg = "Command '%s' died with unknown signal %d." % (
                    self.cmd, -self.returncode)
        else:
            msg = "Command '%s' returned non-zero exit status %d." % (
                self.cmd, self.returncode)

        return f'{msg}\n' \
               f'Stdout:\n' \
               f'{self.output}\n' \
               f'Stderr:\n' \
               f'{self.stderr}'


def run_mris_convert(input_path: Path, output_path: Path, dry_run: bool) -> None:
    # TODO: Can we just use run_mri_convert?
    cmd = ["mris_convert", str(input_path), str(output_path)]
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _surface_to_tkr_transform(t1_path: Path) -> np.ndarray:
    img = nib.load(str(t1_path))
    affine = img.header.get_vox2ras() @ np.linalg.inv(img.header.get_vox2ras_tkr())
    return np.linalg.inv(affine)


def apply_t1_tkr_alignment(surface_path: Path, t1_path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"apply_t1_tkr_alignment {surface_path} using {t1_path}")
        return
    transform = _surface_to_tkr_transform(t1_path)
    vertices, faces = read_geometry(str(surface_path))
    vertices_h = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    vertices_aligned = (transform @ vertices_h.T).T[:, :3]
    write_geometry(str(surface_path), vertices_aligned, faces)


def build_t1_from_nrrd(
    t1_nrrd_path: Path,
    fs_subject_dir: Path,
    overwrite: bool=False,
    dry_run: bool=False,
) -> None:
    mri_dir = Path(fs_subject_dir) / "mri"
    orig_path = mri_dir / "orig.mgz"
    conform_path = mri_dir / "orig_conformed.mgz"
    nu_path = mri_dir / "orig_nu.mgz"  # Intensity correction
    norm_path = mri_dir / "orig_normed.mgz"  # gradient scalings normalized across subjects
    t1_path = mri_dir / "T1.mgz"

    if not orig_path.exists() or overwrite:
        if not Path(t1_nrrd_path).exists():
            raise FileNotFoundError(f"Missing NRRD volume: {t1_nrrd_path}")
        img = nrrd_to_mgh_image(t1_nrrd_path)
        orig_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, str(orig_path))

    if not conform_path.exists() or overwrite:    
        run_mri_convert(
            orig_path,
            conform_path,
            extra_args=["--conform"],
            dry_run=dry_run
        )

    if not nu_path.exists() or overwrite:
        command = "mri_nu_correct.mni --i {} \\\n                   --o {} --n 2"
        command = command.format(str(conform_path), str(nu_path))
        print(command)
        run_command(command)


    if not t1_path.exists() or overwrite:    
        command = "mri_normalize  {} \\\n         {}"
        command = command.format(str(nu_path), str(t1_path))
        print(command)
        run_command(command)

    return


def nrrd_to_mgh_image(path_nrrd):
    data, info = nrrd.read(str(path_nrrd))
    directions = np.asarray(info.get("space directions"), dtype=float)
    if directions.shape != (3, 3):
        raise ValueError(f"Unexpected 'space directions' shape: {directions.shape}")
    origin = np.asarray(info.get("space origin", [0.0, 0.0, 0.0]), dtype=float)
    if origin.shape != (3,):
        raise ValueError(f"Unexpected 'space origin' shape: {origin.shape}")

    affine_lps = np.eye(4, dtype=float)
    affine_lps[:3, :3] = directions
    affine_lps[:3, 3] = origin
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_ras = lps_to_ras @ affine_lps

    return nib.MGHImage(np.asarray(data, dtype=np.float32), affine=affine_ras)


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


def get_subject_session_from_nrrd_fname(nrrd_path):
    import re
    fname = Path(nrrd_path).name
    m = re.search(r"_([0-9]+)_V([0-9]+)_", fname)
    if not m:
        raise ValueError(
            f"Filename {fname} does not match expected pattern: '*_<subject>_V<session>_*'"
            )
    subject_id, session_id = m.groups()
    return subject_id, session_id


def main() -> int:
    args = parse_args()

    # Checks
    if not args.civet_dir.exists():
        print(f"CIVET dir not found: {args.civet_dir}", file=sys.stderr)
        return 1
    if shutil.which("mris_convert") is None and not args.list:
        print("mris_convert not found on PATH.", file=sys.stderr)
        return 1
    if shutil.which("mri_convert") is None:
        print("mri_convert not found on PATH; required for --t1-conversion.", file=sys.stderr)
        return 1

    # Build output directory
    subject, session = get_subject_session_from_nrrd_fname(args.t1_nrrd)
    fs_subject_dir = args.subjects_dir / f"sub-{subject}_ses-{session}"
    surf_dir = fs_subject_dir / "surf"
    t1_path = fs_subject_dir / "mri" / "T1.mgz"

    # Convert Volume
    if not args.t1_nrrd.exists():
        print(f"T1 NRRD not found: {args.t1_nrrd}", file=sys.stderr)
        return 1
    build_t1_from_nrrd(
        args.t1_nrrd,
        fs_subject_dir=fs_subject_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run
    )

    # Convert Surfaces
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

    for _, input_path, output_path in mapping:
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing: {output_path}")
            continue
        run_mris_convert(input_path, output_path, dry_run=args.dry_run)
        apply_t1_tkr_alignment(output_path, t1_path=t1_path, dry_run=args.dry_run)

    print("\nNext step for BEM surfaces (watershed):")
    print(
        "mne watershed_bem "
        f"--subject sub-{subject}_ses-{session} "
        f"--subjects-dir {args.subjects_dir} "
        "--overwrite"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
