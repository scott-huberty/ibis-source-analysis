"""Align EEG sensors to MRI space using MNE coregistration.

Example (current parameters):
python scripts/_03_align.py \
    --raw eeg/STL7009_6m_20191209_095440/STL7009_6m_20191209_095440_proc-cleaned_raw.fif \
    --montage GSN-HydroCel-129 \
    --subject sub-210772_ses-06 \
    --subjects-dir freesurfer \
    --fiducials estimated \
    --icp-iterations 500 \
    --nasion-weight 2.0 \
    --surfaces head \
    --output-dir Results_Seg/coregistration
"""

from argparse import ArgumentParser
import faulthandler
import multiprocessing as mp
import os
from pathlib import Path

import mne


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run MNE coregistration and visualize the alignment.",
        epilog=(
            "Example with current parameters:\n"
            "  python scripts/_03_align.py "
            "--raw eeg/STL7009_6m_20191209_095440/"
            "STL7009_6m_20191209_095440_proc-cleaned_raw.fif "
            "--montage GSN-HydroCel-129 "
            "--subject sub-210772_ses-06 "
            "--subjects-dir freesurfer "
            "--fiducials estimated "
            "--icp-iterations 500 --nasion-weight 2.0 "
            "--surfaces head --output-dir Results_Seg/coregistration"
        ),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=(
            Path(__file__).parents[1]
            / "eeg/STL7009_6m_20191209_095440/"
            "STL7009_6m_20191209_095440_proc-cleaned_raw.fif"
        ),
        help="Path to input raw FIF file (default: %(default)s).",
    )
    parser.add_argument(
        "--montage",
        default="GSN-HydroCel-129",
        help="Standard montage name (default: %(default)s).",
    )
    parser.add_argument(
        "--subject",
        default="sub-210772_ses-06",
        help="FreeSurfer subject ID (default: %(default)s).",
    )
    parser.add_argument(
        "--subjects-dir",
        type=Path,
        default=Path(__file__).parents[1] / "freesurfer",
        help="Path to FreeSurfer subjects directory (default: %(default)s).",
    )
    parser.add_argument(
        "--fiducials",
        default="estimated",
        help="Fiducial strategy for coregistration (default: %(default)s).",
    )
    parser.add_argument(
        "--icp-iterations",
        type=int,
        default=500,
        help="Number of ICP iterations (default: %(default)s).",
    )
    parser.add_argument(
        "--nasion-weight",
        type=float,
        default=2.0,
        help="Nasion weighting for ICP fit (default: %(default)s).",
    )
    parser.add_argument(
        "--surfaces",
        default="head",
        help="Surface(s) to plot in alignment view (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "Results_Seg" / "coregistration",
        help="Directory where alignment screenshots are saved (default: %(default)s).",
    )
    parser.add_argument(
        "--render-timeout-sec",
        type=int,
        default=180,
        help="Timeout for 3D rendering subprocess in seconds (default: %(default)s).",
    )
    return parser


def save_alignment_views(fig, output_dir: Path, subject: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    views = (
        ("view1", 135, 80, 0.6),
        ("view2", 45, 80, 0.6),
        ("view3", 270, 80, 0.6),
    )
    for label, azimuth, elevation, distance in views:
        mne.viz.set_3d_view(
            figure=fig,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
        )
        out_file = output_dir / f"coregistration_{subject}_{label}.png"
        fig.plotter.screenshot(str(out_file))


def configure_runtime() -> None:
    # os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    # os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    mne.viz.set_3d_backend("pyvista")   



def main() -> None:
    args = build_parser().parse_args()
    raw_path = args.raw.resolve()
    subjects_dir = args.subjects_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not raw_path.exists():
        raise FileNotFoundError(f"raw file does not exist: {raw_path}")
    if not subjects_dir.exists():
        raise FileNotFoundError(f"subjects_dir does not exist: {subjects_dir}")

    configure_runtime()

    raw = mne.io.read_raw_fif(raw_path)
    montage = mne.channels.make_standard_montage(args.montage)
    raw.set_montage(montage, on_missing="ignore")

    coreg = mne.coreg.Coregistration(
        raw.info,
        subject=args.subject,
        subjects_dir=subjects_dir,
        fiducials=args.fiducials,
    )
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(
        n_iterations=args.icp_iterations,
        nasion_weight=args.nasion_weight,
        verbose=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    trans_out = output_dir / f"{args.subject}-coreg-trans.fif"
    mne.write_trans(trans_out, coreg.trans, overwrite=True)

    fig = mne.viz.plot_alignment(
        info=raw.info,
        trans=coreg.trans,
        subject=args.subject,
        subjects_dir=subjects_dir,
        surfaces=args.surfaces,
    )
    save_alignment_views(
        fig=fig,
        output_dir=Path(output_dir),
        subject=args.subject,
    )

if __name__ == "__main__":
    main()
