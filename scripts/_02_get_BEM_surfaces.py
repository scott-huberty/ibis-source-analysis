"""Create BEM surfaces with mne watershed.

Example (current parameters):
python scripts/_02_get_BEM_surfaces.py \
    --subject sub-210772_ses-06 \
    --subjects-dir freesurfer \
    --volume T1 \
    --overwrite
"""

from argparse import ArgumentParser
from pathlib import Path

from mne.bem import make_watershed_bem


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Create BEM surfaces using mne.bem.make_watershed_bem.",
        epilog=(
            "Example with current parameters:\n"
            "  python scripts/_02_get_BEM_surfaces.py "
            "--subject sub-210772_ses-06 --subjects-dir freesurfer "
            "--volume T1 --overwrite"
        ),
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
        "--volume",
        default="T1",
        help="MRI volume to use (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    subjects_dir = args.subjects_dir.resolve()

    if not subjects_dir.exists():
        raise FileNotFoundError(f"subjects_dir does not exist: {subjects_dir}")

    make_watershed_bem(
        subject=args.subject,
        subjects_dir=subjects_dir,
        volume=args.volume,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
