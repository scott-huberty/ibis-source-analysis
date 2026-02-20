import itertools
from pathlib import Path
import subprocess
from types import SimpleNamespace

import nibabel as nib
import tfce_mediation

RECOMPUTE = False
HEMIS = {"left": "lh", "right": "rh"}  # CIVET to FreeSurfer hemisphere names
SURFACES = {"gray": "pial", "white": "white"}  # CIVET to FreeSurfer surface names
RESOLUTION = 81920  # number of vertices. or 327680 for higher res
civet_root = Path(__file__).resolve().parents[1] / "CIVETv2.0_Surfaces"
fs_subject_dir = Path(__file__).resolve().parents[1] / "freesurfer" / "subjects"
obj_fpath = "/Users/scotterik/devel/sMRI/CIVETv2.0_Surfaces/stx_stx_ibis_210772_V06_T1w_dBias_gray_surface_left_81920.obj"


for (surf_civet, surf_fs), (hemi_civet, hemi_fs) in itertools.product(
    SURFACES.items(), HEMIS.items()
):
    fpath = next(civet_root.glob(f"stx_stx_ibis*_{surf_civet}_surface_{hemi_civet}_{RESOLUTION}.obj"))
    # Parse the subject and Session from the filename
    parts = fpath.name.split("_")
    subject = parts[3]
    session = parts[4].strip("V")
    fs_subj_sess_dir = fs_subject_dir / f"sub-{subject}_ses-{session}"
    fs_subj_sess_dir.mkdir(parents=True, exist_ok=True)
    out_fpath = fs_subj_sess_dir / f"{hemi_fs}.{surf_fs}.srf"
    if out_fpath.exists() and not RECOMPUTE:
        print(f"Skipping {fpath.name}, output {out_fpath} exists...")
        continue
    elif out_fpath.exists() and RECOMPUTE:
        print(f"Recomputing {fpath.name}, deleting existing output {out_fpath}...")
        out_fpath.unlink()
    else:
        pass
    print(f"Converting {fpath.name} to {out_fpath}...")
    # HACK: monkeypatching a SimpleNamespace to mimic argparse.parse_args() output
    # Since I want to run tfce_mediation.tools.tm_convert_surface.run() directly
    # Instead of going through command line parsing (tm_tools convert-surface -i_mniobj ...)
    # Each arg is defined with nargs, so the code expects lists (even for single values).
    # 43097096
    opts = SimpleNamespace(
        inputmniobj=[str(obj_fpath)],
        outputfreesurfer=[str(out_fpath)],
        # Other args set to None
        inputfreesurfer=None,
        inputgifti=None,
        inputply=None,
        inputvoxel=None,
        specifyvolume=None,
        voxelthreshold=None,
        voxelbackbone=None,
        paintvoxelsurface=None,
        outputvoxelsurfmgh=None,
        outputwaveform=None,
        outputstl=None,
        outputply=None,
        paintsurface=None,
        paintsecondsurface=None,
        paintfslabel=None,
        paintfsannot=None,
    )
    tfce_mediation.tools.tm_convert_surface.run(opts)

    # Now we'll smooth, inflate, and sphere the surface using FreeSurfer's mris_smooth and mris_inflate
    # First just check if we can access FreeSurfer
    subprocess.run(["mris_smooth", "--help"], check=True)

    # Smoothed surface file path
    smoothed_fpath = out_fpath.with_name(f"{hemi_fs}.{surf_fs}_smoothed")
    print(f"Smoothing surface and saving to {smoothed_fpath


# fs_surf = nib.freesurfer.io.read_geometry(out_fpath)

