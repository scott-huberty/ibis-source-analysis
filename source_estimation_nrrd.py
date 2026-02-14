import argparse
from types import SimpleNamespace
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from nibabel.freesurfer.io import read_geometry, write_geometry
import numpy as np
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import sys
import mne
import shutil
from scipy import spatial
import pandas as pd
from tqdm.notebook import tqdm
from mne.surface import decimate_surface
import mne
import xarray as xr
import subprocess as sp
from scipy.spatial import KDTree
import signal
# import trimesh
import nrrd
import tfce_mediation

class VerboseCalledProcessError(sp.CalledProcessError):
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


def bash(cmd, print_stdout=True, print_stderr=True):
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


def run_command(command, dry_run=False):
    print("------------------ RUNNING A COMMAND IN BASH ------------------")
    print(command)
    print("-------------------------- OUTPUT -----------------------------")
    if not dry_run:
        bash(command.replace("\\", "").split())
    print("------------------------ END OUTPUT ---------------------------\n")


def _find_civet_surface_file(civet_surface_dir, surface, hemi, resolution="327680"):
    civet_surface_dir = Path(civet_surface_dir)
    patterns = [
        f"*_{surface}_surface_rsl_{hemi}_{resolution}_native_itkSpace.obj",
        f"*_{surface}_surface_rsl_{hemi}_{resolution}.obj",
        f"*_{surface}_surface_rsl_{hemi}_{resolution}_native_itkSpace.vtk",
    ]
    for pattern in patterns:
        matches = sorted(civet_surface_dir.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find CIVET {surface}/{hemi} surface at resolution={resolution} in {civet_surface_dir}"
    )


def _nrrd_to_mgh_image(path_nrrd):
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


def convert_civet_surfaces(subject, fs_subject_dir="/usr/local/freesurfer/subjects/",
                           civet_surface_dir="CIVETv2.0_Surfaces/NATIVE",
                           civet_resolution="327680",
                           recompute=False):
    path_T1 = Path(fs_subject_dir) / fs(subject) / "mri" / "T1.mgz"
    img = nib.load(str(path_T1))

    hemis = {"left": "lh", "right": "rh"}
    surfaces = {"gray": "pial", "white": "white"}

    affine = img.header.get_vox2ras() @ np.linalg.inv(img.header.get_vox2ras_tkr())
    for hemi in hemis:
        for surface in surfaces:
            path_out = Path(fs_subject_dir) / fs(subject) / "surf" / f"{hemis[hemi]}.{surfaces[surface]}"
            if not path_out.exists() or recompute:
                path_in = _find_civet_surface_file(
                    civet_surface_dir,
                    surface=surface,
                    hemi=hemi,
                    resolution=civet_resolution,
                )

                path_out.parent.mkdir(parents=True, exist_ok=True)
                if path_out.exists():
                    path_out.unlink()

                if path_in.suffix.lower() != ".obj":
                    raise ValueError(
                        f"tfce_mediation.tools.tm_convert_surface.run expects .obj input, got: {path_in}"
                    )
                opts = SimpleNamespace(
                    inputmniobj=[str(path_in)],
                    outputfreesurfer=[str(path_out)],
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

                path_converted = path_out if path_out.exists() else Path(f"{path_out}.srf")
                vertices, faces = read_geometry(str(path_converted))
                vertices = (np.linalg.inv(affine) @ np.hstack([vertices, np.ones((vertices.shape[0], 1))]).T).T[:, :3]
                write_geometry(str(path_out), vertices, faces)


    path_T1 = Path(fs_subject_dir) / fs(subject) / "mri" / "T1.mgz"
    img = nib.load(str(path_T1))
    affine = img.header.get_vox2ras() @ np.linalg.inv(img.header.get_vox2ras_tkr())
    for surface in ["inner_skull", "outer_skull", "outer_skin"]:
        path_in = fs_subject_dir / fs(subject) / "bem" / "watershed" / "{}_{}_surface".format(fs(subject), surface)
        path_out = fs_subject_dir / fs(subject) / "bem" /  "{}_1.surf".format(surface)
        vertices, faces, volume_info = read_geometry(str(path_in), read_metadata=True)
        vertices += img.header["Pxyz_c"]
        vertices = (np.linalg.inv(affine) @ np.hstack([vertices, np.ones((vertices.shape[0], 1))]).T).T[:, :3]
        write_geometry(str(path_out),  vertices, faces)


def build_civet_brainmask(subject, fs_subject_dir="/usr/local/freesurfer/subjects/",
                          recompute=True):
    path_T1 = Path(fs_subject_dir) / fs(subject) / "mri" / "T1.mgz"
    path_out = path_T1.parent / 'brainmask.mgz'
    if not path_out or recompute:
        img = nib.load(str(path_T1))

        # Loading the cortical mesh
        mesh = []
        for surface in ["lh.pial", "rh.pial"]:
            path_out = Path(fs_subject_dir) / fs(subject) / "surf" / surface
            vertices, faces = read_geometry(str(path_out))
            vertices, faces = fix_all_defects(vertices, faces)
            mesh.append(trimesh.Trimesh(vertices, faces))
        mesh = trimesh.util.concatenate(mesh)

        # Voxelizing the mesh
        x = mesh.voxelized(img.header.get_zooms()[0])
        x = x.fill()
        voxel_vertices = x.points - x.translation + mesh.vertices.min(0)

        # Make a brainmask
        inds = np.round((np.linalg.inv(img.header.get_vox2ras_tkr()) @ np.hstack(
            [voxel_vertices, np.ones((voxel_vertices.shape[0], 1))]).T)[:3]).astype(int)
        mask = np.zeros(img.header.get_data_shape())
        mask[inds[0], inds[1], inds[2]] = True

        # Save the mask as an MRI image
        mask_img = nib.Nifti1Image(mask, img.affine, img.header)
        nib.save(mask_img, path_out)


def get_subject_civet_path(subject, civet_root="/media/christian/ElementsSE/lewis/CIVET-2.1.0/"):
    return civet_root / fs(subject)


def fs(subject):
    return f'{subject}_V1_anlm0.5r'


def freesurfer_preprocessing(subject, fs_subject_dir="/usr/local/freesurfer/subjects/",
                             t1_nrrd_path="stx_ibis_210772_V06_T1w_dBias.nrrd",
                             recompute=False):

    fs_subject = fs(subject)

    path_not_nu_not_conf = Path(fs_subject_dir) / fs_subject / "mri" / "T1_not_nu_not_conf.mgz"
    if not path_not_nu_not_conf.exists() or recompute:
        path_nrrd = Path(t1_nrrd_path)
        if not path_nrrd.exists():
            raise FileNotFoundError(f"Missing NRRD volume: {path_nrrd}")
        img = _nrrd_to_mgh_image(path_nrrd)
        path_not_nu_not_conf.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, str(path_not_nu_not_conf))

    path_not_nu = Path(fs_subject_dir) / fs(subject) / "mri" / "T1_not_nu.mgz"
    if not path_not_nu.exists() or recompute:    
        command = (f"mri_convert --conform {path_not_nu_not_conf}\\"
                   f"\n                      {path_not_nu}")
        print(command)
        run_command(command)

    path_not_norm = Path(fs_subject_dir) / fs(subject) / "mri" / "T1_not_norm.mgz"
    if not path_not_norm.exists() or recompute:
        command = "mri_nu_correct.mni --i {} \\\n                   --o {} --n 2"
        command = command.format(str(path_not_nu), str(path_not_norm))
        print(command)
        run_command(command)


    path_T1 = Path(fs_subject_dir) / fs(subject) / "mri" / "T1.mgz"
    if not path_T1.exists() or recompute:    
        command = "mri_normalize  {} \\\n         {}"
        command = command.format(str(path_not_norm), str(path_T1))
        print(command)
        run_command(command)

    compute_talairach_transform(subject, fs_subject_dir=fs_subject_dir, recompute=recompute)


def compute_talairach_transform(subject, fs_subject_dir="/usr/local/freesurfer/subjects/",
                                recompute=False):

    path_tal = Path(fs_subject_dir) / fs(subject) / "mri" / "transforms" / "talairach.xfm"
    if not path_tal.exists() or recompute:
        path_T1 = Path(fs_subject_dir) / fs(subject) / "mri" / "T1.mgz"
        run_command(f"talairach --i {path_T1} --xfm {path_tal}")



"""
def transform_bem_surfaces(subject):
    path_T1 = fs_subject_dir / fs(subject) / "mri" / "T1.mgz"
    img = nib.load(str(path_T1))
    affine = img.header.get_vox2ras() @ np.linalg.inv(img.header.get_vox2ras_tkr())
    for surface in ["inner_skull", "outer_skull", "outer_skin"]:
        path_in = fs_subject_dir / fs(subject) / "bem" / "watershed" / "{}_{}_surface".format(fs(subject), surface)
        path_out = fs_subject_dir / fs(subject) / "bem" /  "{}_1.surf".format(surface)
        vertices, faces, volume_info = read_geometry(str(path_in), read_metadata=True)
        vertices += img.header["Pxyz_c"]
        vertices = (np.linalg.inv(affine) @ np.hstack([vertices, np.ones((vertices.shape[0], 1))]).T).T[:, :3]
        write_geometry(str(path_out),  vertices, faces)
""";


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull.points)

    return hull.find_simplex(p)>=0


def get_convex_hull_with_margin(outer_skin_vertices, vertical_axis, margin=10):
    """
     Return an inner_hull for the vertices comprise in the lower slab of the
     outer_skin mesh, removing its external margin to have a mostly plane surface,
     without the "walls" on its periphery.
    """

    axes = [0, 1, 2]
    axes.pop(vertical_axis)

    points = outer_skin_vertices[:, axes]
    center = points.mean(0)

    hull = ConvexHull(points)
    angles = np.arctan2(*(points[hull.vertices] - center).T)
    r = np.linalg.norm(points[hull.vertices] - center, axis=1)
    y = center[1] + (r-margin)*np.sin(angles)
    x = center[0] + (r-margin)*np.cos(angles)

    inner_hull = ConvexHull(np.array([x, y]).T)
    return inner_hull


def is_inner_point(hull, points, vertical_axis):
    axes = [0, 1, 2]
    axes.pop(vertical_axis)
    return np.array([in_hull(point, hull) for point in points[:, axes]])


def correct_lower_surface_cut(subject, fs_subject_dir="/usr/local/freesurfer/subjects/",
                              head_surfaces=("inner_skull", "outer_skull", "outer_skin"),
                              recompute=False):

    if (Path(fs_subject_dir) / fs(subject) / "bem" / f"{head_surfaces[-1]}_2.surf").exists() and not recompute:
        return

    vertices = {}
    fs_subject = fs(subject)
    for surface in head_surfaces:
        path_in = Path(fs_subject_dir) / fs(subject) / "bem" / "watershed" / f"{fs_subject}_{surface}_surface"
        vertices[surface], faces, volume_info = read_geometry(str(path_in), read_metadata=True)

    vertical_axis = 2
    other_axes = [0, 1, 2]
    other_axes.pop(vertical_axis)

    th = {}
    th["outer_skin"] = np.percentile(vertices["outer_skin"][:, vertical_axis], 2)+20
    th["outer_skull"] = th["outer_skin"]+10
    th["inner_skull"] = th["outer_skin"]+40

    vertice_mask = vertices["outer_skin"][:, vertical_axis] < th["outer_skin"]
    inner_hull = get_convex_hull_with_margin(vertices["outer_skin"][vertice_mask], vertical_axis)

    magin = 3
    for inner_surface, outer_surface in zip(head_surfaces[:-1], head_surfaces[1:]):
        df = {}
        for surface in [inner_surface, outer_surface]:
            cond = vertices[surface][:, vertical_axis] < th[surface]
            cond2 = is_inner_point(inner_hull, vertices[surface], vertical_axis)
            df[surface] = pd.DataFrame({surface: vertices[surface][:, vertical_axis],
                          "y": np.round(vertices[surface][:, other_axes[0]]),
                          "z": np.round(vertices[surface][:, other_axes[1]]),
                          "cond": cond & cond2,
                          "no_vert": np.arange(len(vertices[surface]), dtype=int)
                         })

        x, y, z = zip(*df[inner_surface].loc[df[inner_surface]["cond"], [inner_surface, "y", "z"]].values)

        tree = spatial.KDTree(list(zip(y, z)))

        tree.data_lut = {(y, z): x for x, y, z in zip(x, y, z)}

        for y, z, x, no_vert in df[outer_surface].loc[df[outer_surface]["cond"],
                                                      ["y", "z", outer_surface, "no_vert"]].values:

            inds = tree.query_ball_point([y, z], r=magin)
            if len(inds) == 0:
                dist, ind = tree.query([y, z])
                if dist > 10:
                    inds = [ind]

            if len(inds):
                min_x = np.min([tree.data_lut[tuple(tree.data[ind])] for ind in inds])
                if min_x-magin < x:  # need correction?
                    vertices[outer_surface][int(no_vert), vertical_axis] = min_x-magin

    for surface in head_surfaces:
        path_out = Path(fs_subject_dir) / fs(subject) / "bem" / "{}_2.surf".format(surface)
        _, faces, volume_info = read_geometry(str(path_in), read_metadata=True)
        write_geometry(str(path_out), vertices[surface],
                       faces, volume_info=volume_info)


def decimate_surface_wrapper(subject, fs_subject_dir="/usr/local/freesurfer/subjects/",
                             head_surfaces=("inner_skull", "outer_skull", "outer_skin"),
                             recompute=False):

    for surface in head_surfaces:
        path_out = Path(fs_subject_dir) / fs(subject) / "bem" / "{}.surf".format(surface)
        if not path_out.exists() or recompute:
            path_in = Path(fs_subject_dir) / fs(subject) / "bem" /  "{}_2.surf".format(surface)
            vertices, faces, volume_info = read_geometry(str(path_in), read_metadata=True)

            vertices, faces = decimate_surface(vertices, faces, n_triangles=5120, method='sphere')

            write_geometry(str(path_out), vertices, faces, volume_info=volume_info)


def computing_sphere_files(subject, fs_subject_dir="/usr/local/freesurfer/subjects/", recompute=False):

    for hemi in ["lh", "rh"]:
        path_in = Path(fs_subject_dir) / fs(subject) / "surf" / "{}.pial".format(hemi)
        path_out = Path(fs_subject_dir) / fs(subject) / "surf" / "{}.orig".format(hemi)
        if not path_out.exists() or recompute:
            shutil.copy(path_in, path_out)

    # .smoothwm files
    for hemi in ["lh", "rh"]:
        path_in = Path(fs_subject_dir) / fs(subject) / "surf" / hemi
        if not path_in.with_suffix(".smoothwm").exists() or recompute:
            cmd = ["mris_smooth",
                   str(path_in.with_suffix(".orig")),
                   str(path_in.with_suffix(".smoothwm"))]
            run_command(" ".join(cmd))
        else:
            print("{} already exists. Skipping.".format(path_in.with_suffix(".smoothwm")))

    # .inflated files
    for hemi in ["lh", "rh"]:
        path_in = Path(fs_subject_dir) / fs(subject) / "surf" / hemi
        if not path_in.with_suffix(".inflated").exists() or recompute:
            cmd = ["mris_inflate",
                   str(path_in.with_suffix(".smoothwm")),
                   str(path_in.with_suffix(".inflated"))]
            run_command(" ".join(cmd))
        else:
            print("{} already exists. Skipping.".format(path_in.with_suffix(".inflated")))


    # .sphere files
    for hemi in ["lh", "rh"]:
        path_in = Path(fs_subject_dir) / fs(subject) / "surf" / hemi
        if not path_in.with_suffix(".sphere").exists() or recompute:
            cmd = ["mris_sphere",
                   str(path_in.with_suffix(".inflated")),
                   str(path_in.with_suffix(".sphere"))]
            run_command(" ".join(cmd))
        else:
            print("{} already exists. Skipping.".format(path_in.with_suffix(".sphere")))


def build_mne_bem(fs_subject, fs_subject_dir="/usr/local/freesurfer/subjects/", 
                  ico=None, recompute=True):

    path_out = Path(fs_subject_dir) / fs_subject / "bem" / f"{fs_subject}-bem-sol.fif"
    if path_out.exists() and not recompute:
        return

    src = mne.setup_source_space(subject=fs_subject, subjects_dir=fs_subject_dir,
                                 surface="pial", spacing='oct6')
    src.save(Path(fs_subject_dir) / fs_subject / "bem" / f"{fs_subject}-src.fif", overwrite=True)

    model = mne.make_bem_model(subject=fs_subject, ico=ico, subjects_dir=fs_subject_dir)
    mne.write_bem_surfaces(Path(fs_subject_dir) / fs_subject / "bem" / f"{fs_subject}-bem.fif", model, overwrite=True)

    solution = mne.make_bem_solution(model)
    mne.write_bem_solution(path_out, solution, overwrite=True)


def build_head_model(subject, fs_subject_dir="/usr/local/freesurfer/subjects/", 
                     civet_surface_dir="CIVETv2.0_Surfaces/NATIVE",
                     t1_nrrd_path="stx_ibis_210772_V06_T1w_dBias.nrrd",
                     epoch_dir="/media/christian/ElementsSE/lewis/EEG_epoched/saccade_rejected/",
                     recompute=False):

    if not recompute:
        bem_root = Path(fs_subject_dir) / fs(subject) / "bem"   

        # This file is the last one created by this function 
        if (bem_root / f"{subject}_selected_sources_vert.nc").exists():
            print(f"Files for {subject} already exists. Skipping.")
            return 

    '''try:
        epochs_clean = get_epochs_with_montage(subject, epoch_dir=epoch_dir)
    except FileNotFoundError as e:
        print(f"EEG for {subject} missing. Skipping.")
        return'''

    # Preprocess CIVET volume in a proper T1 volume sfor FreeSurfer
    freesurfer_preprocessing(
        subject,
        fs_subject_dir=fs_subject_dir,
        t1_nrrd_path=t1_nrrd_path,
        recompute=recompute,
    )

    ## Converting brain surfaces
    # Conversion from CIVET to FreeSurfer: https://github.com/trislett/TFCE_mediation/issues/1  (pip install -U tfce-mediation)
    # Done after BEM reconstruction because we are using the metadata from one of the BEM surfaces
    convert_civet_surfaces(
        subject,
        fs_subject_dir=fs_subject_dir,
        civet_surface_dir=civet_surface_dir,
        recompute=recompute,
    )

    # Build brainsmask from CIVET surfaces
    # build_civet_brainmask(subject, fs_subject_dir=fs_subject_dir, recompute=recompute)

    # Extracting BEM surfaces using the wathershed algorithm
    out_path = Path(fs_subject_dir) / fs(subject) / "bem" / "watershed" / f"{fs(subject)}_outer_skull_surface"
    if not out_path.exists() or recompute:
        mne.bem.make_watershed_bem(fs(subject), subjects_dir=None, overwrite=True, volume='T1', atlas=False,
                                gcaatlas=False, preflood=None, show=False, copy=True,
                                T1=None, brainmask='brainmask.mgz', verbose=None)

    # Correction for wrong bem on the lower surface due to the bounding box around
    # the brain  on the MRI being too small
    correct_lower_surface_cut(subject, fs_subject_dir=fs_subject_dir, recompute=recompute)

    # Decimate BEM surfaces
    decimate_surface_wrapper(subject, fs_subject_dir=fs_subject_dir, recompute=recompute)

    # Make the .sphere files
    computing_sphere_files(subject, fs_subject_dir=fs_subject_dir, recompute=recompute)

    # Correct head surfaces that intersects
    correct_intersecting_meshes(Path(fs_subject_dir), fs(subject), suffix="")

    # Compute MNE head model artifacts up to the BEM solution
    build_mne_bem(fs(subject), fs_subject_dir=fs_subject_dir, recompute=recompute)

    compute_and_save_mri_fid(fs(subject), fs_subject_dir=fs_subject_dir, recompute=recompute)

    '''compute_and_save_trans(fs(subject), epochs_clean, fs_subject_dir=fs_subject_dir, 
                           recompute=recompute)

    compute_and_save_fwd(fs(subject), epochs_clean.info, 
                         fs_subject_dir=fs_subject_dir, recompute=recompute)

    select_sources(subject, fs_subject_dir=fs_subject_dir, recompute=recompute)'''


def compute_and_save_mri_fid(fs_subject, fs_subject_dir="/usr/local/freesurfer/subjects/", 
                             coord_frame="mri", recompute=False):
    fid_path = Path(fs_subject_dir) / fs_subject / "bem" / f"{fs_subject}-fiducials.fif"

    if fid_path.exists() and not recompute:
        return

    try:
        fid_path.unlink()
    except FileNotFoundError:
        pass

    digs_mri = mne.coreg.get_mni_fiducials(fs_subject, subjects_dir=fs_subject_dir)
    mne.io.meas_info.write_fiducials(fid_path, digs_mri, coord_frame=coord_frame)


def fit_fiducials(head_pts, mri_pts, n_scale_params=0, weights=(1.0, 10.0, 1.0),
                  parameters=None):
    """Find rotation and translation to fit all 3 fiducials."""
    if parameters is None:
        parameters = [0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0]

    if n_scale_params == 0:
        mri_pts *= parameters[6:9]  # not done in fit_matched_points
    x0 = np.array(parameters[:6 + n_scale_params])
    est = mne.coreg.fit_matched_points(mri_pts.astype(float), head_pts, x0=x0, out='params',
                                       scale=n_scale_params, weights=weights)
    if n_scale_params == 0:
        parameters[:6] = est
    else:
        parameters[:] = np.concatenate([est, [est[-1]] * 2])

    return parameters


def compute_and_save_trans(fs_subject, epochs, fs_subject_dir="/usr/local/freesurfer/subjects/",
                           overwrite=True, recompute=False):

    path_out = Path(mne.coreg.trans_fname.format(raw_dir=raw_dir, subject=fs_subject))
    if path_out.exists() and not recompute:
        return

    head_pts = np.array([dig["r"] for dig in epochs.info["dig"][:3]])

    raw_dir = Path(fs_subject_dir) / fs_subject / "bem"
    digs_mri = mne.io.meas_info.read_fiducials(raw_dir / f"{fs_subject}-fiducials.fif")[0]
    mri_pts = np.array([dig["r"] for dig in digs_mri])

    parameters = fit_fiducials(head_pts, mri_pts)
    trans = mne.transforms.rotation(*parameters[:3]).T
    trans[:3, 3] = -np.dot(trans[:3, :3], parameters[3:6])
    trans = mne.Transform('head', 'mri', trans)

    mne.write_trans(path_out, trans, overwrite=overwrite)


def get_epochs_with_montage(subject, epoch_dir="/media/christian/ElementsSE/lewis/EEG_epoched/saccade_rejected/",
                            verbose=False):
    montage = mne.channels.make_standard_montage('biosemi64')
    epoch_path = Path(epoch_dir) / f"{subject}-epo.fif"
    return mne.read_epochs(str(epoch_path), verbose=False).set_montage(montage)


def compute_and_save_fwd(fs_subject, info, fs_subject_dir="/usr/local/freesurfer/subjects/",
                         recompute=False):
    bem_root = Path(fs_subject_dir) / fs_subject / "bem"
    path_out = bem_root / f"{fs_subject}-fwd.fif"
    if path_out.exists() and not recompute:
        return

    src = mne.read_source_spaces(bem_root / f"{fs_subject}-src.fif")
    bem_sol = mne.read_bem_solution(bem_root / f"{fs_subject}-bem-sol.fif")
    trans = mne.read_trans(bem_root / f"{fs_subject}-trans.fif")

    fwd = mne.make_forward_solution(info, trans, src, bem_sol, mindist=0)
    mne.write_forward_solution(path_out, fwd, overwrite=True)


def validate_coregistration(fs_subject, info, fs_subject_dir="/usr/local/freesurfer/subjects/", save=True, trans=None):

    bem_root = Path(fs_subject_dir) / fs_subject / "bem"
    src = mne.read_source_spaces(bem_root / f"{fs_subject}-src.fif")
    bem_sol = mne.read_bem_solution(bem_root / f"{fs_subject}-bem-sol.fif")
    if trans is None:
        trans = mne.read_trans(bem_root / f"{fs_subject}-trans.fif")

    fig = mne.viz.plot_alignment(info, trans=trans,
                                 subject=fs_subject,
                                 subjects_dir=fs_subject_dir, surfaces='head',
                                 show_axes=True, dig="fiducials", eeg="projected",
                                 coord_frame='mri', mri_fiducials=True,
                                 src=src, bem=bem_sol)

    fig.plotter.off_screen = True

    mne.viz.set_3d_view(figure=fig, azimuth=135, elevation=80, distance=0.6)
    if save:
        fig.plotter.screenshot(f"coregistration_{fs_subject}_1.png")

        mne.viz.set_3d_view(figure=fig, azimuth=45, elevation=80, distance=0.6)
        fig.plotter.screenshot(f"coregistration_{fs_subject}_2.png")

        mne.viz.set_3d_view(figure=fig, azimuth=270, elevation=80, distance=0.6)
        fig.plotter.screenshot(f"coregistration_{fs_subject}_3.png")


def get_source_generator(subject, lambda2=0.1, loose=0.0, inv_method="eLORETA", tmin=None, tmax=None,
                         fs_subject_dir="/usr/local/freesurfer/subjects/", return_epochs=False,
                         epoch_dir="/media/christian/ElementsSE/lewis/EEG_epoched/saccade_rejected/",
                         clear_proj_info=True):

    epochs_clean = get_epochs_with_montage(subject, epoch_dir=epoch_dir).pick("eeg")
    if clear_proj_info:
        #epochs_clean.info["projs"] = []
        for proj in epochs_clean.info['projs']:
            proj["active"] = False        
        epochs_clean.del_proj()
        print("Projections deleted.")

    fs_subject = fs(subject)
    bem_root = Path(fs_subject_dir) / fs(subject) / "bem"
    fwd = mne.read_forward_solution(bem_root / f"{fs_subject}-fwd.fif", verbose=False)

    # We use a diagonal matrix because the covariance will capture the zero-lag connectivity
    # and we don't want this to be modeled in the noise covariance matrix
    noise_cov = mne.compute_covariance(epochs_clean, tmax=0.0, method="auto").as_diag()

    if tmin is not None or tmax is not None:
        epochs_clean.crop(tmin=tmin, tmax=tmax)

    # Compute inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs_clean.info, fwd, noise_cov, loose=loose)
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs_clean, inverse_operator, lambda2, method=inv_method,
                                                 pick_ori=None, return_generator=True)
    if return_epochs:
        return stcs, epochs_clean
    else:
        return stcs


def save_sources(subject, source_dir="/media/christian/ElementsSE/lewis/sources", inv_method="eLORETA", **kwargs):
    stcs = get_source_generator(subject, inv_method=inv_method, **kwargs)
    for no_epoch, stc in tqdm(enumerate(stcs)):
        stc.save(Path(source_dir) / f"{subject}_{inv_method}_source_estimate_{no_epoch}", ftype="h5")


def select_sources(subject, n_sample=100, fs_subject_dir="/usr/local/freesurfer/subjects/",
                   recompute=False):

    path_out = bem_root / f"{subject}_selected_sources_vert.nc"
    if path_out.exists() and not recompute:
        return

    fs_subject = fs(subject)
    bem_root = Path(fs_subject_dir) / fs(subject) / "bem"
    fwd = mne.read_forward_solution(bem_root / f"{fs_subject}-fwd.fif")

    nuse_left = fwd["src"][0]["nuse"]
    nuse_right = fwd["src"][1]["nuse"]

    offset_left = fwd["src"][0]["rr"][fwd["src"][0]["vertno"]]
    offset_right = fwd["src"][1]["rr"][fwd["src"][1]["vertno"]]

    left_norm = fwd["src"][0]["nn"][fwd["src"][0]["vertno"]]
    right_norm = fwd["src"][1]["nn"][fwd["src"][1]["vertno"]]

    offset = np.concatenate((offset_left, offset_right)).mean(0)

    left = offset_left - offset
    right = offset_right - offset

    # Left-right correspondance
    tree = KDTree(left)
    left_inds = np.array([tree.query([-x, y, z])[1] for x, y, z in right])
    right_inds = np.arange(nuse_right)

    # Anteroposterior correspondance
    tree_left = KDTree(left[left_inds])
    dist_ap, inds_ant, inds_post = zip(
        *[(*tree_left.query([x, -y, z]), i) for i, (x, y, z) in enumerate(left[left_inds]) if y > 0])

    # This tend to select point around the interhemispheric fissure
    # inds_ant = np.array(inds_ant)[np.argsort(dist_ap)][:100]
    # inds_post = np.array(inds_post)[np.argsort(dist_ap)][:100]
    np.random.seed(4325467)
    selection = np.random.choice(np.arange(len(dist_ap)), 100, replace=False)
    inds_ant = np.array(inds_ant)[selection]
    inds_post = np.array(inds_post)[selection]

    ant_left = offset_left[left_inds][np.array(inds_ant)]
    post_left = offset_left[left_inds][np.array(inds_post)]
    ant_right = offset_right[right_inds][np.array(inds_ant)]
    post_right = offset_right[right_inds][np.array(inds_post)]

    ant_left_norm = left_norm[left_inds][np.array(inds_ant)]
    post_left_norm = left_norm[left_inds][np.array(inds_post)]
    ant_right_norm = right_norm[right_inds][np.array(inds_ant)]
    post_right_norm = right_norm[right_inds][np.array(inds_post)]

    channels = ([str(i) + "-ant-lh" for i in np.arange(n_sample)] +
                [str(i) + "-pos-lh" for i in np.arange(n_sample)] +
                [str(i) + "-ant-rh" for i in np.arange(n_sample)] +
                [str(i) + "-pos-rh" for i in np.arange(n_sample)])

    dat = np.vstack([ant_left_norm, post_left_norm, ant_right_norm, post_right_norm])
    ind_source_norm = xr.DataArray(dat,
                                   dims=("channel", "coord"),
                                   coords={"channel": channels,
                                           "coord": ["x", "y", "z"]})

    dat = np.vstack([ant_left, post_left, ant_right, post_right])
    ind_source_pos = xr.DataArray(dat,
                                  dims=("channel", "coord"),
                                  coords={"channel": channels,
                                          "coord": ["x", "y", "z"]})

    source_no = np.concatenate([left_inds[np.array(inds_ant)],
                                left_inds[np.array(inds_post)],
                                nuse_left + right_inds[np.array(inds_ant)],
                                nuse_left + right_inds[np.array(inds_post)]])
    source_no = xr.DataArray(source_no,
                             dims=("channel"),
                             coords={"channel": channels})

    with xr.Dataset({"norm": ind_source_norm, "pos": ind_source_pos, "source_no": source_no}) as vert_dataset:
        vert_dataset.to_netcdf(path_out)


def get_fitted_positions(subject, fs_subject_dir="/usr/local/freesurfer/subjects/",
                         epoch_dir="/media/christian/ElementsSE/lewis/EEG_epoched/saccade_rejected/"):
    fs_subject = fs(subject)
    bem_root = Path(fs_subject_dir) / fs(subject) / "bem"
    bem = mne.read_bem_solution(bem_root / f"{fs_subject}-bem-sol.fif")
    trans = mne.read_trans(bem_root / f"{fs_subject}-trans.fif")
    epochs_clean = get_epochs_with_montage(subject, epoch_dir=epoch_dir)

    eeg_picks = mne.pick_types(epochs_clean.info, meg=False, eeg=True, ref_meg=False)
    head_trans = mne.transforms._get_trans(trans, 'head', 'mri')[0]
    head_surf = mne.bem._bem_find_surface(bem, 'head')
    eeg_loc = np.array([epochs_clean.info['chs'][k]['loc'][:3] for k in eeg_picks])
    eeg_loc = mne.transforms.apply_trans(head_trans, eeg_loc)
    return pd.DataFrame(mne.surface._project_onto_surface(eeg_loc, head_surf, project_rrs=True, return_nn=True)[2],
                        index=np.array(epochs_clean.ch_names)[eeg_picks], columns=["x", "y", "z"])


def get_montage_ratio(ch_pos):
    x, y = ch_pos.T[:2]
    return (x.max()-x.min())/(y.max()-y.min())


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Run NRRD-based head model pipeline.")
    parser.add_argument("--subject", required=True, help="Subject identifier.")
    parser.add_argument(
        "--fs-subject-dir",
        default="/usr/local/freesurfer/subjects/",
        help="FreeSurfer subjects directory.",
    )
    parser.add_argument(
        "--civet-surface-dir",
        default="CIVETv2.0_Surfaces/NATIVE",
        help="CIVET surface directory.",
    )
    parser.add_argument(
        "--t1-nrrd-path",
        default="stx_ibis_210772_V06_T1w_dBias.nrrd",
        help="Path to T1 NRRD volume used for preprocessing.",
    )
    parser.add_argument(
        "--epoch-dir",
        default="/media/christian/ElementsSE/lewis/EEG_epoched/saccade_rejected/",
        help="Directory containing EEG epoch files.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute outputs even if they exist.",
    )
    return parser.parse_args()


def main():
    args = _parse_cli_args()
    build_head_model(
        subject=args.subject,
        fs_subject_dir=args.fs_subject_dir,
        civet_surface_dir=args.civet_surface_dir,
        t1_nrrd_path=args.t1_nrrd_path,
        epoch_dir=args.epoch_dir,
        recompute=args.recompute,
    )


if __name__ == "__main__":
    main()
