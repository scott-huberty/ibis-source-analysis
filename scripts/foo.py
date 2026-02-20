import mne

mne.viz.set_3d_backend("pyvista")
raw = mne.io.read_raw_fif("/Users/scotterik/devel/sMRI/eeg/STL7009_6m_20191209_095440/STL7009_6m_20191209_095440_proc-cleaned_raw.fif")
mon = mne.channels.make_standard_montage("GSN-HydroCel-129")
coreg = mne.coreg.Coregistration(raw.info, subject="sub-210772_ses-06", subjects_dir="/Users/scotterik/devel/sMRI/freesurfer", fiducials="estimated")

mne.viz.plot_alignment(info=raw.info, trans=coreg.trans, subject="sub-210772_ses-06", subjects_dir="/Users/scotterik/devel/sMRI/freesurfer", surfaces="head")
