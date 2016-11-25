
import mne
import surface
subject = 'subject_04'

data_dir = '/hpc/comco/brovelli.a/db_brainvisa/meg_te/subject_04/t1mri/default_acquisition/default_analysis/segmentation/mesh/'

trans_fname = '/hpc/comco/brovelli.a/db_mne/meg_te/subject_04/referential/subject_04-trans.trm'
# White matter
# surface_master = surface.get_surface((data_dir + 'subject_04_Lwhite.gii') , subject, 'lh', trans_fname)
# Grey matter
surface_master = surface.get_surface((data_dir + 'subject_04_Lhemi.gii'), subject, 'lh', trans_fname)
# Plot surface after transformation
# surface_master.show('test', (0.3, 0.5, 0.8))

# Create parcellation from MarsAtlas
mne_dir = '/hpc/comco/brovelli.a/db_mne/meg_te/'
fname_tex = mne_dir + '{0}/tex/{0}_Lwhite_parcels_marsAtlas.gii'.format(subject)
fname_atl = mne_dir + 'label/MarsAtlas_BV_2015.xls'
fname_col = mne_dir + 'label/MarsAtlas.ima'
surfaces = surface.get_surface(surface_master, fname_tex, subject, 'lh', fname_atl, fname_col)

# Create volumetric parcellation from MarsAtlas
fname_mri = mne_dir + '{0}/tex/{0}_Lwhite_parcels_marsAtlas.gii'.format(subject)
fname_vol = subjects_dir + '{0}/vol/{0}_gyriVolume_deepStruct.nii.gz'.format(subject)
name_lobe_vol = ['Subcortical']
volumes = get_volume(fname_mri, fname_atl, name_lobe_vol, subject)

fname_atl = mne_dir + 'label/MarsAtlas_BV_2015.xls'
fname_col = mne_dir + 'label/MarsAtlas.ima'
surfaces = get_parcels(surface_master, fname_tex, 'subjcet_04', 'lh', fname_atl, fname_col)
# Test transformation
cd / hpc / comco / brovelli.a / db_mne / meg_te
import data

subject = 'subject_04'
# functional data
pdf_name = '{0}/functional/1/c,rfDC'.format(subject)
config_name = '{0}/functional/1/config'.format(subject)
head_shape_name = 'hs_file'.format(subject)

# anatomic data
fname_surf_L = '{0}/surf/{0}_Lwhite.gii'.format(subject)
fname_surf_R = '{0}/surf/{0}_Rwhite.gii'.format(subject)

fname_tex_L = '{0}/tex/{0}_Lwhite_parcels_marsAtlas.gii'.format(subject)
fname_tex_R = '{0}/tex/{0}_Rwhite_parcels_marsAtlas.gii'.format(subject)
fname_color = 'label/MarsAtlas.ima'

fname_vol = '{0}/vol/{0}_gyriVolume_deepStruct.nii.gz'.format(subject)
name_lobe_vol = ['Subcortical']

fname_atlas = 'label/MarsAtlas_BV_2015.xls'

# file to align coordinate frames
trans_fname = '{0}/trans/test1-trans.fif'.format(subject)

file_trans_ref = 'referential/referential.txt'
ref = '{0}/referential/{0}-trans.trm'

# create file transformation : BrainVisa to FreeSurfer'
fname = file_trans_ref
fname_out = ref
create_trans(subject, fname, fname_out)
