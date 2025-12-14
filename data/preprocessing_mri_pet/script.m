original_mri_path = './temp/mri/I129847.nii.gz';  % Original MRI path
pet_frames_dir = './temp/pet/';  % Directory with PET frames


% % Create static PET by averaging frames
static_pet_path = make_pet_static(pet_frames_dir);

% Set origin to center of gravity for MRI and PET
set_origin_to_cog(original_mri_path);
set_origin_to_cog(static_pet_path);

% % Segment MRI
segment_outputs = segment(original_mri_path);

% segment_outputs.m = './temp/mri/mI129847.nii';
% segment_outputs.c1 = './temp/mri/c1I129847.nii';
% segment_outputs.c2 = './temp/mri/c2I129847.nii';
% segment_outputs.c3 = './temp/mri/c3I129847.nii';
% segment_outputs.y = './temp/mri/y_I129847.nii';
% static_pet_path = './temp/pet/static_mean_pet.nii';

% Coregister PET to MRI
coreg_pet_path = coregister(segment_outputs.m, static_pet_path);

% Create brain masks
c1 = spm_read_vols(spm_vol(segment_outputs.c1));
c2 = spm_read_vols(spm_vol(segment_outputs.c2));
c3 = spm_read_vols(spm_vol(segment_outputs.c3));

brain_mask_mri = (c1 + c2) > 0.3;
brain_mask_pet = (c1 + c2 + c3) > 0.5;

% Load bias-corrected MRI and coregistered PET volumes
bc_mri_vol = spm_vol(segment_outputs.m);
static_pet_vol = spm_vol(coreg_pet_path);

% Apply mask to MRI and PET
mri = spm_read_vols(bc_mri_vol);
static_pet = spm_read_vols(static_pet_vol);

mri(~brain_mask_mri) = 0;
static_pet(~brain_mask_pet) = 0;

%Save masked images
masked_mri_path = './temp/mri/masked_mI129847.nii';
masked_pet_path = './temp/pet/masked_static_mean_pet.nii';

bc_mri_vol.fname = masked_mri_path;
static_pet_vol.fname = masked_pet_path;

spm_write_vol(bc_mri_vol, mri);
spm_write_vol(static_pet_vol, static_pet);

% Normalize to MNI space
normalized_mri_path = normalize(masked_mri_path, segment_outputs.y);
normalized_pet_path = normalize(masked_pet_path, segment_outputs.y);
