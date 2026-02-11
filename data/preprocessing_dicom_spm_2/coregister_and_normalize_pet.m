function out_nii = coregister_and_normalize_pet(mri_path, pet_path, deformation_field_path)
% Robust PET -> MRI coregistration and normalization pipeline for ADNI data.
%
% This version includes a critical pre-alignment step:
% 1. Creates a temporal mean if the input PET is 4D.
% 2. Sets the origin of BOTH the MRI and PET to their image centers.
% 3. Performs co-registration on these centered images for stability.
% 4. Applies the resulting transformation to the ORIGINAL PET scan.
% 5. Normalizes the final resliced PET scan to MNI space.

    % --- Add SPM to MATLAB's path ---
    addpath(genpath(getenv('SPM_PATH')));
    
    % --- Input Validation ---
    if ~exist(mri_path,'file'); error('MRI not found: %s', mri_path); end
    if ~exist(pet_path,'file'); error('PET not found: %s', pet_path); end
    if ~exist(deformation_field_path,'file'); error('Deformation not found: %s', deformation_field_path); end
    
    spm('defaults','pet');
    spm_jobman('initcfg');

    % --- Step 1: Handle 4D PET by creating a temporal mean ---
    pet_static = make_pet_static(pet_path);

    % --- Step 2: Create temporary copies to work with ---
    [pet_dir, pet_name, pet_ext] = fileparts(pet_static);
    temp_pet_path = fullfile(pet_dir, ['temp_for_coreg_' pet_name pet_ext]);
    copyfile(pet_static, temp_pet_path);
    
    % --- Step 3: Set origin of BOTH MRI and PET to their center for stability ---
    % Note: We use a temporary copy of the MRI for this, so the original is untouched.
    [mri_dir, mri_name, mri_ext] = fileparts(mri_path);
    temp_mri_path = fullfile(pet_dir, ['temp_mri_for_coreg_' mri_name mri_ext]);
    copyfile(mri_path, temp_mri_path);
    set_origin_to_center(temp_mri_path);
    set_origin_to_center(temp_pet_path);

    % --- Step 4: Coregister (Estimate only) ---
    % This finds the transformation and applies it to the HEADER of temp_pet_path
    matlabbatch = {};
    matlabbatch{1}.spm.spatial.coreg.estimate.ref = {[temp_mri_path ',1']};
    matlabbatch{1}.spm.spatial.coreg.estimate.source = {[temp_pet_path ',1']};
    matlabbatch{1}.spm.spatial.coreg.estimate.other = {''};
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
    spm_jobman('run', matlabbatch);

    % --- Step 5: Reslice (Write) the PET scan ---
    % THIS IS THE CRITICAL FIX: The source for reslicing MUST be the
    % file that was modified in the estimation step (temp_pet_path).
    matlabbatch = {};
    matlabbatch{1}.spm.spatial.coreg.write.ref = {[mri_path ',1']};
    matlabbatch{1}.spm.spatial.coreg.write.source = {[temp_pet_path ',1']}; % USE THE MODIFIED TEMP FILE
    matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 4;
    matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.coreg.write.roptions.mask = 0;
    matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = 'r_';
    spm_jobman('run', matlabbatch);

    resliced_pet_path = fullfile(pet_dir, ['r_' pet_name pet_ext]);
    if ~exist(resliced_pet_path, 'file')
        % Sometimes SPM uses the basename of the source, not the original
        resliced_pet_path = fullfile(pet_dir, ['r_temp_for_coreg_' pet_name pet_ext]);
        if ~exist(resliced_pet_path, 'file')
            error('Reslicing failed to create output file.');
        end
    end

    % --- Step 6: Normalize the newly resliced PET scan ---
    matlabbatch = {};
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {deformation_field_path};
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {[resliced_pet_path ',1']};
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-90 -126 -72; 90 90 108];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w_';
    spm_jobman('run', matlabbatch);
    
    [~, resliced_name, resliced_ext] = fileparts(resliced_pet_path);
    out_nii = fullfile(pet_dir, ['w_' resliced_name resliced_ext]);
    
    % --- Cleanup temporary files ---
    delete(temp_pet_path);
    delete(temp_mri_path);
end


function pet_static = make_pet_static(pet_in)
    V = spm_vol(pet_in);
    if numel(V) == 1
        pet_static = pet_in;
        return;
    end
    [pp, bb, ~] = fileparts(pet_in);
    out = fullfile(pp, ['mean_' bb '.nii']);
    
    Y = spm_read_vols(V);
    Y_mean = mean(Y, 4); 
    
    V_out = V(1);
    V_out.fname = out;
    V_out.dt = [spm_type('float32'), spm_platform('bigend')];
    spm_write_vol(V_out, Y_mean);
    
    pet_static = out;
end

function set_origin_to_center(nifti_path)
    V = spm_vol(nifti_path);
    if isempty(V)
        return;
    end
    mat = V(1).mat;
    center_vox = (V(1).dim' + 1) / 2; 
    center_mm = mat * [center_vox; 1];
    mat(1:3, 4) = mat(1:3, 4) - center_mm(1:3);
    spm_get_space(V(1).fname, mat);
end
