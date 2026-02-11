function preprocess_pet(pet_folder, bc_mri_path, deformation_field_path, brain_mask_path)
    % bc_mri_path: Path of bias corrected MRI 
    
    % Make static PET
    static_pet_path = make_pet_static(pet_folder);

    % Create copy of deformation field
    [folder, name, ext] = fileparts(deformation_field_path);
    copyfile(deformation_field_path, pet_folder);
    deformation_field_path = fullfile(pet_folder, [name ext]);

    % Create copy of bias corrected mri
    [folder, name, ext] = fileparts(bc_mri_path);
    copyfile(bc_mri_path, pet_folder);
    bc_mri_path = fullfile(pet_folder, [name ext]);

    % Create copy of brain mask
    [folder, name, ext] = fileparts(brain_mask_path);
    copyfile(brain_mask_path, pet_folder);
    brain_mask_path = fullfile(pet_folder, [name ext]);

    % Set origin to center of gravity just for PET
    set_origin_to_cog(static_pet_path);

    % Coregister PET to temp MRI
    coreg_pet_path = coregister(bc_mri_path, static_pet_path);

    % Load PET volume
    coreg_pet_vol = spm_vol(coreg_pet_path);
    pet = spm_read_vols(coreg_pet_vol);

    % Load PET brain mask
    brain_mask_vol = spm_vol(brain_mask_path);
    brain_mask = spm_read_vols(brain_mask_vol);

    % Apply mask to PET
    pet(~brain_mask) = 0;

    % Save masked PET
    masked_pet_path = spm_file(coreg_pet_vol.fname, 'prefix', 'masked_');
    coreg_pet_vol.fname = masked_pet_path;
    spm_write_vol(coreg_pet_vol, pet);

    % Warp PET to MNI
    w_masked_pet_path = normalize(masked_pet_path, deformation_field_path);

    % Warp mask to MNI
    w_brain_mask_path = normalize(brain_mask_path, deformation_field_path, 0);

    % Load warped mask + image
    w_brain_mask = spm_read_vols(spm_vol(w_brain_mask_path)) > 0.5;
    w_img      = spm_read_vols(spm_vol(w_masked_pet_path));

    % Clean artifacts
    w_img(isnan(w_img)) = 0;
    w_img(~w_brain_mask)  = 0;
    w_img(w_img < 0)    = 0;

    % 0-1 normalization
    w_img_max = max(w_img(:));
    w_img = w_img ./ (w_img_max + eps);

    % Save final clean image next to normalized image
    [folder, name, ext] = fileparts(w_masked_pet_path);  % name includes the w_ prefix
    clean_path = fullfile(pet_folder, ['clean_' name ext]);

    w_vol = spm_vol(w_masked_pet_path); % header from normalized image
    w_vol.fname = clean_path;
    spm_write_vol(w_vol, w_img);


    % ------------------------------------------------------------------
    % DELETE ALL OTHER WORKING FILES: keep only final masked, normalized PET
    % ------------------------------------------------------------------
    keep = { clean_path };

    d = dir(folder);
    for i = 1:numel(d)
        if d(i).isdir, continue; end
        fp = fullfile(folder, d(i).name);
        if any(strcmp(fp, keep))
            continue;
        end
        try
            delete(fp);
        catch
            % ignore delete failures (e.g., file locked); optional: fprintf warning
        end
    end

end