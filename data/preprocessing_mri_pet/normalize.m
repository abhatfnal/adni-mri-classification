function normalized_img = normalize(img_path, deformation_field_path)
    
    img_path = char(img_path);
    deformation_field_path = char(deformation_field_path);

    addpath(genpath(char(getenv('SPM_PATH'))));
    spm('Defaults','PET');
    spm_jobman('initcfg');
    
    % Normalize the input image to MNI space using the provided deformation field
    matlabbatch = {};
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {deformation_field_path};
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {[img_path ',1']};
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-90 -126 -72; 90 90 108];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w_';

    [~, name, ext] = fileparts(img_path);
    normalized_img = fullfile(fileparts(img_path), ['w_' name ext]);

    spm_jobman('run', matlabbatch);
end