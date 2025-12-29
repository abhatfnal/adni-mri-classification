function rpet_path = coregister(mri_path, pet_path)

    % Convert to char for SPM
    mri_path = char(mri_path);
    pet_path = char(pet_path);

    % Add SPM and initialize
    spm_path = char(getenv('SPM_PATH'));
    addpath(spm_path);
    spm('Defaults','PET');
    spm_jobman('initcfg');

    % If any input is .nii.gz, SPM may not read it -> gunzip
    if endsWith(mri_path, '.nii.gz')
        gunzip(mri_path, fileparts(mri_path));
        mri_path = mri_path(1:end-3);
    end
    if endsWith(pet_path, '.nii.gz')
        gunzip(pet_path, fileparts(pet_path));
        pet_path = pet_path(1:end-3);
    end

    % One batch: estimate + reslice
    matlabbatch = [];
    matlabbatch{1}.spm.spatial.coreg.estwrite.ref    = {[mri_path ',1']};
    matlabbatch{1}.spm.spatial.coreg.estwrite.source = {[pet_path ',1']};
    matlabbatch{1}.spm.spatial.coreg.estwrite.other  = {''};

    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep      = [4 2];
    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol      = ...
        [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm     = [7 7];

    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = 4;
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap   = [0 0 0];
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask   = 0;
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';

    spm_jobman('run', matlabbatch);

    % Output path: r<petname>.nii in the same folder
    [p,n,e] = fileparts(pet_path);
    rpet_path = fullfile(p, ['r' n e]);

    % Sanity check
    if ~exist(rpet_path, 'file')
        error('Coreg reslice did not produce output: %s', rpet_path);
    end
end
