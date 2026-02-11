function ensure_deformation_field(mri_path)
    [d,b,~] = fileparts(mri_path);
    y_path  = fullfile(d, ['y_'  b '.nii']);
    iy_path = fullfile(d, ['iy_' b '.nii']);
    if exist(y_path,'file') || exist(iy_path,'file'); return; end

    spm('defaults','fmri'); spm_jobman('initcfg');
    tpm = fullfile(spm('Dir'),'tpm','TPM.nii');

    matlabbatch = {};
    matlabbatch{1}.spm.spatial.preproc.channel.vols     = {[mri_path ',1']};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg  = 0.001;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{1}.spm.spatial.preproc.channel.write    = [0 0];

    ngaus = [1 1 2 3 4 2];
    for k = 1:6
        matlabbatch{1}.spm.spatial.preproc.tissue(k).tpm    = {sprintf('%s,%d', tpm, k)};
        matlabbatch{1}.spm.spatial.preproc.tissue(k).ngaus  = ngaus(k);
        matlabbatch{1}.spm.spatial.preproc.tissue(k).native = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(k).warped = [0 0];
    end
    matlabbatch{1}.spm.spatial.preproc.warp.mrf     = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.reg     = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg  = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm    = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.samp    = 3;
    matlabbatch{1}.spm.spatial.preproc.warp.write   = [1 1];
    spm_jobman('run', matlabbatch);
end
