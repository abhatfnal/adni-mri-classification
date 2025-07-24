function normalize(nii)

    spm_path = '/project/aereditato/cestari/spm/spm'
    tpm_path = '/project/aereditato/cestari/spm/spm/tpm/TPM.nii';
    
    % Add SPM path
    addpath(spm_path);
    spm('Defaults','FMRI');
    spm_jobman('initcfg');
  
    % Check file exists
    if ~exist(nii, 'file')
        error('File not found: %s', nii);
    end
  
    [folder, name, ~] = fileparts(nii);
  
    % --- 1. Segmentation: bias correction + tissue probability maps + deformation field
    matlabbatch = [];
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {nii};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{1}.spm.spatial.preproc.channel.write = [1 1];  % save m*.nii and bias field
  
    for i = 1:6
      matlabbatch{1}.spm.spatial.preproc.tissue(i).tpm = {sprintf('%s,%d', tpm_path, i)};
      matlabbatch{1}.spm.spatial.preproc.tissue(i).ngaus = 1;
      matlabbatch{1}.spm.spatial.preproc.tissue(i).native = [1 0];  % save native space tissue maps
      matlabbatch{1}.spm.spatial.preproc.tissue(i).warped = [0 0];
    end
  
    matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0 0.1 0.01 0.4];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{1}.spm.spatial.preproc.warp.write = [1 1];  % save y_*.nii and inverse
  
    spm_jobman('run', matlabbatch);
  
    % --- 2. Create brain mask: c1 + c2 > 0.2
    c1 = spm_read_vols(spm_vol(fullfile(folder, ['c1' name '.nii'])));
    c2 = spm_read_vols(spm_vol(fullfile(folder, ['c2' name '.nii'])));
    brain_mask = (c1 + c2) > 0.2;
  
    % --- 3. Apply mask to bias-corrected image
    m_vol = spm_vol(fullfile(folder, ['m' name '.nii']));
    m_img = spm_read_vols(m_vol);
    m_img(~brain_mask) = 0;
  
    % --- 4. Save skull-stripped bias-corrected image to temp file
    masked_path = fullfile(folder, ['masked_' name '.nii']);
    m_vol.fname = masked_path;
    spm_write_vol(m_vol, m_img);
  
    % --- 5. Normalize masked image to MNI using y_*.nii
    y_field = fullfile(folder, ['y_' name '.nii']);
    matlabbatch2{1}.spm.spatial.normalise.write.subj.def = {y_field};
    matlabbatch2{1}.spm.spatial.normalise.write.subj.resample = {masked_path};
    matlabbatch2{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70; 78 76 85];
    matlabbatch2{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    matlabbatch2{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch2{1}.spm.spatial.normalise.write.woptions.prefix = 'w';
  
    spm_jobman('run', matlabbatch2);
  
    fprintf('Finished preprocessing: %s\n', nii);
  end
  