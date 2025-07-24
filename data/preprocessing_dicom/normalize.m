function normalize(nii)
  % normalize_one  Normalize a single .nii file using SPM12
  % Usage: normalize_one('/path/to/file.nii');
  
  % Add SPM path
  addpath('/project/aereditato/cestari/spm/spm');
  
  % Initialize SPM
  spm('Defaults','FMRI');
  spm_jobman('initcfg');
  
  % Check that the file exists
  if ~exist(nii, 'file')
      error('File not found: %s', nii);
  end
  
  % Build the normalization batch
  matlabbatch = [];
  matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol      = { [nii ',1'] };
  matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = { [nii ',1'] };
  
  % Estimation options
  matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasreg  = 0.0001;
  matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;
  matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm     = { '/project/aereditato/cestari/spm/spm/tpm/TPM.nii' };
  matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.affreg  = 'mni';
  matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.reg     = [0 0 0.1 0.01 0.4];
  matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.fwhm    = 0;
  matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.samp    = 3;
  
  % Writing options
  matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.bb      = [-78 -112 -70; 78 76 85];
  matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.vox     = [2 2 2];
  matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.interp = 4;
  matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.prefix = 'w';
  
  % Execute the job
  spm_jobman('run', matlabbatch);
  
  fprintf('Normalized: %s\n', nii);
  end
  