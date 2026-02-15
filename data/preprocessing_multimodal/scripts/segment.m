function out = segment(original_mri_nii)
% SEGMENT Segments MRI scan with SPM and returns key output paths.
%
% out.y   = deformation field (y_*.nii)
% out.m   = bias-corrected image (m*.nii)
% out.c1  = GM prob map (c1*.nii)
% out.c2  = WM prob map (c2*.nii)
% out.c3  = CSF prob map (c3*.nii)

    % Make char for SPM ---
    original_mri_nii = char(original_mri_nii);

    % Check file existence
    if ~exist(original_mri_nii, 'file')
        error('File not found: %s', original_mri_nii);
    end

    % If input is .nii.gz, decompress to .nii
    if endsWith(original_mri_nii, '.nii.gz')
        outdir = fileparts(original_mri_nii);
        gunzip(original_mri_nii, outdir);
        original_mri_nii = original_mri_nii(1:end-3); % ".nii.gz" -> ".nii"
    end

    % Add SPM path + init
    addpath(genpath(char(getenv('SPM_PATH'))));
    spm('Defaults','FMRI');
    spm_jobman('initcfg');

    % TPM path (robust)
    tpm_path = fullfile(spm('Dir'), 'tpm', 'TPM.nii');

    matlabbatch = {};
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {[original_mri_nii ',1']};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{1}.spm.spatial.preproc.channel.write = [0 1];  % save bias-corrected (m*), not bias field

    ng = [1 1 2 3 4 2];
    for i = 1:6
        matlabbatch{1}.spm.spatial.preproc.tissue(i).tpm   = {sprintf('%s,%d', tpm_path, i)};
        matlabbatch{1}.spm.spatial.preproc.tissue(i).ngaus = ng(i);
        matlabbatch{1}.spm.spatial.preproc.tissue(i).warped = [0 0];

        if i <= 3
            matlabbatch{1}.spm.spatial.preproc.tissue(i).native = [1 0]; % c1/c2/c3
        else
            matlabbatch{1}.spm.spatial.preproc.tissue(i).native = [0 0];
        end
    end

    matlabbatch{1}.spm.spatial.preproc.warp.mrf     = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.reg     = [0 0 0.1 0.01 0.4];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg  = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm    = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.samp    = 3;
    matlabbatch{1}.spm.spatial.preproc.warp.write   = [0 1];  % save y_*.nii only

    spm_jobman('run', matlabbatch);

    % Build expected output filenames
    [p,n,e] = fileparts(original_mri_nii);
    out.y  = fullfile(p, ['y_' n e]);
    out.m  = fullfile(p, ['m'  n e]);
    out.c1 = fullfile(p, ['c1' n e]);
    out.c2 = fullfile(p, ['c2' n e]);
    out.c3 = fullfile(p, ['c3' n e]);

    % Optional: sanity check they exist
    fns = {'y','m','c1','c2','c3'};
    for k = 1:numel(fns)
        if ~exist(out.(fns{k}), 'file')
            warning('Expected output not found: %s', out.(fns{k}));
        end
    end
end
