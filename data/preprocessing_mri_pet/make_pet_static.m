function static_pet = make_pet_static(nii_dir)
% MAKE_PET_STATIC  Merge all 3D NIfTI volumes in nii_dir into 4D, then average to static PET.
% Returns full path to the static PET NIfTI.

    nii_dir = char(nii_dir);

    % SPM setup (needed if not already done in your session)
    addpath(genpath(char(getenv('SPM_PATH'))));
    spm('Defaults','PET');
    spm_jobman('initcfg');

    % List .nii files
    files = dir(fullfile(nii_dir, '*.nii'));
    if isempty(files)
        error('No .nii files found in directory: %s', nii_dir);
    end

    % Build N×M char matrix: one file per row (SPM convention)
    file_list = char(fullfile(nii_dir, {files.name})');

    % Merge into 4D
    merged_nii = fullfile(nii_dir, 'merged_4D.nii');
    spm_file_merge(file_list, merged_nii);

    % Load 4D header list
    V = spm_vol(merged_nii);

    % If already single volume, just return it
    if numel(V) == 1
        static_pet = merged_nii;
        return;
    end

    % Accumulate mean (single pass)
    Ysum = [];
    for i = 1:numel(V)
        Yi = spm_read_vols(V(i));
        if isempty(Ysum)
            Ysum = zeros(size(Yi), 'like', Yi);
        end
        Ysum = Ysum + Yi;
    end
    Ymean = Ysum / numel(V);

    % Write static PET
    Vout = V(1);
    static_pet = fullfile(nii_dir, 'static_mean_pet.nii');
    Vout.fname = static_pet;
    Vout.dt    = [16 0];   % float32
    Vout.n     = [1 1];

    spm_write_vol(Vout, Ymean);

    fprintf('Static PET written to:\n  %s\n', static_pet);
end
