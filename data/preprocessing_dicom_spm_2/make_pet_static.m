function out = make_pet_static(pet_path)
% If PET is 4D, write mean volume as mean_<name>.nii; else return original.

    V = spm_vol(pet_path);
    if numel(V) == 1
        out = pet_path;
        return;
    end

    % Accumulate mean without blowing memory
    Ysum = 0; 
    for i = 1:numel(V)
        Yi = spm_read_vols(V(i));
        Ysum = Ysum + Yi;
    end
    Y = Ysum / numel(V);

    Vout      = V(1);
    Vout.fname = fullfile(fileparts(pet_path), ['mean_' spm_file(pet_path,'basename') '.nii']);
    Vout.dt    = [16 0];  % float32
    Vout.n     = [1 1];

    spm_write_vol(Vout, Y);
    out = Vout.fname;
end
