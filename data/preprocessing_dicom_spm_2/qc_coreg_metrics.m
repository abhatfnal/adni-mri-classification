function qc_coreg_metrics(mri,rpet)
% Prints simple coreg metrics: center-of-mass delta and NCC inside a rough brain mask.
    addpath(genpath(getenv('SPM_PATH')));
    Vt = spm_vol(mri);  Vr = spm_vol(rpet);
    It = spm_read_vols(Vt); Ir = spm_read_vols(Vr);

    % crude brain mask from T1 (Otsu) just for metrics
    thr = graythresh(mat2gray(It));
    msk = It > thr*max(It(:));
    msk = imopen(msk, strel('sphere',2)); msk = imclose(msk, strel('sphere',2));

    % center of mass (in voxels)
    [x,y,z] = ndgrid(1:Vt.dim(1), 1:Vt.dim(2), 1:Vt.dim(3));
    comT = [sum(x(msk).*It(msk))/sum(It(msk)), ...
            sum(y(msk).*It(msk))/sum(It(msk)), ...
            sum(z(msk).*It(msk))/sum(It(msk))];
    comR = [sum(x(msk).*Ir(msk))/sum(Ir(msk)), ...
            sum(y(msk).*Ir(msk))/sum(Ir(msk)), ...
            sum(z(msk).*Ir(msk))/sum(Ir(msk))];
    dcom = norm((comT - comR).*abs(diag(Vt.mat(1:3,1:3)))');

    % normalized cross-correlation inside mask
    t = double(It(msk)); r = double(Ir(msk));
    t = (t - mean(t)) / std(t + eps); r = (r - mean(r)) / std(r + eps);
    ncc = mean(t .* r);

    fprintf('COM distance ≈ %.2f mm | NCC ≈ %.3f | det(T1 affine)=%.3f | det(PET affine)=%.3f\n', ...
        dcom, ncc, det(Vt.mat(1:3,1:3)), det(Vr.mat(1:3,1:3)));
end
