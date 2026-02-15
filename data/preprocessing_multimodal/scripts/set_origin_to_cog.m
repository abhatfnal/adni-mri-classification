function set_origin_to_cog(img)
    % Move image origin to its centre of mass (in mm).
    % Only adjusts the header affine. Useful to stabilize coreg.

    img = char(img); 

    if ~exist(img,'file')
        error('File not found: %s', img);
    end

    % If input is .nii.gz, decompress to .nii
    if endsWith(img, '.nii.gz')
        outdir = fileparts(img);
        gunzip(img, outdir);
        img = img(1:end-3); % ".nii.gz" -> ".nii"
    end

    V = spm_vol(img);
    if isempty(V)
        error('spm_vol failed to read: %s (check type/format)', img);
    end

    Y = spm_read_vols(V);
    msk = isfinite(Y) & (Y > 0);
    if ~any(msk(:)), return; end

    [X,Yg,Z] = ndgrid(1:V.dim(1), 1:V.dim(2), 1:V.dim(3));
    xv = X(msk); yv = Yg(msk); zv = Z(msk);
    cog_vox = [mean(xv) mean(yv) mean(zv) 1]';

    cog_mm = V.mat * cog_vox;
    M = V.mat;
    M(1:3,4) = M(1:3,4) - cog_mm(1:3);

    spm_get_space(V.fname, M);
end
