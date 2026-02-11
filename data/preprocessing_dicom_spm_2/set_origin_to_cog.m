function set_origin_to_cog(img)
% Move image origin to its centre of mass (in mm).
% Only adjusts the header affine. Useful to stabilize coreg.

    if ~exist(img,'file'); return; end
    V = spm_vol(img);
    Y = spm_read_vols(V);
    msk = isfinite(Y) & (Y > 0);     % keep positive intensities

    if ~any(msk(:)), return; end

    % voxel-space coordinates
    [X,Yg,Z] = ndgrid(1:V.dim(1), 1:V.dim(2), 1:V.dim(3));
    xv = X(msk); yv = Yg(msk); zv = Z(msk);
    cog_vox = [mean(xv) mean(yv) mean(zv) 1]';

    % convert to mm; shift origin there
    cog_mm = V.mat * cog_vox;
    M = V.mat;
    M(1:3,4) = M(1:3,4) - cog_mm(1:3);

    spm_get_space(V.fname, M);
end
