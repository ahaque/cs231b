% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function pattern = tldGetPattern(img,bb,patchsize,flip, pattern_size)
% get patch under bounding box (bb), normalize it size, reshape to a column
% vector and normalize to zero mean and unit variance (ZMUV)

% initialize output variable
nBB = size(bb,2);

pattern = zeros(pattern_size, nBB);
if ~exist('flip','var')
    flip= 0;
end

% for every bounding box
for i = 1:nBB
    
    % sample patch
    patch = img_patch(img.input,bb(:,i));
    
    % flip if needed
    if flip
        patch = fliplr(patch);
    end
    
    % normalize size to 'patchsize' and nomalize intensities to ZMUV
    pattern(:,i) = tldPatch2Pattern(patch,patchsize);
end
