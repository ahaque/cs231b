%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS 231B NOTE: This code was written by Albert/Fahim 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Given a cell array of patches, extract features
% Input: patches: cell array of size n x 1 containing a 2D matrix for each patch
% Output: features: a matrix where the ith column is the features for patch i
function [ features ] = extractRaw( patches, patchsize )
    
    features = zeros(prod(patchsize), size(patches, 1));
    
    for i = 1:size(patches, 1)
        patch = fastResize(patches{i}, patchsize);
        patch = patch - mean(double(patch(:)));
        patch = patch / std(patch(:));
        features(:,i) = reshape(patch, [numel(patch), 1]);
    end
end

