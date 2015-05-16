%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS 231B NOTE: This code was written by Albert/Fahim 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Given a cell array of patches, extract features
% Input: patches: cell array of size n x 1 containing a 2D matrix for each patch
% Output: features: a matrix where the ith column is the features for patch i
function [ features ] = extractHOG( patches, patchsize )
    features = [];
    for i = 1:size(patches, 1)
        patch = fastResize(patches{i}, patchsize);
        
        if i > 1
            features(:,i) = extractHOGFeatures(patch);
        else
            feat = extractHOGFeatures(patch);
            features = zeros(size(feat,2), size(patches, 1));
            features(:,1) = feat;
        end
    end
end

