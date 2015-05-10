% Copyright 2011 Zdenek Kalal
% This file is part of TLD.

% Extract patterns for a set of bounding boxes
function patterns = tldGetPattern(tld, img, bb)
    
    nBB = size(bb,2);

    % Generate the patches array
    patches = cell(nBB, 1);
    for i = 1:nBB
        patches{i} = img_patch(img.input,bb(:,i));
    end
    
    % Call the appropriate feature extraction function
    patterns = extractFeaturesFromPatches(tld, patches);

end