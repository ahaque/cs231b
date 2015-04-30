% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function source = tldInitFirstFrame(tld,source,min_win)

% load the first frame into memory
source.im0  = img_get(source,source.idx(1));

% set the initial bounding box: 
% - from file
if exist([source.input '/init.txt'],'file')
    bb = dlmread([source.input '/init.txt']);
    source.bb = bb(:);
    
    % check
    if isempty(source.bb) || min(bb_size(source.bb)) < min_win
        exit('Error: bounding box is incorrectly defined or too small');
    end
% - by mouse    
else
    source.bb = bb_click(tld,source.im0.input);
    
    % check
    if isempty(source.bb) || min(bb_size(source.bb)) < min_win
       source = [];
    end
end


