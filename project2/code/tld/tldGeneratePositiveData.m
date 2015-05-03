% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function [pEx,bbP] = tldGeneratePositiveData(tld,overlap,im0,p_par)

pEx  = [];
bbP = [];
%% ------------------------- (BEGIN) ----------------------------
%% TODO:generate positive bounding boxes
%%
%% given:
%% -----
%% tld (use tld.grid(1:4, :) -- all bounding boxes over the entire image)
%% overlap -- a vector of length size(tld.grid,2). This gives the overlap
%%            of the bounding box from current image with all the grid boxes.
%% im0    -- the current image
%% p_par  -- all the parameters for sampling positive examples around the bounding
%%            box from current image

% Car4
% tld.grid(1:4,:) = 4x34490
% overlap = 1x34490
% im0.input = 240x360

% p_par:
%     num_closest: 1
%       num_warps: 20
%           noise: 5
%           angle: 20
%           shift: 0.0200
%           scale: 0.0200

%% to update or compute:
%% --------------------
%% bbP (4 X P)  - positive bounding boxes sampled around the bounding box from image
% Follow Section 5.6.1

% Select 10 bboxes closest to original bbox
if p_par.num_closest == 1
    % This is an order of magnitude faster than sorting
    [~, closest_neighbors_idx] = max(overlap);
else
    [~, sortIndex] = sort(overlap,'descend');
    closest_neighbors_idx = sortIndex(1:p_par.num_closest);
end

closest_bboxes = tld.grid(1:4,closest_neighbors_idx);
% For each bbox, generate warped versions using parameters in p_par
for i = 1:p_par.num_closest
    current_bbox = closest_bboxes(:,i);
    for w = 1:p_par.num_warps
        patch = img_patch(im0.input, current_bbox, rand, p_par);
        imshow(patch)
        pause
    end
end


%% ------------------------ (END) -----------------------------------------------

%% pEx - the features extracted from the sampled ``positive" bounding boxes (bbP) in image

if ~isempty(bbP)
  pEx = tldGetPattern(im0,bbP,tld.model.patchsize, 0, tld.model.pattern_size);
  if tld.model.fliplr
    pEx = [pEx tldGetPattern(im0,bbP,tld.model.patchsize,1, tld.model.pattern_size)];
  end
end
