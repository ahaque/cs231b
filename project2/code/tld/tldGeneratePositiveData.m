% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function [pEx,bbP] = tldGeneratePositiveData(tld,overlap,im0,p_par)

pEx  = [];
bbP = [];
%% ------------------------- (BEGIN) ----------------------------
%% TODO:generate positive bounig boxes
%%
%% given:
%% -----
%% tld (use tld.grid(1:4, :) -- all bounding boxes over the entire image)
%% overlap -- a vector of length size(tld.grid,2). This gives the overlap
%%            of the bounding box from current image with all the grid boxes.
%% im0    -- the current image
%% p_par  -- all the parameters for sampling positive examples around the bounding
%%            box from current image

%% to update or compute:
%% --------------------
%% bbP (4 X P)  - positive bounding boxes sampled around the bounding box from image



%% ------------------------ (END) -----------------------------------------------

%% pEx - the features extracted from the sampled ``positive" bounding boxes (bbP) in image

if ~isempty(bbP)
  pEx = tldGetPattern(im0,bbP,tld.model.patchsize, 0, tld.model.pattern_size);
  if tld.model.fliplr
    pEx = [pEx tldGetPattern(im0,bbP,tld.model.patchsize,1, tld.model.pattern_size)];
  end
end
