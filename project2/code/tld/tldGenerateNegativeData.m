% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function [nEx,bbN] = tldGenerateNegativeData(tld,I,bb,im0,n_par)

nEx  = [];
bbN = [];

%% ------------------------- (BEGIN) ----------------------------
%% TODO:generate negative bounig boxes
%%
%% given:
%% -----
%% tld (use tld.dt{I}.bb -- to get bounding boxes detected by dtector from current frame)
%%  I     -- index of current frame
%%  bb    -- bounding box of object from current frame 
%% im0    -- the current image
%% n_par  -- all the parameters for sampling negative examples around the bounding
%%            box from current image

%% to update or compute:
%% --------------------
%% bbN (4 X N)  - negative bounding boxes sample from image 
%% (NOTE: It might be faster to just sample bounding-boxes from tld.dt{end}.bb.
%%        This just uses hard-negatives.)


%% ------------------------ (END) -----------------------------------------------

%% nEx - the features extracted from the sampled ``negative" bounding boxes (bbN) in image

if ~isempty(bbN)
  nEx = tldGetPattern(im0,bbN,tld.model.patchsize, 0, tld.model.pattern_size);
end
