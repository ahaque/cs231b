% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.


function [BB Conf tld] = tldDetection(tld,I)
% scanns the image(I) with a sliding window, returns a list of bounding
% boxes and their confidences that match the object description

BB        = [];
Conf      = [];
dt        = struct('bb',[],'idx',[],'conf1',[],'isin',nan(3,1),'patch',[]);

img  = tld.img{I};

tld.tmp.conf = [];
idx_dt = [];

%% ------------------ (BEGIN) ---------------------
%% TODO: Run the detection model on the image patches
%% to identify potential object boxes.
%% 
%% given
%%------
%% tld.img{I} - current image
%% tld.bb(:,1:I-1) - all boxes till now
%% tld.grid(1:4, :) - The set of all bounding boxes to score from the image
%% tld.model.patchsize - Each of the patches corresponding to the grid boxes
%%                       could be resized to this cannonical size before
%%                       feature extraction.
%% tld.detection_model - The detection you have learned from tldLearning.m

%% output
%% ------
%% tld.tmp.conf - size(1, size(tld.grid,2)) a vector of scores for all patches in grid
%% idx_dt - the indices of selected boxes from tld.grid based on detection scores. 
%%          tld.grid(1:4, idx_dt) provides the seltected object boxes based on the detector.
%% HINT: bb_overlap in bbox/ might be a useful code to prune the grid boxes before
%% running your detector. This is just a speed-up and might hurt performance.


%% ------------------ (END) -----------------------



num_dt = length(idx_dt); % get the number detected bounding boxes so-far 
if num_dt == 0, tld.dt{I} = dt; return; end % if nothing detected, return


% initialize detection structure
dt.bb     = tld.grid(1:4,idx_dt); % bounding boxes
dt.idx    = idx_dt; %find(idx_dt); % indexes of detected bounding boxes within the scanning grid
dt.conf1  = nan(1,num_dt); % Relative Similarity (for final nearest neighbour classifier)
dt.isin   = nan(3,num_dt); % detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
dt.patch  = nan(prod(tld.model.patchsize),num_dt); % Corresopnding patches

for i = 1:num_dt % for every remaining detection
    
    ex   = tldGetPattern(img,dt.bb(:,i),tld.model.patchsize,0,tld.model.pattern_size); % measure patch
    [conf1, isin] = tldNN(ex,tld); % evaluate nearest neighbour classifier
    
    % fill detection structure
    dt.conf1(i)   = conf1;
    dt.isin(:,i)  = isin;
    dt.patch(:,i) = ex;
    
end

idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour

if numel(idx) > 10
  [~, sort_idx] = sort(dt.conf1, 'descend');
  idx = false(size(dt.conf1));
  idx(sort_idx(1:10)) = true;
end

if ~any(idx)
  if (I <=2)
    [~, idx] = max(dt.conf1);
  else
    fprintf('Max confidence: %f, Could not find any detection (skipping) \n', max(dt.conf1));
  end
end

%fprintf('IN detection ... \n'); keyboard;

% output
BB    = dt.bb(:,idx); % bounding boxes
Conf  = dt.conf1(:,idx); % conservative confidences
tld.dt{I} = dt; % save the whole detection structure

