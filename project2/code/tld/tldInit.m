% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function tld = tldInit(opt,tld)


%% Initialize Lucas-Kanade tracking
lk(0);

% Setup a handle for the figure
if ~isempty(tld);
    handle = tld.handle;
    tld = opt;
    tld.handle = handle;
else
    tld = opt;
end


% INITIALIZE DETECTOR =====================================================

% Scanning grid (Utility function to scan and get bounding-boxes)
[tld.grid tld.scales] = bb_scan(tld.source.bb,size(tld.source.im0.input),tld.model.min_win);

%keyboard;

% Features
tld.nGrid     = size(tld.grid,2);


% Temporal structures
tld.tmp.conf = zeros(1,tld.nGrid);
%tld.tmp.patt = zeros(tld.model.num_trees,tld.nGrid);

% RESULTS =================================================================

% Initialize Trajectory
tld.img     = cell(1,length(tld.source.idx));
tld.snapshot= cell(1,length(tld.source.idx));
tld.dt      = cell(1,length(tld.source.idx));
tld.bb      = nan(4,length(tld.source.idx));
tld.conf    = nan(1,length(tld.source.idx));
tld.valid   = nan(1,length(tld.source.idx));
tld.size    = nan(1,length(tld.source.idx));
tld.trackerfailure = nan(1,length(tld.source.idx));
tld.draw    = zeros(2,0);
tld.pts     = zeros(2,0);
% Fill first fields
tld.img{1}  = tld.source.im0;
tld.bb(:,1) = tld.source.bb;
tld.conf(1) = 1;
tld.valid(1)= 1;
tld.size(1) = 1;

tld.model.pattern_size = opt.pattern_size;

% TRAIN DETECTOR ==========================================================

% Initialize structures
tld.imgsize = size(tld.source.im0.input);
tld.pEx     = cell(1,length(tld.source.idx)); % training data for NN
tld.nEx     = cell(1,length(tld.source.idx));
overlap     = bb_overlap(tld.source.bb,tld.grid); % bottleneck

% Target (display only)
tld.target = img_patch(tld.img{1}.input,tld.bb(:,1));

% Generate Positive Examples
[pEx,bbP] = tldGeneratePositiveData(tld,overlap,tld.img{1},tld.p_par_init);


% Correct initial bbox
[~, mxo] = max(overlap);
tld.bb(:,1) = tld.grid(1:4, mxo);

% Variance threshold
if ~isempty(pEx)
  tld.var = var(pEx(:,1))/2;
end
% disp(['Variance : ' num2str(tld.var)]);

% Generate Negative Examples
[nEx] = tldInitializeNegativeData(tld,overlap,tld.img{1});

% Split Negative Data to Training set and Validation set
[nEx1,nEx2] = tldSplitNegativeData(nEx);

tld.pEx{1}  = pEx; % save positive patches for later
tld.nEx{1}  = nEx; % save negative patches for later

% Train using training set ------------------------------------------------
tld.detection_model = [];
%%% --------------------- (BEGIN) -----------------------------------------
%%% TODO: initialize the detector based on the positive features (pEx)
%%        and negative example features (nEx)
%%        from source image. Store the model in tld.detection_model.

%%% ------------------------- (END) ----------------------------------------



% Nearest Neightbour 
tld.pex = [];
tld.nex = [];

tld = tldTrainNN(pEx,nEx1,tld);
tld.model.num_init = size(tld.pex,2);

% Estimate thresholds on validation set  ----------------------------------

% Nearest neighbor
conf_nn = tldNN(nEx2,tld);
tld.model.thr_nn = max(tld.model.thr_nn,max(conf_nn));
tld.model.thr_nn_valid = max(tld.model.thr_nn_valid,tld.model.thr_nn);
