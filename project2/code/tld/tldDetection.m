% Copyright 2011 Zdenek Kalal
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
   
    % ------------------ (BEGIN) ---------------------
    % TODO: Run the detection model on the image patches
    % to identify potential object boxes.
    % 
    % given
    %------
    % tld.img{I} - current image
    % tld.bb(:,1:I-1) - all boxes till now
    % tld.grid(1:4, :) - The set of all bounding boxes to score from the image
    % tld.model.patchsize - Each of the patches corresponding to the grid boxes
    %                       could be resized to this cannonical size before
    %                       feature extraction.
    % tld.detection_model - The detection you have learned from tldLearning.m

    % output
    % ------
    % tld.tmp.conf - size(1, size(tld.grid,2)) a vector of scores for all patches in grid
    % idx_dt - the indices of selected boxes from tld.grid based on detection scores. 
    %          tld.grid(1:4, idx_dt) provides the seltected object boxes based on the detector.
    % HINT: bb_overlap in bbox/ might be a useful code to prune the grid boxes before
    % running your detector. This is just a speed-up and might hurt performance.
  
    fprintf('-----------------------------------------\n');
    fprintf('Frame: %i\n', I);
    stage0_bboxes = tld.grid;
    fprintf('Stage 0: %i\n', size(tld.grid, 2));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % STAGE 1: VARIANCE THRESHOLD
    
    % Get the actual patches
    patch_variances = zeros(1, size(stage0_bboxes, 2));    
    patches = cell(size(stage0_bboxes, 2), 1);
    for i=1:size(stage0_bboxes, 2)
        patches{i} = img_patch(img.input, stage0_bboxes(1:4, i));
        this_patch = patches{i};
        patch_variances(i) = var(double(this_patch(:)));
    end
    
    % Filter based on variance
    idx_dt = find(patch_variances > (tld.var/2));
    stage1_bboxes = tld.grid(1:4, idx_dt);
    
    fprintf('Stage 1: %i\n', length(idx_dt)); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % STAGE 2B: BOUNDING BOX SIZE FILTER
    % Select bboxes that are of similar size to the one in previous frame
    s2_bbox_hw = bb_size(stage1_bboxes);

    % Get the size of the last bbox and find thresholds
    previous_bbox_size = bb_size(tld.bb(:,I-1));
    min1 = previous_bbox_size(1) - tld.detection_model_params.bbox_size_delta;
    max1 = previous_bbox_size(1) + tld.detection_model_params.bbox_size_delta;
    min2 = previous_bbox_size(2) - tld.detection_model_params.bbox_size_delta;
    max2 = previous_bbox_size(2) + tld.detection_model_params.bbox_size_delta;
    
    % Filter candidate boxes that are too large or too small
    for i = 1:size(s2_bbox_hw, 2)
        if s2_bbox_hw(1,i) < min1 || s2_bbox_hw(1,i) > max1 || s2_bbox_hw(2,i) < min2 || s2_bbox_hw(2,i) > max2
            idx_dt(i) = 0;
        end
    end
    
    % Remove the zero-value elements
    idx_dt(idx_dt == 0) = [];
    
    fprintf('Stage 2B: %i\n', length(idx_dt));
    
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % STAGE 2C: BOUNDING BOX LOCATION FILTER
    % Bounding box should not abruptly jump across the image
    idx_dt_2c = idx_dt;
    stage2b_bboxes = tld.grid(:, idx_dt_2c);
    
    % Get the size of the last bbox and find thresholds
    previous_bbox_loc = tld.bb(:,I-1);
    min1 = previous_bbox_loc(1) - tld.detection_model_params.bbox_loc_delta;
    max1 = previous_bbox_loc(1) + tld.detection_model_params.bbox_loc_delta;
    min2 = previous_bbox_loc(2) - tld.detection_model_params.bbox_loc_delta;
    max2 = previous_bbox_loc(2) + tld.detection_model_params.bbox_loc_delta;
    min3 = previous_bbox_loc(3) - tld.detection_model_params.bbox_loc_delta;
    max3 = previous_bbox_loc(3) + tld.detection_model_params.bbox_loc_delta;
    min4 = previous_bbox_loc(4) - tld.detection_model_params.bbox_loc_delta;
    max4 = previous_bbox_loc(4) + tld.detection_model_params.bbox_loc_delta;
    
    % Remove candidate boxes that jump too much
    for i = 1:size(stage2b_bboxes, 2)
        % Check the first coordinate
        if stage2b_bboxes(1,i) < min1 || stage2b_bboxes(1,i) > max1 || stage2b_bboxes(2,i) < min2 || stage2b_bboxes(2,i) > max2
            idx_dt_2c(i) = 0;
        % Check 2nd coordinate
        elseif stage2b_bboxes(3,i) < min3 || stage2b_bboxes(3,i) > max3 || stage2b_bboxes(4,i) < min4 || stage2b_bboxes(4,i) > max4
            idx_dt_2c(i) = 0;
        end
    end
    
    % Remove the zero-value elements
    idx_dt_2c(idx_dt_2c == 0) = [];
    
    % Error correction
    % Sometimes there is abrupt motion. If there is, stage 2C will kill all
    % candidate bboxes. If that's the case, reset to stage 2B
    if nnz(idx_dt_2c) ~= 0
        idx_dt = idx_dt_2c;
    end
   
    fprintf('Stage 2C: %i\n', length(idx_dt));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % STAGE 2A: SVM CLASSIFIER
    % Create the feature vectors
    X = extractFeaturesFromPatches(tld, patches(idx_dt)); % Each column is a data point
    % Run the SVM
    y_hat = testLearner(tld, X');
    % Get the positive candidates
    idx_dt = idx_dt(y_hat==1);
    stage2_bboxes = tld.grid(:, idx_dt);
    
    fprintf('Stage 2A: %i\n', length(idx_dt));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % STAGE 3: NEAREST NEIGHBOR (this was provided by starter code)
    
    num_dt = length(idx_dt); % get the number detected bounding boxes so-far 
    if num_dt == 0, tld.dt{I} = dt; return; end % if nothing detected, return
    
    % initialize detection structure
    dt.bb     = tld.grid(1:4,idx_dt); % bounding boxes
    dt.idx    = idx_dt; %find(idx_dt); % indexes of detected bounding boxes within the scanning grid
    dt.conf1  = nan(1,num_dt); % Relative Similarity (for final nearest neighbour classifier)
    dt.isin   = nan(3,num_dt); % detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
    dt.patch  = nan(size(X,1), num_dt); % Corresopnding patches

    
    ex = X(:, y_hat==1); % measure patch

    [conf1, isin] = tldNN(ex,tld); % evaluate nearest neighbour classifier
    conf1(isnan(conf1)) = 0;
    
    % fill detection structure
    dt.conf1   = conf1;
    dt.isin  = isin;
    dt.patch = ex;
    
    idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
    
    fprintf('Stage 3: %i\n', sum(idx));
    
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
    toc
    %fprintf('IN detection ... \n'); keyboard;

    % output
    BB    = dt.bb(:,idx); % bounding boxes
    Conf  = dt.conf1(:,idx); % conservative confidences
    tld.dt{I} = dt; % save the whole detection structure

end
