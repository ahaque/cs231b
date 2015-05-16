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
      
    %fprintf('-----------------------------------------\n');
    %fprintf('Frame: %i\n', I);
    stageA_bboxes = tld.grid;
    idx_dt = 1:size(tld.grid, 2);
    %fprintf('Stage 0: %i\n', size(tld.grid, 2));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % STAGE A: BOUNDING BOX LOCATION FILTER
    % Bounding box should not abruptly jump across the image
    
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
    
    %[min1 max1; min2 max2; min3 max3; min4 max4]
    
    % Find the bboxes which meet each criteron
    pass1 = stageA_bboxes(1,:) > min1;
    pass2 = stageA_bboxes(1,:) < max1;
    pass3 = stageA_bboxes(2,:) > min2;
    pass4 = stageA_bboxes(2,:) < max2;
    pass5 = stageA_bboxes(3,:) > min3;
    pass6 = stageA_bboxes(3,:) < max3;
    pass7 = stageA_bboxes(4,:) > min4;
    pass8 = stageA_bboxes(4,:) < max4;
    
    % Find bboxes which meet all criteron
    passedA = pass1 & pass2 & pass3 & pass4 & pass5 & pass6 & pass7 & pass8;
    idx_dt_temp = find(passedA == 1);
    % In case this stage deletes all bboxes
    if ~isempty(idx_dt_temp)
        idx_dt = idx_dt_temp;
    end

    %fprintf('Stage A (Location): %i\n', length(idx_dt));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % STAGE B: BOUNDING BOX SIZE FILTER
    % Select bboxes that are of similar size to the one in previous frame
    bbox_hw = bb_size(tld.grid(:, idx_dt));

    % Get the size of the last bbox and find thresholds
    del = tld.detection_model_params.bbox_size_delta;
    previous_bbox_size = bb_size(tld.bb(:,I-1));
    min1 = previous_bbox_size(1) - del;
    max1 = previous_bbox_size(1) + del;
    min2 = previous_bbox_size(2) - del;
    max2 = previous_bbox_size(2) + del;
    
    pass1 = bbox_hw(1,:) > min1;
    pass2 = bbox_hw(1,:) < max1;
    pass3 = bbox_hw(2,:) > min2;
    pass4 = bbox_hw(2,:) < max2;
    
    passedB = pass1 & pass2 & pass3 & pass4;
    idx_dt_temp = idx_dt(passedB);
    if ~isempty(idx_dt_temp)
        idx_dt = idx_dt_temp;
    end
    
    %fprintf('Stage B (Size): %i\n', length(idx_dt));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % STAGE C: VARIANCE THRESHOLD
    stageC_bboxes = tld.grid(:, idx_dt);
    
    % Get the actual patches
    patch_variances = zeros(1, size(stageC_bboxes, 2));    
    patches = cell(size(stageC_bboxes, 2), 1);
    
    for i=1:size(stageC_bboxes, 2)
        patches{i} = img_patch(img.input, stageC_bboxes(1:4, i));
        this_patch = patches{i};
        patch_variances(i) = var(double(this_patch(:)));
    end
    
    % Filter based on variance
    patch_var_idx = patch_variances > (tld.var/2);
    idx_dt_temp = idx_dt(patch_var_idx);
    if ~isempty(idx_dt_temp)
        idx_dt = idx_dt_temp;
    end
    
    %fprintf('Stage C (Variance): %i\n', length(idx_dt)); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % STAGE D: CLASSIFIER
    % Create the feature vectors
    X = extractFeaturesFromPatches(tld, patches(patch_var_idx)); % Each column is a data point
    
    if size(X, 1) > 0
        y_hat = testLearner(tld, X');
        % Get the positive candidates
        idx_dt_temp = idx_dt(y_hat==1);

        % In case this stage deletes all bboxes
        if ~isempty(idx_dt_temp)
            idx_dt = idx_dt_temp;
        end
    end
    
    %fprintf('Stage D: %i\n', length(idx_dt));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % STAGE E: NEAREST NEIGHBOR (this was provided by starter code)
    
    num_dt = length(idx_dt); % get the number detected bounding boxes so-far 
    if num_dt == 0, tld.dt{I} = dt; return; end % if nothing detected, return
    
    % initialize detection structure
    dt.bb     = tld.grid(1:4,idx_dt); % bounding boxes
    dt.idx    = idx_dt; %find(idx_dt); % indexes of detected bounding boxes within the scanning grid
    dt.conf1  = nan(1,num_dt); % Relative Similarity (for final nearest neighbour classifier)
    dt.isin   = nan(3,num_dt); % detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
    dt.patch  = nan(size(X,1), num_dt); % Corresopnding patches

    if size(X, 1) > 0
        ex = X(:, y_hat==1); % measure patch

        [conf1, isin] = tldNN(ex,tld); % evaluate nearest neighbour classifier
        conf1(isnan(conf1)) = 0;

        % fill detection structure
        dt.conf1   = conf1;
        dt.isin  = isin;
        dt.patch = ex;
    end
    
    idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour

    
    %fprintf('Stage E: %i\n', sum(idx));
    
    if numel(idx) > 10
      [~, sort_idx] = sort(dt.conf1, 'descend');
      idx = false(size(dt.conf1));
      idx(sort_idx(1:10)) = true;
    end

    if ~any(idx)
      if (I <=2)
        [~, idx] = max(dt.conf1);
      else
        %fprintf('Max confidence: %f, Could not find any detection (skipping) \n', max(dt.conf1));
      end
    end
    
    %fprintf('IN detection ... \n'); keyboard;

    % output
    BB    = dt.bb(:,idx); % bounding boxes
    Conf  = dt.conf1(:,idx); % conservative confidences
    tld.dt{I} = dt; % save the whole detection structure

end
