% Copyright 2011 Zdenek Kalal
% This file is part of TLD.

function [pEx,bbP] = tldGeneratePositiveData(tld,overlap,im0,p_par)


    % ------------------------- (BEGIN) ----------------------------
    % TODO: generate positive bounding boxes
    %
    % given:
    % -----
    % tld (use tld.grid(1:4, :) -- all bounding boxes over the entire image)
    % overlap -- a vector of length size(tld.grid,2). This gives the overlap
    %            of the bounding box from current image with all the grid boxes.
    % im0    -- the current image
    % p_par  -- all the parameters for sampling positive examples around the bounding
    %            box from current image

    % to update or compute:
    % --------------------
    % bbP (4 X P)  - positive bounding boxes sampled around the bounding box from image

    % Select p_par.num_closest bboxes closest to original bbox
    if p_par.num_closest == 1
        % This is an order of magnitude faster than sorting
        [~, closest_neighbors_idx] = max(overlap);
    else
        % TODO: Optimize if possible, since n << N
        [~, sortIndex] = sort(overlap,'descend');
        closest_neighbors_idx = sortIndex(1:p_par.num_closest);
    end

    closest_bboxes = tld.grid(1:4,closest_neighbors_idx);

    % Augment positive examples
    num_pos_examples = size(closest_bboxes,2)*tld.p_par_init.num_warps;

    bbP = zeros(size(closest_bboxes,1), num_pos_examples);

    idx = 1;
    num_patches = p_par.num_closest * tld.p_par_init.num_warps;
    patches = cell(num_patches, 1);

    % For each candidate patch
    for i=1:p_par.num_closest
        % Perform warps
        for w=1:tld.p_par_init.num_warps
            % Include the original at least once
            if w == 1
                patches{idx} = img_patch(im0.input, closest_bboxes(:,i));
            % Warp the patch
            else
                rand_seed = randn(1) * idx;
                patches{idx} = img_patch(im0.input, closest_bboxes(:,i), rand_seed, tld.p_par_init);
            end
            bbP(:,idx) = closest_bboxes(:,i);
            % Index counter
            idx = idx + 1;
        end
    end
    
    % Compute the actual features
    pEx = extractFeaturesFromPatches(tld, patches);
    
    % ------------------------ (END) -----------------------------------------------

    % pEx - the features extracted from the sampled ``positive" bounding boxes (bbP) in image

end