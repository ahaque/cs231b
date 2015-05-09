% Copyright 2011 Zdenek Kalal
% This file is part of TLD.

function tld = tldTrainNN(pEx,nEx,tld)
    nP = size(pEx,2); % get the number of positive example 
    nN = size(nEx,2); % get the number of negative examples
    
    % ------------------ (BEGIN) --------------------
    % TODO: Update tld.pex and tld.nex

    % These are the positive and negative examples respectively retained
    % from all images seen so far.
    % given:
    % -----
    %  (pEx, nEx) -- Your are provided a set of positives and negatives from current image
    % respectively.
    %
    % to update:
    % ---------
    %  tld.pex, tld.nex -- Choose a good scheme to update the positives and negatives.
    %                      Naively extending tld.pex and tld.nex with all of pEx and nEx will
    %                      blow up computation. Choose wisely!

    MAX_HISTORY_SIZE = 1000;
    num_pos_history = size(tld.pex, 2);
    num_neg_history = size(tld.nex, 2);
    
    %fprintf('Before\n');
    %[size(tld.pex, 2) size(tld.nex, 2) nP nN]
    
    % If pos or neg history is empty, set tld.pex to pEx (init phase)
    if isempty(tld.pex) && isempty(tld.nex)
        tld.pex = pEx;
        tld.nex = nEx;
    else
        % Select the first tld.p_par_update.always_keep from history
        num_pos_keep = min(tld.p_par_update.always_keep, num_pos_history);
        num_neg_keep = min(tld.p_par_update.always_keep, num_neg_history);
        
        pex_keep = tld.pex(:, 1:num_pos_keep);
        nex_keep = tld.nex(:, 1:num_neg_keep);
        
        % Shuffle tld.pex and tld.pex to introduce randomness in the history
        % Create the shuffled indices which are offsetted into the array by always_keep
        pos_shuf_idx = randperm(num_pos_history - num_pos_keep) + num_pos_keep;
        neg_shuf_idx = randperm(num_neg_history - num_neg_keep) + num_neg_keep;

        old_pex = tld.pex(:, pos_shuf_idx);
        old_nex = tld.nex(:, neg_shuf_idx);

        % Concatenate the new data first so we're more likely to keep it
        tld.pex = cat(2, pex_keep, pEx, old_pex);
        tld.nex = cat(2, nex_keep, nEx, old_nex);
                
        % Select the first MAX_HISTORY_SIZE elements (or everything if the
        % concatenated array is still less than MAX_HISTORY_SIZE
        pos_end_idx = min(size(tld.pex, 2), MAX_HISTORY_SIZE);
        neg_end_idx = min(size(tld.nex, 2), MAX_HISTORY_SIZE);

        tld.pex = tld.pex(:, 1:pos_end_idx);
        tld.nex = tld.nex(:, 1:neg_end_idx);
    end
    
    %fprintf('After\n');
    %[size(tld.pex, 2) size(tld.nex, 2)]
    
    % ------------------ (END) ----------------------

end