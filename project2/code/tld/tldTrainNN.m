% Copyright 2011 Zdenek Kalal
%
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

    % Perform random forgetting
    
    MAX_NUM_EXAMPLES = 500;
    num_prev_pos_examples = size(tld.pex, 2);
    if (num_prev_pos_examples + size(pEx, 2)) > MAX_NUM_EXAMPLES
        num_to_keep = MAX_NUM_EXAMPLES - size(pEx, 2);
        num_to_keep = min(num_prev_pos_examples, max(num_to_keep, 0));
        
        % Choosing random indices to keep
        tld.pex = tld.pex(datasample(1:num_prev_pos_examples, num_to_keep, 'Replace', false))
    end
    
    num_prev_neg_examples = size(tld.nex, 2);
    if (num_prev_neg_examples + size(nEx, 2)) > MAX_NUM_EXAMPLES
        num_to_keep = MAX_NUM_EXAMPLES - size(nEx, 2);
        num_to_keep = min(num_prev_neg_examples, max(num_to_keep, 0));
        
        % Choosing random indices to keep
        tld.nex = tld.nex(datasample(1:num_prev_neg_examples, num_to_keep, 'Replace', false))
    end
    
    tld.pex = [tld.pex pEx];
    tld.nex = [tld.nex nEx];

    % ------------------ (END) ----------------------
end
