% Copyright 2011 Zdenek Kalal
% This file is part of TLD.

function [conf1,isin] = tldNN(x,tld)
    % 'conf1' ... full model (Relative Similarity)
    % 'isnin' ... inside positive ball, id positive ball, inside negative ball

    isin = nan(3,size(x,2));

    if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
        conf1 = zeros(1,size(x,2));
        return;
    end

    if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
        conf1 = ones(1,size(x,2));
        return;
    end

    conf1 = nan(1,size(x,2));


    % ------------------------ (BEGIN) ----------------------------
    % TODO: set conf1, isin for all columns in x
    % given
    %------
    % x: each column in x corresponds to a patch feature. Let N be number of patches.
    % tld.pex: positive examples stored in tld so far
    % tld.nex: negative examples stored in tld so far
    % output
    %-------
    % conf1: a N dimension vector of confidence scores for each patch in x
    %        One good confidence measure is (1 - max NCC of patch with negs)/(2 - max NCC of patch with negs - max NCC of patch with pos)
    % isin: size(3, N)
    %        -- isin(1,i) is set to 1 (else set to nan) if patch i belongs to positives
    %        -- isin(2,i) index of the maximally correlated positive patch from pex
    %        -- inin(3,i) is set to 1 (else set to nan) if patch i belongs to negatives 
    % HINT: Make use of distance function in mex folder to make things faster

    for i=1:size(x, 2)
        pEx_dists = distance(x(:,i), tld.pex, 1);
        nEx_dists = distance(x(:,i), tld.nex, 1);

        [pEx_value, pEx_index] = max(pEx_dists);
        [nEx_value, ~] = max(nEx_dists);
        
        isin(2,i) = pEx_index;
        if pEx_value > tld.model.ncc_thesame
            % NN is a positive patch
            isin(1,i) = 1;
            isin(3,i) = nan;
        else %if nEx_value > tld.model.ncc_thesame
            % Else negative
            isin(1,i) = nan;
            isin(3,i) = 1;
        end
        conf1(i) = (1 - nEx_value)/(2 - nEx_value - pEx_value);
        %[pEx_value, nEx_value, conf1(i), tld.model.thr_nn]
    end

   % ----------------------- (END) -------------------------------

   
end