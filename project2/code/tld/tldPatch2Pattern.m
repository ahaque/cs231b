% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function [pattern, patch] = tldPatch2Pattern(patch,patchsize)
%% ------------------ (BEGIN) -------------------
%% TODO: extract a feature from the patch after optionally resizing
%%       it to patchsize. Store this feature in pattern. At the simplest
%%       level, this could be a simple mean adjusted version
%%       of the resized patch itself.
%%       Some other features to try might be binary features such
%%       as LBP, BRISK, FREAK.

%% given:
%%-------
%% patch (M X N X 3) -- image patch
%% patchsize (2 X 2) -- pathchsize to resize the image patch to before
%%                      feature extraction
%%
%% Definitely wrong = path is (M,N)
%% Probably wrong - patchsize should be (1 x 2)
%%
%% to update or compute:
%% --------------------
%% pattern (1 x tld.model.pattern_size) vector feature extracted from patch
%% Probably wrong = pattern should be a column vector

%% -----------------  (END) ---------------------
    if length(size(patch)) == 3
        fprintf('Extracting features: Working with RGB\n');
    else
        fprintf('Extracting features: Working with Grayscale\n');
    end
    
    patch = imresize(patch, patchsize);
    mean_patch = mean(mean(patch, 1),2);

    patch = patch - mean_patch;
    pattern = reshape(patch, [numel(patch), 1]);
    
    % DEEP LEARNING - better feature extractor
end
