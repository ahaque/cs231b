% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function pattern = tldPatch2Pattern(patch,patchsize)

pattern = [];
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
%% to update or compute:
%% --------------------
%% pattern (1 x tld.model.pattern_size) vector feature extracted from patch

%% -----------------  (END) ---------------------

if ~isempty(patch) && isempty(pattern)
  pattern = zeros(2,1);
end
