% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function [nEx] = tldInitializeNegativeData(tld,overlap,img)

% Measure patterns on all bboxes that are far from initial bbox
idxN        = find(overlap<tld.n_par.overlap);

% Randomly select 'num_patches' bboxes and measure patches
idx = randvalues(1:length(idxN),tld.n_par.num_patches);
bb  = tld.grid(:,idxN(idx));

nEx = tldGetPattern(img,bb,tld.model.patchsize,0,tld.model.pattern_size);
