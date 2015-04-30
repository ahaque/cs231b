% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.
% 

function ntuple = ntuples(varargin)
% Computes all possible ntupples.

x = varargin;

ntuple = x{1};
for i = 2:length(x)
    num_col  = size(ntuple,2);
    num_item = length(x{i});
    ntuple   = repcel(ntuple, 1, num_item);
    newline  = repmat(x{i},1,num_col);
    ntuple   = [ntuple;newline];
end
