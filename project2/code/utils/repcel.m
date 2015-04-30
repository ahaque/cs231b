% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.
% 

function out = repcel(in,M,N)
% Repeats cells MxN times.

out = in(repmat(1:size(in,1),M,1),repmat(1:size(in,2),N,1));
