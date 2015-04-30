% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.
% 

function m = median2(x)
% Median without nan

x(isnan(x)) = [];
m = median(x);
