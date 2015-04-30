% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.
% 

function s = n2s(x,N)
% Number to string.

s = num2str(x,['%0' num2str(N) 'd']);
