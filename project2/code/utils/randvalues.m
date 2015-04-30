% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.
% 

function out = randvalues(in,k)
% Randomly selects 'k' values from vector 'in'.

out = [];

N = size(in,2);

if k == 0
  return;
end

if k > N
  k = N;
end

if k/N < 0.0001
 i1 = unique(ceil(N*rand(1,k)));
 out = in(:,i1);
 
else
 i2 = randperm(N);
 out = in(:,sort(i2(1:k)));
end
