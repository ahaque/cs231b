% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function [nEx1,nEx2] = tldSplitNegativeData(nEx)
% Splits negative data to training and validation set

N    = size(nEx,2);
idx  = randperm(N);
nEx  = nEx(:,idx);
nEx1 = nEx(:,1:N/2); 
nEx2 = nEx(:,N/2+1:end);
