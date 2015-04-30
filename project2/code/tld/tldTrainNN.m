% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function tld = tldTrainNN(pEx,nEx,tld)

nP = size(pEx,2); % get the number of positive example 
nN = size(nEx,2); % get the number of negative examples


%% ------------------ (BEGIN) --------------------
%% TODO: Update tld.pex and tld.nex

%% These are the positive and negative examples respectively retained
%% from all images seen so far.
%% given:
%% -----
%%  (pEx, nEx) -- Your are provided a set of positives and negatives from current image
%% respectively.
%%
%% to update:
%% ---------
%%  tld.pex, tld.nex -- Choose a good scheme to update the positives and negatives.
%%                      Naively extending tld.pex and tld.nex with all of pEx and nEx will
%%                      blow up computation. Choose wisely!

%% ------------------ (END) ----------------------
