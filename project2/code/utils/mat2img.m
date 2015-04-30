% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.
% 

function img = mat2img(data,no_row)
% 'Data' contains square images stored in columns.

if ~exist('RATIO','var')
    RATIO = 1;
end


if isempty(data)
    img = []; return;
end

[M,N] = size(data);

sM = sqrt(M);
if ceil(sM) ~= sM
    img = []; disp('Wrong input!'); return;
end

W     = sM;
H     = sM;

if no_row > N, no_row = N; end
no_col = ceil(N/no_row);
%no_row = ceil(N/no_col);
img    = zeros(no_row*H,no_col*W);

for i = 1:N
   
    [I, J] = ind2sub([no_row, no_col], i);

    row = 1+(I-1)*H:(I-1)*H+H;
    col = 1+(J-1)*W:(J-1)*W+W;

    img0 = reshape(data(:,i),sM,sM);
    img0 = (img0 - min(img0(:))) / (max(img0(:)) - min(img0(:)));
    img(row, col) = img0;
    
end



