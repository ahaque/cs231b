% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function tldSaveImages(tr,dirname)

mkdir(dirname);

i = 1;
while 1
    if isempty(tr.snapshot{i}),
        break;
    end
    imwrite(tr.snapshot{i}.cdata,[dirname '/' num2str(i,'%05d') '.png']);
    i = i+1;
end


