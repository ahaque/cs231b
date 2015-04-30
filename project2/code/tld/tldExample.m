% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

function [bb,conf] = tldExample(opt, max_files)

global tld; % holds results and temporal variables

% INITIALIZATION ----------------------------------------------------------

%opt.source = tldInitSource(opt.source); % select data source, camera/directory
opt.source.files = img_dir(opt.source.input);

if nargin <= 1
  max_files = length(opt.source.files);
end

max_files = min(max_files, length(opt.source.files));
opt.source.files = opt.source.files(1:max_files);
opt.source.idx = 1:length(opt.source.files);

figure(2); %set(2,'KeyPressFcn', @handleKey); % open figure for display of results
finish = 0; %function handleKey(~,~), finish = 1; end % by pressing any key, the process will exit

while 1
    source = tldInitFirstFrame(tld,opt.source,opt.model.min_win); % get initial bounding box, return 'empty' if bounding box is too small
    if ~isempty(source), opt.source = source; break; end % check size
end

tld = tldInit(opt,[]); % train initial detector and initialize the 'tld' structure
tld = tldDisplay(0,tld); % initialize display

% RUN-TIME ----------------------------------------------------------------

for i = 2:length(tld.source.idx) % for every frame
    %keyboard;
   
    tld = tldProcessFrame(tld,i); % process frame i
    tldDisplay(1,tld,i); % display results on frame i
    if finish % finish if any key was pressed
        if tld.source.camera
            stoppreview(tld.source.vid);
            closepreview(tld.source.vid);
             close(1);
        end
        close(2);
        bb = tld.bb; conf = tld.conf; % return results
        return;
    end
    
    if tld.plot.save == 1
        img = getframe;
        imwrite(img.cdata,[tld.output num2str(i,'%05d') '.png']);
    end
        
    
end

bb = tld.bb; conf = tld.conf; % return results

end
