% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

% Compiles mex files (on corn.stanford.edu)
clc; clear all; cd mex;

include = ' -I/usr/include/opencv -I/usr/include/opencv2/ ';
libpath = '/usr/lib/x86_64-linux-gnu/'; % This is for corn, change accordingly for local machine
   
lib = [' ' libpath 'libopencv_core.so ' libpath 'libopencv_highgui.so ' libpath 'libopencv_imgproc.so ' libpath 'libopencv_video.so'];

%keyboard;
eval(['mex lk.cpp -O' include lib]);
mex -O -c tld.cpp
mex -O linkagemex.cpp
mex -O bb_overlap.cpp
mex -O warp.cpp
mex -O distance.cpp
    
cd ..
disp('Compilation finished.');

