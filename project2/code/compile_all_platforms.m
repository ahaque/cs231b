% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

% Compiles mex files
clc; clear all; cd mex;

if ispc
    disp('PC');
    include = ' -Ic:\OpenCV2.2\include\opencv\ -Ic:\OpenCV2.2\include\';

    libpath = 'c:\OpenCV2.2\lib\';
    files = dir([libpath '*.lib']);
    
    lib = [];
    for i = 1:length(files),
        lib = [lib ' ' libpath files(i).name];
    end
    
    eval(['mex lk.cpp -O' include lib]);
    mex -O -c tld.cpp
    mex -O linkagemex.cpp
    mex -O bb_overlap.cpp
    mex -O warp.cpp
    mex -O distance.cpp
end

if ismac
    disp('Mac');
    
    include = ' -I/opt/local/include/opencv/ -I/opt/local/include/'; 
    libpath = '/opt/local/lib/'; 
    
    files = dir([libpath 'libopencv*.dylib']);
    
    lib = [];
    for i = 1:length(files),
        lib = [lib ' ' libpath files(i).name];
    end
 
    eval(['mex lk.cpp -O' include lib]);
    mex -O -c tld.cpp
    mex -O linkagemex.cpp
    mex -O bb_overlap.cpp
    mex -O warp.cpp
    mex -O distance.cpp
    
end

if isunix
    disp('Unix');
    
    include = ' -I/usr/local/include/opencv/ -I/usr/local/include/';
    libpath = '/usr/local/lib/';
    
    files = dir([libpath 'libopencv*.so.2.4.6']);
    %files = [libpath 'libopencv_core.so ' libpath 'libopencv_highgui.so'];
    %lib = files;
    lib = [];
    for i = 1:length(files),
        lib = [lib ' ' libpath files(i).name];
    end
    
    lib = ' /usr/local/lib/libopencv_core.so.2.4.6 /usr/local/lib/libopencv_highgui.so.2.4.6 /usr/local/lib/libopencv_imgproc.so.2.4.6 /usr/local/lib/libopencv_video.so.2.4.6';

    keyboard;
    eval(['mex lk.cpp -O' include lib]);
    mex -O -c tld.cpp
    mex -O linkagemex.cpp
    mex -O bb_overlap.cpp
    mex -O warp.cpp
    mex -O distance.cpp
    
end


cd ..
disp('Compilation finished.');

