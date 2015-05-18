clear all;

% TODO: Edit this to point to the folder your caffe mex file is in.
path_to_matcaffe = '/data/jkrause/cs231b/caffe-rc2/matlab/caffe';
addpath(path_to_matcaffe)


% Load up the image
im = imread('peppers.png');

% Get some random image regions (format of each row is [x1 y1 x2 y2])
% Note: If you want to change the number of regions you extract features from,
% then you need to change the first input_dim in cnn_deploy.prototxt.
regions = [
1 1 100 100;
100 50 400 250;
1 1 512 284;
200 200 230 220
100 100 300 200];
 

% Convert image from RGB to BGR and single, which caffe requires.
im = single(im(:,:,[3 2 1]));

% Get the image mean and crop it to the center
mean_data = load('ilsvrc_2012_mean.mat');
image_mean = mean_data.image_mean;
cnn_input_size = 227; % Input size to the cnn we trained.
off = floor((size(image_mean,1) - cnn_input_size)/2)+1;
image_mean = image_mean(off:off+cnn_input_size-1, off:off+cnn_input_size-1, :);

% Extract each region
ims = zeros(cnn_input_size, cnn_input_size, 3, size(regions, 1), 'single');
for i = 1:size(regions, 1)
  r = regions(i,:);
  reg = im(r(2):r(4), r(1):r(3), :);
  % Resize to input CNN size and subtract mean
  reg = imresize(reg, [cnn_input_size, cnn_input_size], 'bilinear', 'antialiasing', false);
  reg = reg - image_mean;

  % Swap dims 1 and 2 to work with caffe
  ims(:,:,:,i) = permute(reg, [2 1 3]);
end

% Initialize caffe with our network.
% -cnn_deploy.prototxt gives the structure of the network we're using for
%   extracting features and is how we specify we want fc6 features.
% -cnn512.caffemodel is the binary network containing all the learned weights.
% -'test' indicates that we're only going to be extracting features and not
%   training anything
init_key = caffe('init', 'cnn_deploy.prototxt', 'cnn512.caffemodel', 'test');
caffe('set_device', 0); % Specify which gpu we want to use. In this case, let's use the first gpu.
caffe('set_mode_gpu');
%caffe('set_mode_cpu'); % Use if you want to use a cpu for whatever reason

% Run the CNN
f = caffe('forward', {ims});
% Convert the features to (num. dims) x (num. regions)
feat = single(reshape(f{1}(:), [], size(ims, 4)));
