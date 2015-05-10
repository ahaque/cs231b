function  matcaffe_init(use_gpu, model_def_file, model_file)
% matcaffe_init(model_def_file, model_file, use_gpu)
% Initilize matcaffe wrapper

if nargin < 1
  use_gpu = 1;
end
if nargin < 2 || isempty(model_def_file)
  % By default use imagenet_deploy
  %model_def_file = '../../models/bvlc_reference_caffenet/deploy.prototxt';
  model_def_file = './caffe/models/VGG_ILSVRC_16_layers_deploy.prototxt';

end
if nargin < 3 || isempty(model_file)
  % By default use caffe reference model
  %model_file = '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
  model_file = './caffe/models/VGG_ILSVRC_16_layers.caffemodel';
end


if caffe('is_initialized') == 0
  if exist(model_file, 'file') == 0
    error('You need a network model file');
  end
  if ~exist(model_def_file,'file')
    error('You need the network prototxt definition');
  end
  % load network in TEST phase
  caffe('init', model_def_file, model_file, 'test')
end
fprintf('Done with init\n');

% set to use GPU or CPU
if use_gpu
  fprintf('Using GPU Mode\n');
  caffe('set_mode_gpu');
else
  fprintf('Using CPU Mode\n');
  caffe('set_mode_cpu');
end
fprintf('Done with set_mode\n');
