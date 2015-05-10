
list_im = {'0001.jpg','0134.jpg'};

patches = cell(600,1);

idx = 1;
for i = 1:length(list_im)
   img = imread(list_im{i}); 
   % Select random 25x25 patch
   for j = 1:300
       x = randi([1, size(img, 2)-61],1,1);
       y = randi([1, size(img, 1)-61],1,1);
       patches{idx} = img(y:y+59, x:x+59);
       idx = idx + 1;
   end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BEGIN EXTRACTCNN FUNCTION extractCNN(patches)
% patches = cell array of size 1 x n

% Load the VGG Image mean
fprintf('Initializing caffe...\n');
matcaffe_init();

fprintf('Starting CNN feature extraction...\n');
tic

num_images = size(patches, 1);

IMAGE_DIM = 224;
batch_size = 50;
FEATURE_DIM = 4096; % This can come from TLD.model

d = load('ilsvrc_2012_mean');
IMAGE_MEAN = uint8(imresize(d.image_mean, [IMAGE_DIM IMAGE_DIM]));

cnn_input = zeros(IMAGE_DIM,IMAGE_DIM,3, num_images,'single');

fprintf('Resizing patches to CNN input...\n');
for i = 1:num_images
    im = imresize(patches{i}, [IMAGE_DIM IMAGE_DIM], 'bilinear');
    % Transform GRAY to RGB
    im = cat(3,im,im,im);
    % permute from RGB to BGR (IMAGE_MEAN is already BGR)
    im = im(:,:,[3 2 1]) - IMAGE_MEAN;
    % Crop the center of the image
    cnn_input(:,:,:,i) = permute(im,[2 1 3]);
end

scores = zeros(FEATURE_DIM, num_images, 'single');
num_batches = ceil(num_images/batch_size);
initic = tic;
for bb = 1 : num_batches
    fprintf('Starting batch %i...\n',bb);
    % Create the batch
    batchtic = tic;
    range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
    input_data = zeros(IMAGE_DIM, IMAGE_DIM,3, batch_size,'single');
    input_data(:,:,:,1:length(range)) = cnn_input(:,:,:,range);
    
    % Run caffe
    output_data = caffe('forward', {input_data});
    % Add to output vector
    output_data = squeeze(output_data{1});
    scores(:,range) = output_data(:,mod(range-1,batch_size)+1);
end
toc(initic);
size(scores)


