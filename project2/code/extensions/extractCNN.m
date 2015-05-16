
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS 231B NOTE: This code was written by Albert/Fahim with parts borrowed
% from the Caffe Github repository
% https://github.com/BVLC/caffe
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ features ] = extractCNN( tld, patches )
    % patches = cell array of size 1 x n
    %           where each cell contains a PATCH_SIZE x PATCH_SIZE matrix

    batch_size = 180; % Set by GPU limitations. For every 10 more in batch size, VGG-16 takes 582MB more memory
    FEATURE_DIM = 4096;  % This can come from TLD.model. Should equal the VGG-fc7-relu output layer

    num_images = size(patches, 1);
    
    IMAGE_DIM = 224;
    d = load('ilsvrc_2012_mean');
    IMAGE_MEAN = uint8(imresize(d.image_mean, [IMAGE_DIM IMAGE_DIM]));

    cnn_input = zeros(IMAGE_DIM,IMAGE_DIM,3, num_images,'single');

    %fprintf('Resizing patches to CNN input...\n');
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

    for bb = 1 : num_batches
        fprintf('Starting batch %i...\n',bb);
        % Create the batch
        range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
        input_data = zeros(IMAGE_DIM, IMAGE_DIM,3, batch_size,'single');
        input_data(:,:,:,1:length(range)) = cnn_input(:,:,:,range);

        % Run forward pass
        output_data = caffe('forward', {input_data});
        
        % Add to output vector
        output_data = squeeze(output_data{1});
        scores(:,range) = output_data(:,mod(range-1,batch_size)+1);
    end
    
    % Bug with AdaBoost. Need to keep feature vector size somewhat small
    if strcmp('adaboost', tld.detection_model_params.learning_model) == 1
        features = scores(1:390,:);
    else
        features = scores;
    end
end

