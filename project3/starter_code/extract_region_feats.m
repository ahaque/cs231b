function extract_region_feats()

splits = {'train', 'test'};

for s = 1:numel(splits)
  split = splits{s};
  im_data = load(sprintf('%s_ims.mat', split));
  images = im_data.images;

  ssearch_data = load(sprintf('ssearch_%s.mat', split));
  ssearch_boxes = ssearch_data.ssearch_boxes;
  % Get rid of test bounding boxes, just so we're not tempted to use them
  if strcmp(split, 'test')
    [images.bboxes] = deal([]);
    [images.classes] = deal([]);
  end

  for i = 1:numel(images)
    % TODO:
    % Extract CNN features from each selective search region
    % Remember to also extract them around the ground truth box for training images
    % Be careful about how you incorporate context.
    % Save features for each image to file, to be used later.
  end
end
