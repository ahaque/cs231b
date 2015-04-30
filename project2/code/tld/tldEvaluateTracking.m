%% @brief: compute the average overlap between tracked box and ground-truth box,
%%         as well as the area under curve of success plot. Additionally, compute
%%         the map@0.5 overlap.
%% @input:
%%    - detection_file: a text file containing the detections comma separated in following format
%%                      top_left_x,top_left_y,bottom_right_x,bottom_right_y,conf
%%                      Each line corresponds to the deteciton in that frame. For instance, first
%%                      line corresponds to the detection in the first frame, and so on.
%%                      If there is not detection in a certain frame, then that line should have
%%                      nan,nan,nan,nan,nan
%%    - ground_truth_file: a text file containing the true boudning box for the object in each frame.
%%                         same format as before but without confidence score

%%  @output:
%%    - average_overlap: a single number giving the average overlap across all frames
%%    - auc: area under curve of the success plot.
%%    - map: mean average precision for boxes which have an overlap of atleast 0.5 with ground-truth

function [average_overlap, auc, map] = tldEvaluateTracking(detection_file, ground_truth_file, num_test_frames)


% Read detected boxes
db_with_scores = dlmread(detection_file, ',');
num_frames = size(db_with_scores,1);
assert(num_frames >= num_test_frames);
if num_frames > num_test_frames
  db_with_scores = db_with_scores(1:num_test_frames, :);
  num_frames = num_test_frames;
end

db_boxes = db_with_scores(:,1:4);
db_scores = db_with_scores(:,end);

% Read ground-truth boxes
gt_boxes = dlmread(ground_truth_file, ',');
if (size(gt_boxes,1) > num_test_frames)
  gt_boxes = gt_boxes(1:num_test_frames, :);
end

%fprintf('%d:%d\n',num_frames, size(gt_boxes,1));
assert(num_frames==size(gt_boxes,1));

overlap = ones(num_frames, 1);
isvalid = ~isnan(sum(gt_boxes,2));
isvalid_det = ~isnan(sum(db_boxes,2));
for i = 1:num_frames
  if isvalid(i)
    if isvalid_det(i)
      overlap(i) = bb_overlap(db_boxes(i,:)', gt_boxes(i,:)');
    else
      overlap(i) = 0;
    end
  else
    if isvalid_det(i)
      overlap(i) = 0;
    end
  end
end

%% get the mean overlap
average_overlap = mean(overlap);


%% get the AUC for success plot
success_ratio = zeros(1,11);
ctr = 1;
for overlap_thresh = 0:0.1:1.0
  success_ratio(ctr) = sum(overlap>=overlap_thresh)/num_frames;
  ctr = ctr + 1;
end
auc = trapz(0:0.1:1.0, success_ratio);

%% get map
overlap = overlap(isvalid_det);
[~, sort_idx] = sort(db_scores(isvalid_det), 'descend');
map = 0;
num_corr = 0;
for i = 1:numel(sort_idx)
  if overlap(sort_idx(i)) >= 0.5
    num_corr = num_corr + 1;
    map = map + (num_corr)/i;
  end
end

map = map/sum(isvalid);
