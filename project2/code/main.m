%% --- run model on a test video ----
% class_name - name of the video like Car4
% data_dir - path to the tiny_tracking_data directory
% num_frames_to_track - number of frames to track the object for (default = inf)

%% First test with only 20 frames on Car4 (you should get map=1.0 and avg_overlap >= 0.8)
%% At num_frames=250, you should expect avg_overlap >= 0.68 and map >= 0.78)
%% These values should improve with better features such as LBP.

function main(class_name ,num_frames_to_track, data_dir)

if nargin<3
  data_dir = '../tiny_tracking_data';
end

if nargin <2
  num_frames_to_track = inf;
end

output_video_directory = sprintf('_output/%s/', class_name);

input_video_directory = sprintf('%s/%s/img/', data_dir, class_name);
ground_truth_file = sprintf('%s/%s/groundtruth.txt', data_dir, class_name);

gt = dlmread(ground_truth_file);
num_frames=size(gt,1);

if num_frames_to_track < num_frames
  num_frames = num_frames_to_track;
end

fprintf('Tracking ... %d frames\n', num_frames);

run_TLD_on_video(input_video_directory, output_video_directory, ...
      ground_truth_file, num_frames);

detection_file = [output_video_directory '/tld.txt'];
[avg_overlap, success_auc, map] = tldEvaluateTracking(detection_file, ...
                                    ground_truth_file, num_frames);

fprintf('The evaluation values: average-overlap=%f, success auc=%f, map=%f\n', ...
         avg_overlap, success_auc, map);

fid=fopen(sprintf('%s/%s_res.txt', output_video_directory, class_name), 'w');
fprintf(fid, 'The evaluation values: average-overlap=%f, success auc=%f, map=%f\n', ...
         avg_overlap, success_auc, map);
fclose(fid);
