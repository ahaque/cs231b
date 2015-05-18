% --- run model on a test video ----
% class_name - name of the video like Car4
% data_dir - path to the tiny_tracking_data directory
% num_frames_to_track - number of frames to track the object for (default = inf)

% First test with only 20 frames on Car4 (you should get map=1.0 and avg_overlap >= 0.8)
% At num_frames=250, you should expect avg_overlap >= 0.68 and map >= 0.78)
% These values should improve with better features such as LBP.

function main(varargin)

options = struct(   'data_dir', '../data', ...
                    'num_frames_to_track', inf, ...
                    'feature', 'hog', ...
                    'classifier', 'svm');
                
required_params = {'video'; 'feature'; 'classifier'};
                
nArgs = length(varargin);
if round(nArgs/2) ~= nArgs/2
   error('Must specify name-value parameter pairs into main.')
end         

input_param_names = cell(nArgs/2, 1);

% Store all args
idx = 1;
for pair = reshape(varargin,2,[])
   paramName = lower(pair{1});
   input_param_names{idx} = paramName;
   switch(paramName)
       case 'video'
           options.video_name = pair{2};
       case 'feature'
           options.feature = lower(pair{2});
       case 'classifier'
           options.classifier = lower(pair{2});
       case 'data_dir'
           options.data_dir = pair{2};
       case 'num_frames_to_track'
           options.num_frames_to_track = pair{2};
   end
   idx = idx + 1;
end

% Make sure necessary args have been inputted
exit = 0;
for i = 1:length(required_params);
    idx1 = strfind(input_param_names, required_params{i});
    idx2 = find(not(cellfun('isempty', idx1)), 1);
    if isempty(idx2)
        fprintf('\tError: Missing required input parameter "%s"\n', required_params{i});
        exit = 1;
    end
end

if exit == 1
    fprintf('\tExample matlab commands:\n');
    fprintf('\t   main(''video'',''Dancer2'',''feature'',''raw'',''classifier'',''svm'')\n');
    fprintf('\t   main(''video'',''Dancer2'',''feature'',''raw'',''classifier'',''svm'',''num_frames_to_track'',20)\n');
    return
end

output_video_directory = sprintf('_output/%s/', options.video_name);

input_video_directory = sprintf('%s/%s/img/', options.data_dir, options.video_name);
ground_truth_file = sprintf('%s/%s/groundtruth.txt', options.data_dir, options.video_name);

gt = dlmread(ground_truth_file);
num_frames=size(gt,1);

if options.num_frames_to_track < num_frames
  num_frames = options.num_frames_to_track;
end

fprintf('Tracking ... %d frames\n', num_frames);
tic
run_TLD_on_video(options, input_video_directory, output_video_directory, ...
      ground_truth_file, num_frames);

detection_file = [output_video_directory '/tld.txt'];
[avg_overlap, success_auc, map] = tldEvaluateTracking(detection_file, ...
                                    ground_truth_file, num_frames);

fprintf('The evaluation values: average-overlap=%f, success auc=%f, map=%f\n', ...
         avg_overlap, success_auc, map);
a = toc;
fprintf('Completed in %f seconds at %f fps\n', a, num_frames/a);
fid=fopen(sprintf('%s/%s_res.txt', output_video_directory, options.video_name), 'w');
fprintf(fid, 'The evaluation values: average-overlap=%f, success auc=%f, map=%f\n', ...
         avg_overlap, success_auc, map);
fclose(fid);
