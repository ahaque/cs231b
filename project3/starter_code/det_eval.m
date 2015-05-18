function [ap, prec, rec] = det_eval(gt_bboxes, pred_bboxes)
% det_eval
% Arguments:
% gt_bboxes: Cell array of ground truth bounding boxes for a single class.
%   Each element corresponds to a single image and is a matrix of dimensions
%   n x 5, where n is the number of bounding boxes. Each bounding box is
%   represented as [x1 y1 x2 y2 score], where 'score' is the detection score.
% pred_bboxes: Cell array of predicted bounding boxes in the same format as
%   gt_bboxes, without the score component.
%
% Returns:
%   ap: average precision
%   prec: The precision at each point on the PR curve
%   rec: The recall at each point on the PR curve.



minoverlap = 0.5;
assert(numel(gt_bboxes) == numel(pred_bboxes));
num_ims = numel(gt_bboxes);

num_gt_boxes = sum(cellfun(@(x)size(x, 1), gt_bboxes));
num_pred_boxes = sum(cellfun(@(x)size(x, 1), pred_bboxes));
[all_pred_bboxes, im_nums] = sort_bboxes(pred_bboxes);

tp=zeros(num_pred_boxes,1);
fp=zeros(num_pred_boxes,1);

detected_pred_bboxes=cell(num_ims,1);
detected=cell(num_ims,1);

for pred_ind=1:num_pred_boxes
  if mod(pred_ind, 4096) == 0
    fprintf('calculating detection box %d of %d\n', pred_ind, num_pred_boxes);
  end
  num_gt_im_boxes = size(gt_bboxes{im_nums(pred_ind)}, 1);
  if isempty(detected{im_nums(pred_ind)})
    detected{im_nums(pred_ind)}=zeros(num_gt_im_boxes,1);
  end
  bb=all_pred_bboxes(pred_ind,:);
  ovmax=-inf;
  for g=1:num_gt_im_boxes
    bbgt=gt_bboxes{im_nums(pred_ind)}(g,:); 
    bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
    iw=bi(3)-bi(1)+1;
    ih=bi(4)-bi(2)+1;
    if iw>0 & ih>0                
      % compute overlap as area of intersection / area of union
      ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
        (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
        iw*ih;
      ov=iw*ih/ua;
      if ov>ovmax
        ovmax=ov;
        jmax=g;
      end
    end
  end
  % assign detection as true positive/don't care/false positive
  im_detected=detected{im_nums(pred_ind)};
  if ovmax>=minoverlap
    if ~im_detected(jmax)
      tp(pred_ind)=1; %true positive
      detected{im_nums(pred_ind)}(jmax)=1;
      detected_pred_bboxes{im_nums(pred_ind)}=[detected_pred_bboxes{im_nums(pred_ind)};bb]; 
    else
      fp(pred_ind)=1;%false positive (multiple detections)
    end
  else
    fp(pred_ind)=1; %false positive 
  end
end


% compute precision/recall
npos=num_gt_boxes;
fp=cumsum(fp);
tp=cumsum(tp);

rec=tp/npos;
prec=tp./(fp+tp);
ap=VOCap(rec,prec);


end



function [bboxes, im_nums] = sort_bboxes(pred_bboxes)

num_pred_boxes = sum(cellfun(@(x)size(x, 1), pred_bboxes));

bboxes = vertcat(pred_bboxes{:});
im_nums = zeros(size(bboxes, 1), 1);

% Concatenate them
bbox_ind = 1;
for im_ind = 1:numel(pred_bboxes)
  im_nums(bbox_ind:bbox_ind+size(pred_bboxes{im_ind}, 1)-1) = im_ind;
  bbox_ind = bbox_ind + size(pred_bboxes{im_ind});
end

% Sort
[~, inds] = sort(bboxes(:, end), 'descend');
bboxes = bboxes(inds, :);
im_nums = im_nums(inds);
end




function ap = VOCap(rec,prec)

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
end
