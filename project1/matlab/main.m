
im_name = 'banana1.bmp';

NUM_COMPONENTS = 5;

% convert the pixel values to [0,1] for each R G B channel.
im_data = double(imread(im_name)) / 255;
% display the image
imagesc(im_data);

% a bounding box initialization
% disp('Draw a bounding box to specify the rough location of the foreground');
% set(gca,'Units','pixels');
% ginput(1);
% p1=get(gca,'CurrentPoint');fr=rbbox;p2=get(gca,'CurrentPoint');
% p=round([p1;p2]);
% xmin=min(p(:,1));xmax=max(p(:,1));
% ymin=min(p(:,2));ymax=max(p(:,2));
% [im_height, im_width, channel_num] = size(im_data);
% xmin = max(xmin, 1);
% xmax = min(im_width, xmax);
% ymin = max(ymin, 1);
% ymax = min(im_height, ymax);
%  
% bbox = [xmin ymin xmax ymax];
% line(bbox([1 3 3 1 1]),bbox([2 2 4 4 2]),'Color',[1 0 0],'LineWidth',1);
% if channel_num ~= 3
%     disp('This image does not have all the RGB channels, you do not need to work on it.');
%     return;
% end

bbox = [25 25 613 443];

alpha = zeros(im_height, im_width);

for h = 1 : im_height
    for w = 1 : im_width
        if (w > xmin) && (w < xmax) && (h > ymin) && (h < ymax)
            % this pixel belongs to the initial foreground
            alpha(h,w) = 1;
        end
    end
end

img_flat = reshape(im_data, [im_height*im_width, 3]);
alpha_flat = reshape(alpha, [im_height*im_width, 1]);

% INITIALIZE THE FOREGROUND & BACKGROUND GAUSSIAN MIXTURE MODEL (GMM)
disp('Initializing foreground GMM...');
f_mu = zeros(NUM_COMPONENTS, 3);
f_sigma = zeros(3,3,NUM_COMPONENTS);
idx = kmeans(img_flat(alpha_flat == 1), NUM_COMPONENTS);
% Update GMM parameters
weights = zeros(NUM_COMPONENTS, 1);
for i = 1:NUM_COMPONENTS
    f_mu(i,:) = mean(img_flat(idx == i,:));
    f_sigma(:,:,i) = cov(img_flat(idx == i,:));
    weights(i) = length(img_flat(idx == i));
end
weights = weights ./ sum(weights);
f_gmm = gmdistribution(f_mu, f_sigma, weights);

disp('Initializing background GMM...');
b_mu = zeros(NUM_COMPONENTS, 3);
b_sigma = zeros(3,3,NUM_COMPONENTS);
idx = kmeans(img_flat(alpha_flat == 0), NUM_COMPONENTS);
% Update GMM parameters
weights = zeros(NUM_COMPONENTS, 1);
for i = 1:NUM_COMPONENTS
    b_mu(i,:) = mean(img_flat(idx == i,:));
    b_sigma(:,:,i) = cov(img_flat(idx == i,:));
    weights(i) = length(img_flat(idx == i));
end
weights = weights ./ sum(weights);
b_gmm = gmdistribution(b_mu, b_sigma, weights);

for i = 1:10
   % 1. Assign GMM components to pixels for each n in foreground 
   
   
   % 2. Learn GMM parameters from data z
   
   
   % 3. Estimate segmentation. Use min cut to solve
   
   
   % 4. Repeat until convergence
end

% 
% while CONVERGENCE
%     
%     UPDATE THE GAUSSIAN MIXTURE MODELS
%     
%     MAX-FLOW/MIN-CUT ENERGY MINIMIZATION
%     
%     IF THE ENERGY DOES NOT CONVERGE AFTER A SUFFICIENTLY LONG TIME
%         
%         break;
%     
%     END
% end