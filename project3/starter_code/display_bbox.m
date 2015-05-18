function display_bbox(im, x1, y1, x2, y2)
% Displays a bounding box on an image.
imagesc(im);
axis image ij off;
rectangle('position', [x1, y1, (x2-x1+1), (y2-y1+1)], 'linewidth', 5, 'edgecolor', 'green');

