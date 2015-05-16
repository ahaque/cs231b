%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS 231B NOTE: This code was written by Albert/Fahim
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ resized_img ] = fastResize( img, patchsize )

    target_H = patchsize(1);
    target_W = patchsize(2);
    
    H = size(img, 1);
    W = size(img, 2);
    
    % If the patch needs to be downscaled
    if H >= target_H && W >= target_W
        h_index = floor(1:H/target_H:H);
        w_index = floor(1:W/target_W:W);

        resized_img = zeros(patchsize);

        for i = 1:target_H
            for j = 1:target_W
                resized_img(i,j) = img(h_index(i), w_index(j));
            end
        end
    % If patch is too smale, upscale it
    else
    	resized_img = imresize(patch, patchsize);
    end
end

