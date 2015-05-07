function [ resized_img ] = fastResize( img, patchsize )

    target_H = patchsize(1);
    target_W = patchsize(2);
    
    H = size(img, 1);
    W = size(img, 2);
    
    h_index = floor(1:H/target_H:H);
    w_index = floor(1:W/target_W:W);
    
    resized_img = zeros(patchsize);

    for i = 1:target_H
        for j = 1:target_W
            resized_img(i,j) = img(h_index(i), w_index(j));
        end
    end

end

