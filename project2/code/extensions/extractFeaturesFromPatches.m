%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS 231B NOTE: This code was written by Albert/Fahim 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ features ] = extractFeaturesFromPatches( tld, patches )

    if strcmp('cnn', tld.detection_model_params.feature) == 1
        features = extractCNN(tld, patches);
    elseif strcmp('raw', tld.detection_model_params.feature) == 1
        features = extractRaw(patches, tld.model.patchsize);
    elseif strcmp('hog', tld.detection_model_params.feature) == 1
        features = extractHOG(patches, tld.model.patchsize);
    elseif strcmp('rawhog', tld.detection_model_params.feature) == 1
        a = extractHOG(patches, tld.model.patchsize);
        b = extractRaw(patches, tld.model.patchsize);
        features = [a; b];
    elseif strcmp('all', tld.detection_model_params.feature) == 1
        a = extractHOG(patches, tld.model.patchsize);
        b = extractRaw(patches, tld.model.patchsize);
        c = extractCNN(tld, patches);
        features = [a; b; c];
    end
    
end

