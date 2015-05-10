
function [ features ] = extractFeaturesFromPatches( tld, patches )

    if strcmp('cnn', tld.detection_model_params.feature) == 1
        features = extractCNN(patches);
    elseif strcmp('raw', tld.detection_model_params.feature) == 1
        features = extractRaw(patches, tld.model.patchsize);
    end
    
end

