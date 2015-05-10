function [ y_hat ] = testLearner( tld, X )

    learning_method = tld.detection_model_params.learning_model;
    
    if strcmp('svm', learning_method) == 1
        y_hat = predict(tld.detection_model, X);
        
    elseif strcmp('adaboost', learning_method) == 1
        y_hat = adaboost('apply', X, tld.detection_model);
        
    end
end

