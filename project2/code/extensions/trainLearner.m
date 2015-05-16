%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS 231B NOTE: This code was written by Albert/Fahim 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input: X: each row is a training example
%        y: each value is a label for the corresponding training example
% Output: Fitted model
function [ model ] = trainLearner( tld, X, y )
    if strcmp('svm', tld.detection_model_params.learning_model) == 1
        model = fitcsvm(X, y);
    elseif strcmp('adaboost', tld.detection_model_params.learning_model) == 1
        [~, model] = adaboost('train', X, y, 50);
    end
end

