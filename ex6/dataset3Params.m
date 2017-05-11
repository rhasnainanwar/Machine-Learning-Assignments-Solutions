function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
values  = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
n = length(values);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        
%
min_error = Inf; % error is infinite
for i = 1:n
  % choosing a value of C
  regularizationConstant = values(i); % C for instance
  for j = 1:n
    % choosing each sigma for a C
    sigmaConstant = values(j); % sigman for instance
    % traning model
    model = svmTrain(X, y, regularizationConstant, @(x1, x2) gaussianKernel(x1, x2, sigmaConstant));
    % predicting from the training
    predictions = svmPredict(model, Xval);
    % a vector of errors for each model
    error = mean(double(predictions ~= yval));
    %finidng the minimum error model
    if error < min_error
      min_error = error;
      C = regularizationConstant;
      sigma = sigmaConstant;
    end
  end
end
% =========================================================================

end
