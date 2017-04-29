function p = predictOneVsAll(all_theta, X)
%  PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%  are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

%some important variables
m = size(X, 1); %training examples
num_labels = size(all_theta, 1);
n = size(X, 2); %features

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.


  % for multiplication
  theta = all_theta';
  
  %hypothesis of some sort
  h = X * theta;
  
  % hypothesis is a m x k matrix, max would give index of most probable k for each example
  [tmp p] = max(h, [], 2);


  % the following code can also make prediction, but as it doesn't account for MOST PROBABLE prediction, its accuracy is low.
%  for i = 1:m
%    for j = 1:num_labels
%      if( (all_theta(j, :) * new_X(:, i)) >= 0 )
%        % if the parameter of sigmoid is >= 0, sigmoid is >= 0.5, so we can just check the parameter rather than the result of sigmoid.
%        % if the parameter is non-negative, sigmoid will give a positive label. I'm just check the positive answer and noting its index here, instead of making matrices and then finding the index from 1-10
%        % NOT A GOOD SOLUTION but a TRY
%       p(i) = j;
%      endif
%    end
%  end

% =========================================================================


end
