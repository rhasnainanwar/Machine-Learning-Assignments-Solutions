function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
m = size(X,1); % number of training examples
idx = zeros(m, 1); % centroid indices vectorize

for i = 1:m % iterating over all examples
   minError = Inf; %infinity
  for k = 1:K % iterating over all centroids
    error =  sum( (X(i, :) - centroids(k, :)).^2 );
    if minError > error
      idx(i) = k;
      minError = error; % setting new minimum error
    end %end if
  end
end
% =============================================================
end
