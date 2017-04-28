function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%

    h = sigmoid( X*theta ); % hypothesis
    theta_ = theta; % temporary
    theta_(1) = 0;   % because we don't add anything for j = 0  
    
    % regularized cost function
    J = ( 2*sum( -y.*log( h ) - (1 - y).*log(1 - h ) )  + lambda*sum( theta_.^2 ) )/(2*m);
  
    %gradient
    grad = ( X'*(h - y) )/m; % unregularized gradient
    grad = grad + lambda*theta_/m; % regularized gradient

% =============================================================

grad = grad(:);

end
