%       ONE OF MY WORST CODES EVER       %

function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

  h = sigmoid( X*theta ); %hypothesis
  
  %using thea_ as temporary theta
  %regularization of theta except theta(1)
  for i = 2:size(theta)
    theta_(i-1) = theta(i)^2;
  end  
  %cost function
  J = ( 2*sum( -y.*log( h ) - (1 - y).*log(1 - h ) )  + lambda*sum( theta_ ) )/(2*m);
  
  
  %calculating gradient like before
  grad = ( X'*(h - y) )/m;
  
  %regularization
  for i = 2:size(grad)
    grad(i) = grad(i) + lambda*theta(i)/m;
  end  
% =============================================================

end
