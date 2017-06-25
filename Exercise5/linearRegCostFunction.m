function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


%Cost func calc
predictions=X*theta;
difference=(predictions-y).^2;
diff2=(predictions-y);
term1=(1/(2*m))*(sum(difference));
thetsum=sum(theta.^2)-(theta(1).^2);
term2=(lambda/(2*m))*thetsum;
J=term1+term2;

%Grad Calc
%{
X(1)=1;
theta(1)=0;

%theta
grad=(1/m)*(((diff2*X))+(lambda*theta));
%}


theta(1,1)=zeros(1,1);


terrm= (predictions-y);
grad=(1/m)*(X' * terrm)+((lambda/m)*theta);











% =========================================================================

grad = grad(:);

end
