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

%CostFunction calc
subb=(theta(1,1)).^2;
temp=X*theta; 
prediction=sigmoid(temp);
term1= -y.*log(prediction);
term2=(1-y).*log(1-prediction);
alms=term1-term2;
T=(1/m)*sum(alms);
J=T+(lambda/(2*m))*sum((theta' * theta)-subb);

%gradientcalc
theta(1,1)=zeros(1,1);


terrm= (prediction-y);
grad=(1/m)*(X' * terrm)+((lambda/m)*theta);





% =============================================================

end
