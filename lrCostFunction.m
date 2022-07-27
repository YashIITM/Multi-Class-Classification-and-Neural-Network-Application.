function [J,grad] = lrCostFunction(X,y,theta,lambda)
    %computes cost and gradient for logistic regression with
    %regularization
    %It computes the cost of using theta as the parameter for 
    %regularized logistic regression and the gradient of the cost
    %w.r.t to the parameters.
    
    m = length(y); 
    
    J = 0;
    grad = zeros(size(theta));
    
    %DIMENSIONS
    %       theta = n+1 x 1
    %       X     = m x n+1
    %       y     = m x 1
    %       grad  = n+1 x 1
    %       J     = Scalar
    
    z = X*theta;
    h_x = sigmoid(z);
    
    reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
    
    J = (1/m)*sum((-y.*log(h_x))-((1-y).*log(1-h_x))) + reg_term; % scalar
    
    grad(1) = (1/m) * (X(:,1)'*(h_x-y));                                    % 1 x 1
    grad(2:end) = (1/m) * (X(:,2:end)'*(h_x-y)) + (lambda/m)*theta(2:end);
    
    grad = grad(:);
end 