function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

	% We need to save the theta values in order to update them all at once later on.
	updatedThetas = zeros(size(X, 2), 1);

    % Caching the hypothesis in order to avoid calculating it every time
    hypothesis = X * theta;
	
	for i = 1:size(X, 2),
	    newTheta = theta(i) - ((alpha * sum((hypothesis - y) .* X(:, i))) / m);
		updatedThetas(i) = newTheta;
	end
	
	theta = updatedThetas;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
