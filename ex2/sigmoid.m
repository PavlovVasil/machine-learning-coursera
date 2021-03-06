function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

rowSize = size(z)(1);
colSize = size(z)(2);
for i = 1:rowSize
    for j = 1:colSize
        currentEl = z(i,j);
        g(i,j) = 1 / (1 + (e .^ -currentEl));
    end
end

% =============================================================

end
