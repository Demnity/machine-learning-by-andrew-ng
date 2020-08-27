function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
steps = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
C = 0.01;
sigma = 0.01;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
min_error = 100;
fprintf('Selecting C and sigma:\nC\tsigma\tError validation\n');
for i = 1 : size(steps)
	C_temp = steps(i);
	for j = 1 : size(steps)
		sigma_temp = steps(j);
		model = svmTrain(X, y, C_temp, @(x1,x2) gaussianKernel(x1, x2, sigma_temp));
		predictions = svmPredict(model, Xval);
		error_validation = mean(double(predictions ~= yval));
	
		if(error_validation < min_error)
			C = C_temp;
			sigma = sigma_temp;
			min_error = error_validation;
		endif

	end
end

fprintf('%f\t%f\t%f\n', C, sigma, error_validation);



% =========================================================================

end
