function [nmse] = NMSE(Y_output, Y_target)
outSize = size(Y_target);
outNum = outSize(1);
mu = mean(Y_output);
sigma = ((Y_output-mu)'*(Y_output-mu)/outNum)^0.5;

div = Y_output - Y_target;
mse = (div'*div)/outNum;
nmse = mse/sigma^2;
end

