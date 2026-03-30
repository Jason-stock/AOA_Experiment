function [mse] = MSE(Y_output, Y_target)
outSize = size(Y_target);
outNum = outSize(1);

div = Y_target - Y_output;
mse = ((div'*div)/outNum);
end