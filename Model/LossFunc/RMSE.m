function [rmse] = RMSE(Y_output, Y_target)
outSize = size(Y_target);
outNum = outSize(1);

div = Y_target - Y_output;
rmse = ((div'*div)/outNum)^0.5;
end