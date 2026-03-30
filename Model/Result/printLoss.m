function [] = printLoss(Y_predict_train, Y_train, Y_predict_test, Y_test)
addpath(fullfile(pwd, 'Model', 'LossFunc'));
fprintf("train data sse=%.6f, test data sse=%.6f\n", SSE(Y_predict_train, Y_train), SSE(Y_predict_test, Y_test));
fprintf("train data mse=%.6f, test data mse=%.6f\n", MSE(Y_predict_train, Y_train), MSE(Y_predict_test, Y_test));
fprintf("train data rmse=%.6f, test data rmse=%.6f\n", RMSE(Y_predict_train, Y_train), RMSE(Y_predict_test, Y_test));
fprintf("train data nmse=%.6f, test data nmse=%.6f\n", NMSE(Y_predict_train, Y_train), NMSE(Y_predict_test, Y_test));
rmpath(fullfile(pwd, 'Model', 'LossFunc'));
end

	

