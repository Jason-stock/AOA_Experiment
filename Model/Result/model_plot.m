function [] = model_plot(Tr,Ts,Loss)
%learning curve繪製
Y_predict_train = Tr(:,1);
Y_train = Tr(:,2);
Y_predict_test = Ts(:,1);
Y_test = Ts(:,2);

figure;
result_plot_lrnCurve(length(Loss), Loss);
title('Learning Curve');

% 計算誤差
% error_train = Y_predict_train - Y_train;
% error_test = Y_predict_test - Y_test;

% === 實部 (開盤價) 繪圖 ===
% figure;
% result_plot_error(1:500, real(error_train), 1:498, real(error_test));
% title('Function approximation error (Open Price / Real Part)');

figure;
result_plot_graph(1:500, real(Y_train), real(Y_predict_train));
title("Graph of training (Open Price / Real Part)");

figure;
result_plot_graph(501:998, real(Y_test), real(Y_predict_test));
title("Graph of testing (Open Price / Real Part)");


% === 虛部 (收盤價) 繪圖 ===
% figure;
% result_plot_error(1:500, imag(error_train), 1:498, imag(error_test));
% title('Function approximation error (Close Price / Imaginary Part)');

figure;
result_plot_graph(1:500, imag(Y_train), imag(Y_predict_train));
title("Graph of training (Close Price / Imaginary Part)");

figure;
result_plot_graph(501:998, imag(Y_test), imag(Y_predict_test));
title("Graph of testing (Close Price / Imaginary Part)");

end
