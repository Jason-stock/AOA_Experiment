addpath(fullfile(pwd, 'Model'),fullfile(pwd, 'Model/Result'));
data = readtable("Tesla Stock Price History.csv");

openPrice = data.Open;     % 開盤價欄位
openPrice = openPrice(1:1000);

closePrice = data.Price;   % 收盤價欄位 (原程式碼變數名為 closePrice)
closePrice = closePrice(1:1000);

% 保存原始的最大最小值，用於後續還原
minOpen  = min(openPrice);
maxOpen  = max(openPrice);
minClose = min(closePrice);
maxClose = max(closePrice);

% 2. 將數列依照 min-max scaling 方式正規化到 [0,1]
openPrice  = (openPrice  - minOpen ) ./ (maxOpen  - minOpen );
closePrice = (closePrice - minClose) ./ (maxClose - minClose);

num_samples = 998;
X = zeros(num_samples, 4);
Y1 = zeros(num_samples, 1);
Y2 = zeros(num_samples, 1);

for i = 1:num_samples
    t = i + 1;
    X(i, :) = [openPrice(t-1), openPrice(t), closePrice(t-1), closePrice(t)];
    Y1(i) = openPrice(t+1);
    Y2(i) = closePrice(t+1);
end

H_train = X(1:500, :);
Y_train = Y1(1:500) + 1j*Y2(1:500);

H_test = X(501:end, :);
Y_test = Y1(501:end) + 1j*Y2(501:end);

%記錄跑幾次、最好、最差、平均值以及標準差
nRuns = 15;
finalRMSEs = zeros(nRuns,1);      % 每次最後 RMSE

% --- 用於儲存「最佳」實驗的結果變數 ---
bestRMSE = inf;             % 初始設為無限大
bestLossCurve = [];         % 最佳的收斂曲線
bestPredTrain = [];         % 最佳的訓練集預測
bestPredTest = [];          % 最佳的測試集預測

%使用optimizer找出模型最佳參數
tIter = 30;
for r = 1:nRuns
    fprintf('=== 第 %d 次實驗 ===\n', r);
    [ifParm, cnsqParm, baseVarFuzzyN, lossAll] = optimizer(H_train, Y_train, tIter);
    finalRMSEs(r) = lossAll(end);        % 記錄最後 RMSE

    %使用approxiamtor預測資料
    Y_predict_train = approximator(H_train, ifParm, cnsqParm, baseVarFuzzyN);
    Y_predict_test = approximator(H_test, ifParm, cnsqParm, baseVarFuzzyN);
    
    % --- 判斷是否為目前最佳結果 ---
    if finalRMSEs(r) < bestRMSE
        bestRMSE = finalRMSEs(r);
        bestLossCurve = lossAll;
        bestPredTrain = Y_predict_train;
        bestPredTest = Y_predict_test;
    end
end

% ====== 計算統計量 ======
bestVal = min(finalRMSEs);
worstVal= max(finalRMSEs);
meanVal = mean(finalRMSEs);
stdVal  = std(finalRMSEs);

% ====== 顯示統計結果 ======
fprintf('\n========= 統計結果 (共 %d 次) =========\n', nRuns);
fprintf('Best  RMSE: %.6f\n', bestVal);
fprintf('Worst RMSE: %.6f\n', worstVal);
fprintf('Mean  RMSE: %.6f\n', meanVal);
fprintf('Std   RMSE: %.6f\n', stdVal);

% ==========================================================
% ====== 以下為新增：還原數值與繪製最佳結果圖形 ======
% ==========================================================

% 1. 準備資料：將 Complex Y 分離回 Real (Open) 和 Imag (Close)
%    並串接 Train 與 Test 以呈現完整時間軸
Y_Target_All = [Y_train; Y_test];
Y_Predict_All = [bestPredTrain; bestPredTest];

% 2. 還原數值 (Denormalization)
%    公式: x_orig = x_norm * (max - min) + min

% 真實值還原
Real_Open_Target = real(Y_Target_All) * (maxOpen - minOpen) + minOpen;
Real_Close_Target = imag(Y_Target_All) * (maxClose - minClose) + minClose;

% 預測值還原
Real_Open_Predict = real(Y_Predict_All) * (maxOpen - minOpen) + minOpen;
Real_Close_Predict = imag(Y_Predict_All) * (maxClose - minClose) + minClose;

% 3. 建立時間軸索引
time_axis = 1:length(Real_Open_Target);
split_point = length(Y_train); % 分隔線位置

% --- 繪圖 1: 最佳學習曲線 (Convergence Curve) ---
figure('Name', 'Best Run Convergence Curve');
plot(1:tIter, bestLossCurve, 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
title(['Best Run Convergence (RMSE: ' num2str(bestRMSE) ')']);
xlabel('Iteration');
ylabel('RMSE');
grid on;

% --- 繪圖 2: 開盤價還原後比較圖 (Open Price) ---
figure('Name', 'Tesla Open Price Prediction (Denormalized)');
plot(time_axis, Real_Open_Target, 'b-', 'LineWidth', 1.2); hold on;
plot(time_axis, Real_Open_Predict, 'r--', 'LineWidth', 1.2);
xline(split_point, 'k--', 'Label', 'Train/Test Split'); % 畫出訓練/測試分隔線
legend('Actual Open Price', 'Predicted Open Price', 'Location', 'best');
title('Tesla Stock Open Price Prediction (Best Run)');
xlabel('Samples (Days)');
ylabel('Price (USD)');
grid on;
hold off;

% --- 繪圖 3: 收盤價還原後比較圖 (Close Price) ---
figure('Name', 'Tesla Close Price Prediction (Denormalized)');
plot(time_axis, Real_Close_Target, 'b-', 'LineWidth', 1.2); hold on;
plot(time_axis, Real_Close_Predict, 'm--', 'LineWidth', 1.2);
xline(split_point, 'k--', 'Label', 'Train/Test Split'); % 畫出訓練/測試分隔線
legend('Actual Close Price', 'Predicted Close Price', 'Location', 'best');
title('Tesla Stock Close Price Prediction (Best Run)');
xlabel('Samples (Days)');
ylabel('Price (USD)');
grid on;
hold off;

rmpath(fullfile(pwd, 'Model'),fullfile(pwd, 'Model/Result'));