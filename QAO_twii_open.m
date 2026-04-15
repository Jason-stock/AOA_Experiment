% 清理工作區
clear; clc;
% 直接寫死您電腦中的絕對路徑，保證一定抓得到
addpath(genpath('C:\Users\jason\LAB_code\Model'));
% ==========================================================
% 1. 讀取資料與前處理
% ==========================================================
data = readtable("twii_history.csv", 'Delimiter', ',');

% 確保資料筆數至少有 1000 筆
if height(data) < 1000
    error('資料筆數不足 1000 筆！');
end

% 從最近的日期往前數 1000 天
openPrice = data.Open(end-999:end);
closePrice = data.Close(end-999:end);

% 保存原始的最大最小值，用於後續數值還原
minOpen  = min(openPrice);
maxOpen  = max(openPrice);
minClose = min(closePrice);
maxClose = max(closePrice);

% 2. 數值正規化到 [0,1]
openPrice  = (openPrice  - minOpen ) ./ (maxOpen  - minOpen );
closePrice = (closePrice - minClose) ./ (maxClose - minClose);

% 3. 構建輸入與輸出特徵
% 輸入 X: [昨天開盤, 今天開盤, 昨天收盤, 今天收盤]
% 輸出 Y: [明天開盤, 明天收盤] (實部為開盤、虛部為收盤)
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

% 4. 分割資料集 (前 500 筆為 Train，後 498 筆為 Test)
H_train = X(1:500, :);
Y_train = Y1(1:500) + 1j*Y2(1:500);

H_test = X(501:end, :);
Y_test = Y1(501:end) + 1j*Y2(501:end);

% ==========================================================
% 5. 模型訓練 (使用 AOA + CFNN)
% ==========================================================
nRuns = 1;        % 實驗執行次數
tIter = 1;        % AOA 的迭代次數
finalRMSEs = zeros(nRuns,1);

% 用於儲存「最佳」實驗結果的變數
bestRMSE = inf;             
bestLossCurve = [];         
bestPredTrain = [];         
bestPredTest = [];          
bestRunIdx = 1;

for r = 1:nRuns
    fprintf('=== 第 %d 次實驗 ===\n', r);
    [ifParm, cnsqParm, baseVarFuzzyN, lossAll] = optimizer(H_train, Y_train, tIter);
    finalRMSEs(r) = lossAll(end);

    % 注意：這裡預設您已採用我們上次修改的非線性版本推論函式
    Y_predict_train = approximator_nonlinear(H_train, ifParm, cnsqParm, baseVarFuzzyN);
    Y_predict_test  = approximator_nonlinear(H_test, ifParm, cnsqParm, baseVarFuzzyN);
    
    % 判斷是否為最佳解並覆寫記錄
    if finalRMSEs(r) < bestRMSE
        bestRMSE = finalRMSEs(r);
        bestLossCurve = lossAll;
        bestPredTrain = Y_predict_train;
        bestPredTest = Y_predict_test;
        bestRunIdx = r;
    end
end

% ==========================================================
% 6. 計算與印出統計結果
% ==========================================================
fprintf('\n========= 統計結果 (共 %d 次) =========\n', nRuns);
fprintf('Best  RMSE: %.6f (出現在第 %d 次)\n', bestRMSE, bestRunIdx);
fprintf('Worst RMSE: %.6f\n', max(finalRMSEs));
fprintf('Mean  RMSE: %.6f\n', mean(finalRMSEs));
fprintf('Std   RMSE: %.6f\n', std(finalRMSEs));

% ==========================================================
% 7. 計算最佳解在「測試集」開盤價的各項指標 (RMSE, MSE, NMSE, SSE)
% ==========================================================
% 由於預測目標是開盤價，我們只提取預測結果的「實部 (Real Part)」進行評估
test_target_open = real(Y_test);
test_pred_open   = real(bestPredTest);

% 呼叫 LossFunc 資料夾內的函式
best_test_RMSE = RMSE(test_pred_open, test_target_open);
best_test_MSE  = MSE(test_pred_open, test_target_open);
best_test_NMSE = NMSE(test_pred_open, test_target_open);
best_test_SSE  = SSE(test_pred_open, test_target_open);

fprintf('\n========= 最佳解 (Best Run) 測試集 [開盤價] 評估指標 =========\n');
fprintf('Test RMSE: %.6f\n', best_test_RMSE);
fprintf('Test MSE : %.6f\n', best_test_MSE);
fprintf('Test NMSE: %.6f\n', best_test_NMSE);
fprintf('Test SSE : %.6f\n', best_test_SSE);

% ==========================================================
% 8. 還原數值與繪圖 (僅繪製開盤價)
% ==========================================================
Y_Target_All = [Y_train; Y_test];
Y_Predict_All = [bestPredTrain; bestPredTest];

% 真實值與預測值的反正規化 (Denormalization)
Real_Open_Target = real(Y_Target_All) * (maxOpen - minOpen) + minOpen;
Real_Open_Predict = real(Y_Predict_All) * (maxOpen - minOpen) + minOpen;

% 建立時間軸
time_axis = 1:length(Real_Open_Target);
split_point = length(Y_train); % 訓練/測試分隔線位置

% 繪圖 1: 最佳學習曲線
figure('Name', 'Best Run Convergence Curve');
plot(1:tIter, bestLossCurve, 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
title(sprintf('Best Run Convergence (Train RMSE: %.6f)', bestRMSE));
xlabel('Iteration');
ylabel('RMSE');
grid on;

% 繪圖 2: TWII 開盤價預測與實際比較圖
figure('Name', 'TWII Open Price Prediction');
plot(time_axis, Real_Open_Target, 'b-', 'LineWidth', 1.2); hold on;
plot(time_axis, Real_Open_Predict, 'r--', 'LineWidth', 1.2);
xline(split_point, 'k--', 'Label', 'Train/Test Split', 'LabelOrientation', 'horizontal', 'LabelHorizontalAlignment', 'center'); 
legend('Actual Open Price', 'Predicted Open Price', 'Location', 'best');
title('TWII Stock Open Price Prediction (Best Run)');
xlabel('Samples (Days)');
ylabel('Price (TWD)');
grid on;
hold off;

% 修改後 (最後一行)
rmpath(genpath('C:\Users\jason\LAB_code\Model'));