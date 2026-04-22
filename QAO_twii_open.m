% ==========================================================
% 實驗一: TAIEX 單目標收盤價預測 (依據文獻 式62 與 式63)
% 模型架構: 算術優化演算法 (AOA) + 複數類神經模糊系統 (CFNN)
% ==========================================================

% 清理工作區
clear; clc;

% 加入副程式路徑 (直接寫死您電腦中的絕對路徑)
addpath(fullfile(pwd, 'Model'), fullfile(pwd, 'Model/Result'));

% ==========================================================
% 1. 讀取資料與前處理
% ==========================================================
data = readtable("twii_history.csv", 'Delimiter', ',');

% 確保資料筆數至少有 1000 筆
if height(data) < 1000
    error('資料筆數不足 1000 筆！請檢查資料集。');
end

% 實驗一僅針對「收盤價」進行預測，提取最近 1000 天
closePrice = data.Close(end-999:end);

% 保存原始的最大最小值，用於後續反正規化 (Denormalization)
minClose = min(closePrice);
maxClose = max(closePrice);

% 2. 數值正規化到 [0,1] 以加速收斂並防止特徵尺度影響
closePrice = (closePrice - minClose) ./ (maxClose - minClose);

% ==========================================================
% 3. 建構輸入特徵與預測目標 (時間窗大小為 5)
% ==========================================================
num_data = length(closePrice);
num_samples = num_data - 4; % 總數據對數量 |TD| (應為 996)

% 預先配置記憶體空間
X = complex(zeros(num_samples, 2)); % 2 個維度的複數特徵
Y = zeros(num_samples, 1);          % 1 維實數預測目標

for i = 1:num_samples
    % 依據式(63)構建輸入數據對 h：
    % 第一維度複數：實部為第 i 天，虛部為第 i+1 天
    X(i, 1) = complex(closePrice(i), closePrice(i+1));
    % 第二維度複數：實部為第 i+2 天，虛部為第 i+3 天
    X(i, 2) = complex(closePrice(i+2), closePrice(i+3));
    
    % 預測目標 y：為第 i+4 天的實數收盤價
    Y(i) = closePrice(i+4);
end

% ==========================================================
% 4. 分割資料集 (約 10 個月為 Train，2 個月為 Test)
% ==========================================================
split_ratio = 0.8; % 取 80% 作為訓練集 (約符合文獻設定)
split_point = floor(num_samples * split_ratio);

H_train = X(1:split_point, :);
Y_train = Y(1:split_point);     % 修正為單目標實數
H_test  = X(split_point+1:end, :);
Y_test  = Y(split_point+1:end); % 修正為單目標實數

% ==========================================================
% 5. 模型訓練 (使用 AOA + CFNN)
% ==========================================================
nRuns = 1;        % 獨立實驗執行次數 (正式發表建議設為 10~30 取平均)
tIter = 3;        % AOA 的最大迭代次數 (測試用，正式訓練需增加)
finalRMSEs = zeros(nRuns, 1);

% 儲存最佳實驗結果
bestRMSE = inf;             
bestLossCurve = [];         
bestPredTrain = [];         
bestPredTest = [];          
bestRunIdx = 1;

for r = 1:nRuns
    fprintf('=== 第 %d 次實驗 ===\n', r);
    
    % 訓練階段：若 optimizer 強制要求 Y 須為複數，可在此傳入 complex(Y_train, 0)
    [ifParm, cnsqParm, baseVarFuzzyN, lossAll] = optimizer(H_train, Y_train, tIter);
    finalRMSEs(r) = lossAll(end);
    
    % 推論階段 (非線性版本)
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
% 7. 測試集評估與誤差指標 (針對收盤價)
% ==========================================================
% 若模型輸出為複數，預測實數目標時通常取實部 (Real Part)
test_target_close = real(Y_test);
test_pred_close   = real(bestPredTest);

% 計算指標 (假定您有 CalculateMetrics 函式)
Results = CalculateMetrics(test_pred_close, test_target_close);

fprintf('\n========= 測試集評估指標 =========\n');
fprintf('SSE   : %.6f\n', Results.SSE);
fprintf('MSE   : %.6f\n', Results.MSE);
fprintf('RMSE  : %.6f\n', Results.RMSE);
fprintf('MAE   : %.6f\n', Results.MAE);
fprintf('MAPE  : %.6f %%\n', Results.MAPE);
fprintf('SMAPE : %.6f %%\n', Results.SMAPE);

% ==========================================================
% 8. 還原數值 (Denormalization) 與繪圖
% ==========================================================
Y_Target_All = [Y_train; Y_test];
Y_Predict_All = [bestPredTrain; bestPredTest];

% 真實值與預測值的反正規化
Real_Close_Target = real(Y_Target_All) * (maxClose - minClose) + minClose;
Real_Close_Predict = real(Y_Predict_All) * (maxClose - minClose) + minClose;

time_axis = 1:length(Real_Close_Target);

% 繪圖 1: 最佳學習曲線
figure('Name', 'Best Run Convergence Curve');
plot(1:tIter, bestLossCurve, 'ro-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
title(sprintf('AOA Convergence Curve (Best Train RMSE: %.6f)', bestRMSE));
xlabel('Iteration');
ylabel('RMSE');
grid on;

% 繪圖 2: TWII 收盤價預測與實際比較圖
figure('Name', 'TWII Close Price Prediction');
plot(time_axis, Real_Close_Target, 'b-', 'LineWidth', 1.2); hold on;
plot(time_axis, Real_Close_Predict, 'r--', 'LineWidth', 1.2);
xline(split_point, 'k--', 'Label', 'Train/Test Split', 'LabelOrientation', 'horizontal', 'LabelHorizontalAlignment', 'center'); 
legend('Actual Close Price', 'Predicted Close Price', 'Location', 'best');
title('TAIEX Stock Close Price Prediction (Best Run)');
xlabel('Samples (Days)');
ylabel('Price (TWD)');
grid on;
hold off;

% 移除路徑
rmpath(fullfile(pwd, 'Model'), fullfile(pwd, 'Model/Result'));