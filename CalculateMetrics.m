function metrics = CalculateMetrics(Y_predict, Y_target)
% CALCULATEMETRICS 整合計算六種常見的迴歸預測誤差指標
% 
% 輸入:
%   Y_predict - 模型的預測值向量
%   Y_target  - 真實的目標值向量
%
% 輸出:
%   metrics   - 包含 SSE, MSE, RMSE, MAE, MAPE, SMAPE 的結構體 (Struct)

    % 確保輸入為行向量 (Column vectors)，避免矩陣運算維度錯誤
    Y_predict = Y_predict(:);
    Y_target = Y_target(:);
    
    % 檢查資料長度是否一致
    if length(Y_predict) ~= length(Y_target)
        error('預測值與真實值的資料長度必須一致！');
    end

    n = length(Y_target);
    error_val = Y_target - Y_predict;
    
    % 加入一個極小值 epsilon，用以防止 MAPE 與 SMAPE 計算時發生「分母為零」的錯誤
    epsilon = 1e-8; 

    % ==========================================
    % 1. SSE (Sum of Squared Errors) 誤差平方和
    % 反映總體誤差累積量，數值受資料筆數影響大。
    % ==========================================
    metrics.SSE = sum(error_val.^2);
    
    % ==========================================
    % 2. MSE (Mean Squared Error) 均方誤差
    % 將 SSE 平均化，對極端值（Outliers）非常敏感。
    % ==========================================
    metrics.MSE = metrics.SSE / n;
    
    % ==========================================
    % 3. RMSE (Root Mean Squared Error) 均方根誤差
    % 量綱與原始數據相同，是最常被當作演算法 Fitness 的指標。
    % ==========================================
    metrics.RMSE = sqrt(metrics.MSE);
    
    % ==========================================
    % 4. MAE (Mean Absolute Error) 平均絕對誤差
    % 反映實際預測誤差的絕對大小，對極端值容忍度較高。
    % ==========================================
    metrics.MAE = mean(abs(error_val));
    
    % ==========================================
    % 5. MAPE (Mean Absolute Percentage Error) 平均絕對百分比誤差
    % 以百分比呈現，直觀易懂，但當真實值趨近於零時會產生極大偏差。
    % ==========================================
    metrics.MAPE = mean(abs(error_val ./ (Y_target + epsilon))) * 100;
    
    % ==========================================
    % 6. SMAPE (Symmetric Mean Absolute Percentage Error) 對稱平均絕對百分比誤差
    % 修正了 MAPE 在高估與低估時懲罰不對稱的缺點，值域落在 0%~200% 之間。
    % ==========================================
    denominator = (abs(Y_target) + abs(Y_predict)) / 2 + epsilon;
    metrics.SMAPE = mean(abs(error_val) ./ denominator) * 100;

end