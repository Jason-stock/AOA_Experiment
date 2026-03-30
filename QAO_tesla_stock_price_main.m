addpath(fullfile(pwd, 'Model'),fullfile(pwd, 'Model/Result'));
data = readtable("Tesla Stock Price History.csv");

openPrice = data.Open;     % 開盤價欄位
openPrice = openPrice(1:1000);

closePrice = data.Price;     % 開盤價欄位
closePrice = closePrice(1:1000);

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
nRuns = 3;
finalRMSEs = zeros(nRuns,1);      % 每次最後 RMSE

%使用optimizer找出模型最佳參數
tIter = 30;
for r = 1:nRuns
    fprintf('=== 第 %d 次實驗 ===\n', r);
    [ifParm, cnsqParm, baseVarFuzzyN, lossAll] = optimizer(H_train, Y_train, tIter);
    finalRMSEs(r) = lossAll(end);        % 記錄最後 RMSE

    %使用approxiamtor預測資料
    Y_predict_train = approximator(H_train, ifParm, cnsqParm, baseVarFuzzyN);
    Y_predict_test = approximator(H_test, ifParm, cnsqParm, baseVarFuzzyN);

    %輸出圖形與損失值
    printLoss(Y_predict_train, Y_train, Y_predict_test, Y_test);
    model_plot([Y_predict_train,Y_train],[Y_predict_test,Y_test], lossAll );
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

rmpath(fullfile(pwd, 'Model'),fullfile(pwd, 'Model/Result'));