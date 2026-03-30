function [gBest, thenParm, yAll] = AO(tIter, H_train, Y_train, particleNum, baseVarFuzzyN)
    % Aquila Optimizer (AO)
    % 初始化參數
    parmDim = sum(baseVarFuzzyN) * 3;
    error = 0.001;
    pCnsqParm = cell(particleNum, 1);
    
    % AO 參數
    alpha = 0.1;            % 調整參數 (exploitation adjustment parameter)
    delta = 0.1;            % 調整參數 (exploitation adjustment parameter)
    u = 0.00565;            % Levy flight 參數
    r1 = 10;                % Levy flight 參數
    
    % 初始化解 (population)
    X = rand(particleNum, parmDim);
    fitness = zeros(particleNum, 1);
    
    % 計算初始 fitness
    for i = 1:particleNum
        [Y_output, pCnsqParm{i}] = cFIS(H_train, Y_train, baseVarFuzzyN, X(i, :));
        fitness(i) = RMSE(Y_output, Y_train);
    end
    
    [gBestVal, idx] = min(fitness);
    gBest = X(idx, :);
    thenParm = pCnsqParm{idx};

    % 儲存歷史
    yAll = zeros(tIter, 1);
    
    for t = 1:tIter
        % 更新 x_mean
        X_mean = mean(X);
        
        % 更新 G1 和 G2
        G1 = 2 * rand() - 1; %
        G2 = 2 * (1 - t / tIter); %
        
        for i = 1:particleNum
            % 根據 t 和 rand 選擇策略
            if t <= (2/3) * tIter
                if rand < 0.5
                    % Expanded exploration (1)
                    X_new = gBest .* (1 - t/tIter) + (X_mean - gBest * rand);
                else
                    % Narrowed exploration (2)
                    levy_val = levy(parmDim, u, r1);
                    rand_idx = floor(particleNum * rand) + 1;
                    X_rand = X(rand_idx, :);
                    X_new = gBest .* levy_val + X_rand + (rand - 0.5) * 0.001;
                end
            else
                if rand < 0.5
                    % Expanded exploitation (3)
                    X_new = (gBest - X_mean) * alpha - rand + ((1 - 0) * rand + 0) * delta;
                else
                    % Narrowed exploitation (4)
                    QF = t^((2*rand-1)/(1-tIter)^2);
                    X_new = QF .* gBest - (G1 .* X(i, :) * rand) - G2 .* levy(parmDim, u, r1) + rand * G1;
                end
            end
            
            % 修正範圍
            X_new = max(X_new, 0);
            X_new = min(X_new, 1);
            
            % 計算新 fitness
            [Y_output, pCnsqParm_new] = cFIS(H_train, Y_train, baseVarFuzzyN, X_new);
            fNew = RMSE(Y_output, Y_train);
            
            % 更新個體
            if fNew < fitness(i)
                X(i, :) = X_new;
                fitness(i) = fNew;
                pCnsqParm{i} = pCnsqParm_new;
            end
        end
        
        % 更新全域最佳
        [minFit, minIdx] = min(fitness);
        
        if minFit < gBestVal
            gBestVal = minFit;
            gBest = X(minIdx, :);
            thenParm = pCnsqParm{minIdx};
        end
        
        yAll(t) = gBestVal;
        fprintf('%d %f\n', t, gBestVal);
        
        if gBestVal < error
            break;
        end
    end
end

function o = levy(d, u, r1)
    w = u * randn(1, d);
    v = randn(1, d);
    beta = 1.5;
    step = w ./ (abs(v).^(1/beta));
    o = 0.01 * step;
end