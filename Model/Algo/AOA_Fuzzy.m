function [gBest, thenParm, yAll] = AOA_Fuzzy(tIter, H_train, Y_train, particleNum, baseVarFuzzyN)
    % Arithmetic Optimization Algorithm (AOA) 應用於模糊類神經網路
    
    % --- 初始化參數 ---
    parmDim = sum(baseVarFuzzyN) * 3;
    error = 0.001;
    pCnsqParm = cell(particleNum, 1);
    
    % AOA 專屬參數
    MOP_Max = 1.0;
    MOP_Min = 0.2;
    Alpha = 5;
    Mu = 0.499;
    
    % 初始化解 (population)，假設範圍與 AO 相同介於 [0, 1]
    LB = 0;
    UB = 1;
    X = rand(particleNum, parmDim);
    Xnew = zeros(particleNum, parmDim);
    fitness = zeros(particleNum, 1);
    
    % --- 計算初始 fitness ---
    for i = 1:particleNum
        [Y_output, pCnsqParm{i}] = cFIS(H_train, Y_train, baseVarFuzzyN, X(i, :));
        fitness(i) = RMSE(Y_output, Y_train);
    end
    
    % 找出初始的全域最佳解 (Global Best)
    [gBestVal, idx] = min(fitness);
    gBest = X(idx, :);
    thenParm = pCnsqParm{idx};
    
    % 儲存歷史適應度
    yAll = zeros(tIter, 1);
    
    % --- 主迴圈 (Main loop) ---
    for t = 1:tIter
        % 計算 Math Optimizer Probability (MOP) 與 Math Optimizer Accelerated (MOA)
        MOP = 1 - ((t)^(1/Alpha) / (tIter)^(1/Alpha));
        MOA = MOP_Min + t * ((MOP_Max - MOP_Min) / tIter);
        
        for i = 1:particleNum
            % 針對每一個維度進行位置更新
            for j = 1:parmDim
                r1 = rand();
                if r1 < MOA
                    % 乘法或除法 (Exploration)
                    r2 = rand();
                    if r2 > 0.5
                        Xnew(i,j) = gBest(j) / (MOP + eps) * ((UB - LB) * Mu + LB);
                    else
                        Xnew(i,j) = gBest(j) * MOP * ((UB - LB) * Mu + LB);
                    end
                else
                    % 加法或減法 (Exploitation)
                    r3 = rand();
                    if r3 > 0.5
                        Xnew(i,j) = gBest(j) - MOP * ((UB - LB) * Mu + LB);
                    else
                        Xnew(i,j) = gBest(j) + MOP * ((UB - LB) * Mu + LB);
                    end
                end
            end
            
            % 修正範圍 (確保不超過 0~1 的邊界)
            Xnew(i, :) = max(Xnew(i, :), LB);
            Xnew(i, :) = min(Xnew(i, :), UB);
            
            % 計算新 fitness
            [Y_output, pCnsqParm_new] = cFIS(H_train, Y_train, baseVarFuzzyN, Xnew(i, :));
            fNew = RMSE(Y_output, Y_train);
            
            % 如果新解比較好，則更新個體資訊
            if fNew < fitness(i)
                X(i, :) = Xnew(i, :);
                fitness(i) = fNew;
                pCnsqParm{i} = pCnsqParm_new;
            end
        end
        
        % --- 更新全域最佳 (Global Best) ---
        [minFit, minIdx] = min(fitness);
        if minFit < gBestVal
            gBestVal = minFit;
            gBest = X(minIdx, :);
            thenParm = pCnsqParm{minIdx};
        end
        
        % 記錄並印出當前迭代的最佳值
        yAll(t) = gBestVal;
        fprintf('Iteration %d: Best Cost = %f\n', t, gBestVal);
        
        % 提前結束條件
        if gBestVal < error
            % 如果提早結束，截斷多餘的 yAll 長度
            yAll = yAll(1:t); 
            break;
        end
    end
end