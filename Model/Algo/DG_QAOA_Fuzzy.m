function [gBest, thenParm, yAll] = DG_QAOA_Fuzzy(tIter, H_train, Y_train, particleNum, baseVarFuzzyN)
    % =========================================================================
    % Dynamic Gaussian Quantum Arithmetic Optimization Algorithm (DG-QAO)
    % 應用於複數模糊類神經網路 (cFIS) 特徵參數最佳化
    % 包含創新點：
    % 1. 動態高斯參數擾動 (Dynamic Gaussian Mu & Alpha)
    % 2. 雙層量子波函數映射 (4s for Exploration, 1s for Exploitation)
    % 3. 反轉換採樣法 O(1) 極速抽樣 (Inverse Transform Sampling)
    % =========================================================================
    
    % --- QAO 波函數初始化 (僅執行一次，節省龐大運算資源) ---
    fprintf('Initializing Quantum Wave Function Tables...\n');
    [cdf_1s, r_vals_1s, cdf_4s, r_vals_4s] = init_quantum_tables();
    fprintf('Quantum Tables Loaded Successfully!\n');

    % --- 網路與演算法參數初始化 ---
    parmDim = sum(baseVarFuzzyN) * 3; % 每個隸屬函數對應實部/虛部參數
    error = 0.001;
    pCnsqParm = cell(particleNum, 1);
    
    % AOA 專屬參數 (MOA 邊界)
    MOP_Max = 1.0;
    MOP_Min = 0.2;
    
    % DG-AOA 動態高斯擾動的邊界設定 (隨迭代衰減)
    sigma_start_mu = 0.15;
    sigma_end_mu = 0.005;
    sigma_start_alpha = 1.5;
    sigma_end_alpha = 0.05;
    
    % QAO 量子波函數截斷半徑設定
    R_4s_max = 70; % 探索階段 (覆蓋 99.99% 的 4s 機率)
    R_1s_max = 10; % 開發階段 (覆蓋 99.99% 的 1s 機率)
    
    % 初始化解 (Population)，設定邊界為 [0, 1]
    LB = 0;
    UB = 1;
    X = rand(particleNum, parmDim);
    Xnew = zeros(particleNum, parmDim);
    fitness = zeros(particleNum, 1);
    
    % --- 計算初始 Fitness (RMSE) ---
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
    
    % =========================================================================
    % 主迴圈 (Main Optimization Loop)
    % =========================================================================
    for t = 1:tIter
        
        % 1. DG-AOA: 計算當前迭代的動態標準差，並套用常態分佈擾動 Mu 與 Alpha
        sigma_mu = sigma_start_mu - (sigma_start_mu - sigma_end_mu) * (t / tIter);
        sigma_alpha = sigma_start_alpha - (sigma_start_alpha - sigma_end_alpha) * (t / tIter);
        
        Mu = 0.5 + sigma_mu * randn();      % 基準值 0.5 + 動態擾動
        Alpha = 5.0 + sigma_alpha * randn();% 基準值 5.0 + 動態擾動 (呼吸效應)
        
        % 2. 計算 MOP 與 MOA (代入具備擾動的 Alpha)
        MOP = 1 - ((t)^(1/Alpha) / (tIter)^(1/Alpha));
        MOA = MOP_Min + t * ((MOP_Max - MOP_Min) / tIter);
        
        for i = 1:particleNum
            for j = 1:parmDim
                r1 = rand();
                
                % ---------------------------------------------------------
                % 探索階段 (Exploration) - 乘法/除法 + 4s波函數大跳躍
                % ---------------------------------------------------------
                if r1 < MOA 
                    r2 = rand();
                    % 計算 X_temp (基礎 AOA 運算)
                    if r2 > 0.5 % 除法
                        X_temp = gBest(j) / (MOP + eps) * ((UB - LB) * Mu + LB);
                    else        % 乘法
                        X_temp = gBest(j) * MOP * ((UB - LB) * Mu + LB);
                    end
                    
                    % 獲取 4s 波函數採樣半徑
                    r_sample = get_rsample(cdf_4s, r_vals_4s);
                    
                    % 決定正負向擾動
                    pm = (rand() > 0.5) * 2 - 1; % 快速產生 1 或 -1
                    
                    % 最終更新公式 (分母為 16)
                    Xnew(i,j) = X_temp + pm * (r_sample / R_4s_max) * (gBest(j) - X_temp) / 16;
                    
                % ---------------------------------------------------------
                % 開採階段 (Exploitation) - 加法/減法 + 1s波函數高精度微調
                % ---------------------------------------------------------
                else 
                    r3 = rand();
                    % 計算 X_temp (基礎 AOA 運算)
                    if r3 > 0.5 % 減法
                        X_temp = gBest(j) - MOP * ((UB - LB) * Mu + LB);
                    else        % 加法
                        X_temp = gBest(j) + MOP * ((UB - LB) * Mu + LB);
                    end
                    
                    % 獲取 1s 波函數採樣半徑
                    r_sample = get_rsample(cdf_1s, r_vals_1s);
                    
                    % 決定正負向擾動
                    pm = (rand() > 0.5) * 2 - 1; % 快速產生 1 或 -1
                    
                    % 最終更新公式 (分母為 32，增強局部收斂穩定度)
                    Xnew(i,j) = X_temp + pm * (r_sample / R_1s_max) * (gBest(j) - X_temp) / 32;
                end
            end
            
            % 修正範圍 (確保不超過 0~1 的邊界)
            Xnew(i, :) = max(Xnew(i, :), LB);
            Xnew(i, :) = min(Xnew(i, :), UB);
            
            % 計算新解的 fitness
            [Y_output, pCnsqParm_new] = cFIS(H_train, Y_train, baseVarFuzzyN, Xnew(i, :));
            fNew = RMSE(Y_output, Y_train);
            
            % 貪婪選擇 (Greedy Selection)：若新解較好則替換
            if fNew < fitness(i)
                X(i, :) = Xnew(i, :);
                fitness(i) = fNew;
                pCnsqParm{i} = pCnsqParm_new;
            end
        end
        
        % --- 更新全域最佳解 (Global Best) ---
        [minFit, minIdx] = min(fitness);
        if minFit < gBestVal
            gBestVal = minFit;
            gBest = X(minIdx, :);
            thenParm = pCnsqParm{minIdx};
        end
        
        % 記錄當前迭代的最佳值
        yAll(t) = gBestVal;
        fprintf('DG-QAO Iteration %d: Best RMSE = %f\n', t, gBestVal);
        
        % 提前結束條件
        if gBestVal < error
            yAll = yAll(1:t); 
            break;
        end
    end
end

% =========================================================================
% 以下為局部輔助函數 (Local Functions)
% =========================================================================

function [cdf_1s, r_vals_1s, cdf_4s, r_vals_4s] = init_quantum_tables()
    % 初始化 1s 與 4s 波函數的 CDF 查找表
    % P(r) = r^2 * |R_nl(r)|^2
    
    % 1s 波函數 (n=1, l=0)
    P_1s = @(r) 4 .* (r.^2) .* exp(-2 .* r);
    % 4s 波函數 (n=4, l=0)
    P_4s = @(r) (r.^2) .* 0.25 .* exp(-r./2) .* ...
                (1 - 0.75.*r + 0.125.*(r.^2) - (1/192).*(r.^3)).^2;

    % 1s CDF 查找表
    r_vals_1s = linspace(0, 10, 2000);
    pdf_1s = P_1s(r_vals_1s);
    cdf_1s = cumtrapz(r_vals_1s, pdf_1s);
    cdf_1s = cdf_1s / max(cdf_1s); % 歸一化
    
    % 4s CDF 查找表
    r_vals_4s = linspace(0, 70, 5000);
    pdf_4s = P_4s(r_vals_4s);
    cdf_4s = cumtrapz(r_vals_4s, pdf_4s);
    cdf_4s = cdf_4s / max(cdf_4s); % 歸一化
    
    % 移除重複的 CDF 值確保可嚴格內插
    [cdf_1s, unique_idx_1s] = unique(cdf_1s);
    r_vals_1s = r_vals_1s(unique_idx_1s);
    
    [cdf_4s, unique_idx_4s] = unique(cdf_4s);
    r_vals_4s = r_vals_4s(unique_idx_4s);
end

function r_sample = get_rsample(cdf_table, r_vals)
    % 反轉換採樣法抽取量子半徑
    U = rand();
    % 1D 快速線性內插
    r_sample = interp1(cdf_table, r_vals, U, 'linear', 'extrap');
    if r_sample < 0
        r_sample = 0;
    end
end