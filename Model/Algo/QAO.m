function [gBest, thenParm, yAll] = QAO(tIter, H_train, Y_train, particleNum, baseVarFuzzyN)
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
                    % Expanded exploration (1) - Hydrogen Atom Wave Function
                    % gBest 當作質子位置, X_mean 當作電子的平均位置
                    % 使用逆變換抽樣從 1s 軌域的徑向分佈中為每個維度獨立得到距離 r
                    r_norm = inverse_transform_sampling(parmDim);
                    
                    % 對每個維度獨立計算新位置
                    % 每個維度的擾動範圍都在 gBest 和 X_mean 之間
                    % 使用氫原子波函數的機率分佈來決定新位置
                    direction = X_mean - gBest; % 從質子到電子平均位置的方向向量
                    
                    % 新位置 = 質子位置 + 正規化距離 * 方向向量
                    % 這確保每個維度都根據氫原子1s軌域機率分佈進行擾動
                    X_new = gBest + r_norm .* direction;
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

function r_norm = inverse_transform_sampling(dim)
    % 根據氫原子1s軌域的徑向機率分佈 P(r) ~ r^2 * exp(-2r) 進行逆變換抽樣
    % CDF: F(r) = 1 - (2r^2 + 2r + 1) * exp(-2r)
    % 我們需要解 u = F(r) 來找到 r，其中 u 是 (0,1) 上的均勻隨機數。
    % 這是一個超越方程式，我們使用數值方法（牛頓法）求解。
    
    u = rand(1, dim); % 為每個維度生成一個隨機數
    r = ones(1, dim); % 初始猜測值
    
    % 牛頓法參數
    max_iter = 20; % 增加最大迭代次數以提高精度
    tolerance = 1e-8; % 提高精度

    % 對每個維度獨立進行逆變換抽樣
    for i = 1:dim
        % 目標函數 g(r) = F(r) - u = 0
        g = @(x) 1 - (2*x^2 + 2*x + 1) * exp(-2*x) - u(i);
        % g(r) 的導數 g'(r) = F'(r) = P(r) = 4*x^2*exp(-2*x)
        g_prime = @(x) 4 * x^2 * exp(-2*x);
        
        % 改善初始猜測值，根據 u(i) 的值來估計
        if u(i) < 0.5
            ri = 0.5; % 對於小的 u 值，從較小的 r 開始
        else
            ri = 2.0; % 對於大的 u 值，從較大的 r 開始
        end
        
        % 牛頓法迭代
        for j = 1:max_iter
            fx = g(ri);
            if abs(fx) < tolerance
                break;
            end
            f_prime_x = g_prime(ri);
            % 避免除以零或過小的導數值
            if abs(f_prime_x) < 1e-12
                ri = ri + tolerance; % 輕微擾動以跳出
                continue;
            end
            
            ri_new = ri - fx / f_prime_x;
            
            % 保持 r 為正且在合理範圍內
            if ri_new < 0
                ri_new = tolerance;
            elseif ri_new > 10 % 限制上界，因為 F(10) 已經非常接近 1
                ri_new = 10;
            end
            
            % 檢查收斂
            if abs(ri_new - ri) < tolerance
                break;
            end
            
            ri = ri_new;
        end
        r(i) = ri;
    end
    
    % 改善正規化方法
    % 使用更精確的正規化：將 r 值映射到 [0, 1] 區間
    % 由於 99.9% 的機率密度在 r ∈ [0, 6] 範圍內，使用此作為正規化基準
    r_max = 6.0; % 氫原子 1s 軌域的有效截止半徑
    r_norm = r / r_max;
    
    % 確保所有值都在 [0, 1] 範圍內
    r_norm = min(max(r_norm, 0), 1);
end
