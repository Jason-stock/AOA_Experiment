function [gBest, thenParm, yAll] = QAO_V2(tIter, H_train, Y_train, particleNum, baseVarFuzzyN)
    % QAO (Quantum-based Aquila Optimizer) - 修改版
    % 核心邏輯來自 QAO_benchmark_multimodal_V2.m 中的 qao_run
    % 保持 cFIS 的適應度評估框架
    
    % 初始化參數 (來自原始 QAO.m)
    parmDim = sum(baseVarFuzzyN) * 3;
    error = 0.001;
    pCnsqParm = cell(particleNum, 1);
    
    % QAO 參數 (來自 qao_run)
    alpha = 0.1;            % exploitation 調整
    delta = 0.1;            % exploitation 調整
    u = 0.00565;            % Levy flight 參數
    w  = 0.005;               % ω (用於 Narrowed exploration 2)
    theta1 = 3*pi/2;        % (用於 Narrowed exploration 2)

    % 初始化解 (population) (來自原始 QAO.m)
    X = rand(particleNum, parmDim);
    fitness = zeros(particleNum, 1);
    
    % 計算初始 fitness (來自原始 QAO.m, 使用 cFIS 和 RMSE)
    for i = 1:particleNum
        [Y_output, pCnsqParm{i}] = cFIS(H_train, Y_train, baseVarFuzzyN, X(i, :));
        fitness(i) = RMSE(Y_output, Y_train);
    end
    
    [gBestVal, idx] = min(fitness);
    gBest = X(idx, :);
    thenParm = pCnsqParm{idx};

    % 儲存歷史 (來自原始 QAO.m)
    yAll = zeros(tIter, 1);
    
    % --- QAO (qao_run) 氫原子機率分佈初始化 ---
    a0 = 1; 
    num_slots = 10000;
    num_points = num_slots + 1;
    % 原始 X 軸：物理距離 r (範圍 [0, 30])
    r_values_original = linspace(0, 30, num_points);
    % Y 軸：計算對應的機率密度 P(r)
    P_r_values = (4 / a0^3) .* (r_values_original.^2) .* exp(-2 .* r_values_original ./ a0);
    % 歸一化Y軸面積
    area = trapz(r_values_original, P_r_values);
    P_r_values_normalized = P_r_values / area;
    % -----------------------------------------
    
    for t = 1:tIter
        % 更新 x_mean (來自 qao_run)
        X_mean = mean(X, 1);
        G1 = 2*rand() - 1;
        G2 = 2*(1 - t/tIter); % 使用 tIter

        for i = 1:particleNum % 遍歷所有粒子
            
            % --- QAO 核心邏輯 (來自 qao_run) ---
            if t <= (2/3)*tIter
                if rand < 0.5
                    % Expanded exploration (1) - 氫原子 1s 徑向 (gBest-X_mean 模型)
                    Umin_x = mean(X_mean);
                    Umax_x = mean(gBest);

                    r_values_rescaled = rescale_axis_range(r_values_original, Umin_x, Umax_x);
                    generated_samples = inverse_transform_sampling(r_values_rescaled, P_r_values_normalized, parmDim);
                    X_new = gBest .* (1 - t/tIter) + generated_samples;
                else
                    % Narrowed exploration (2) - 氫原子 1s 徑向分佈（原點-電子模型）
                    rand_idx = floor(particleNum * rand) + 1;
                    X_rand = X(rand_idx, :);  % 隨機個體作為電子位置
                    
                    origin = zeros(1, parmDim);
                    Umin_x = mean(origin);
                    Umax_x = mean(X_rand);

                    r_values_rescaled = rescale_axis_range(r_values_original, Umin_x, Umax_x);
                    generated_samples = inverse_transform_sampling(r_values_rescaled, P_r_values_normalized, parmDim);

                    r1_qao = 1 + (20-1)*rand;     % r1 ∈ [1,20] (qao_run中的r1)
                    D1 = 1:parmDim;
                    r = r1_qao + u * D1;
                    theta = -w * D1 + theta1;
    
                    y = r .* cos(theta);
                    x = r .* sin(theta);
                    spiral = (y - x) .* rand(1, parmDim);
                    
                    X_new = gBest.*levy_step(parmDim, u) + X_rand + generated_samples + spiral;
                end
            else
                if rand < 0.5
                    % Expanded exploitation (3)
                    Umin_x = mean(gBest);  % 氫原子
                    Umax_x = mean(X_mean); % 電子

                    r_values_rescaled = rescale_axis_range(r_values_original, Umin_x, Umax_x);
                    generated_samples = inverse_transform_sampling(r_values_rescaled, P_r_values_normalized, parmDim);

                    % 原始 qao_run: ((ub - lb) * rand)*delta
                    % 由於 QAO.m 在 [0,1] 範圍操作, (ub-lb) = 1
                    X_new = (gBest - generated_samples)*alpha - rand + (rand)*delta;
                else
                    % Narrowed exploitation (4)
                    QF = t^((2*rand - 1)/(1 - tIter)^2); % 使用 tIter
                    X_new = QF .* gBest - (G1 .* X(i, :) * rand) - G2 .* levy_step(parmDim, u) + rand*G1;
                end
            end
            % --- QAO 核心邏輯結束 ---
            
            % 修正範圍 (來自原始 QAO.m)
            X_new = max(X_new, 0);
            X_new = min(X_new, 1);
            
            % 計算新 fitness (來自原始 QAO.m, 使用 cFIS 和 RMSE)
            [Y_output, pCnsqParm_new] = cFIS(H_train, Y_train, baseVarFuzzyN, X_new);
            fNew = RMSE(Y_output, Y_train);
            
            % 更新個體 (來自原始 QAO.m)
            if fNew < fitness(i)
                X(i, :) = X_new;
                fitness(i) = fNew;
                pCnsqParm{i} = pCnsqParm_new;
            end
        end
        
        % 更新全域最佳 (來自原始 QAO.m)
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

% ===================== 輔助函數 (來自 QAO_benchmark_multimodal_V2.m) =====================

function step = levy_step(d, u)
    % 與 qao_run 版本一致的 Lévy 取樣（beta=1.5，比例 0.01）
    w = u * randn(1, d);
    v = randn(1, d);
    beta = 1.5;
    step = 0.01 * (w ./ (abs(v).^(1/beta)));
end

function samples = inverse_transform_sampling(x_values, pdf_values, num_samples)
%INVERSE_TRANSFORM_SAMPLING 從數值 PDF 生成隨機樣本 (來自 qao_run)

    % 1. 計算數值 CDF (累積分佈函數)
    x_values = x_values(:)';
    pdf_values = pdf_values(:)';
    CDF_values = cumtrapz(x_values, pdf_values);

    % 2. 確保 CDF 範圍是 [0, 1] 
    CDF_values(1) = 0;
    CDF_values(end) = 1;
    
    % 3. 處理 CDF 中的平坦區域 (重複值)
    [CDF_unique, ia] = unique(CDF_values, 'last');
    x_unique = x_values(ia);

    % 4. 生成 U(0, 1) 的均勻分佈隨機數
    u = rand(num_samples, 1);

    % 5. 執行逆轉換：使用 "乾淨" 且 "唯一" 的 CDF 和 X 值進行插值
    samples = interp1(CDF_unique, x_unique, u);
    samples = samples(:)'; 
end

function U_x = rescale_axis_range(G_x, Umin_x, Umax_x)
%RESCALE_AXIS_RANGE 將輸入的 X 軸向量 G_x 線性映射到一個新的範圍 [Umin_x, Umax_x] (來自 qao_run)
%
    % 獲取 G_x 的實際最大值和最小值
    Gmin_x = min(G_x);
    Gmax_x = max(G_x);
    
    % 檢查 Gmax 和 Gmin 是否相同，以避免除以零
    if Gmax_x == Gmin_x
        U_x = ones(size(G_x)) * Umin_x;
        % warning('輸入的 X 軸向量所有元素都相同。'); % 註解掉 warning 避免干擾
        return;
    end

    % 應用標準範圍調整公式
    U_x = ((G_x - Gmin_x) ./ (Gmax_x - Gmin_x)) .* (Umax_x - Umin_x) + Umin_x;
end

% 附註：原始 QAO.m 中的 RMSE 函數未在此處提供，
% 假設它在 QAO.m 的可呼叫範圍內（例如在同一目錄或已加入路徑）。