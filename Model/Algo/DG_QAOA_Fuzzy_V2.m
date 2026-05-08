function [gBest, thenParm, yAll] = DG_QAOA_Fuzzy_V2(tIter, H_train, Y_train, particleNum, baseVarFuzzyN)
    % =========================================================================
    % Dynamic Gaussian Quantum Arithmetic Optimization Algorithm (DG-QAOA)
    % Boolean-Controlled Hydrogen Wave Function Version
    %
    % 核心概念：
    % 1. 每個粒子 X(i,:) 代表一條完整候選解
    % 2. gBest 代表歷史最佳候選解
    % 3. 徑向距離方向指向 gBest
    % 4. use4sWave = true  時，使用 4s 波函數更新候選解，偏探索
    % 5. use4sWave = false 時，使用 1s 波函數更新候選解，偏開採
    % =========================================================================

    % --- 初始化氫原子波函數 CDF 查找表 ---
    fprintf('Initializing Quantum Wave Function Tables...\n');
    [cdf_1s, r_vals_1s, cdf_4s, r_vals_4s] = init_quantum_tables();
    fprintf('Quantum Tables Loaded Successfully!\n');

    % --- 基本參數 ---
    parmDim = sum(baseVarFuzzyN) * 3;
    targetRMSE = 0.001;

    pCnsqParm = cell(particleNum, 1);

    % AOA 參數
    MOP_Max = 1.0;
    MOP_Min = 0.2;

    % 動態高斯擾動參數
    sigma_start_mu = 0.15;
    sigma_end_mu   = 0.005;

    sigma_start_alpha = 1.5;
    sigma_end_alpha   = 0.05;

    % 搜尋邊界
    LB = 0;
    UB = 1;

    % 氫原子波函數截斷半徑
    R_4s_max = 70;
    R_1s_max = 10;

    % 初始化族群
    X = LB + (UB - LB) .* rand(particleNum, parmDim);
    Xnew = zeros(particleNum, parmDim);
    fitness = zeros(particleNum, 1);

    % --- 初始 fitness ---
    for i = 1:particleNum
        [Y_output, pCnsqParm{i}] = cFIS(H_train, Y_train, baseVarFuzzyN, X(i, :));
        fitness(i) = RMSE(Y_output, Y_train);
    end

    % --- 初始歷史最佳解 ---
    [gBestVal, idx] = min(fitness);
    gBest = X(idx, :);
    thenParm = pCnsqParm{idx};

    % 收斂曲線
    yAll = zeros(tIter, 1);

    % =========================================================================
    % 主迴圈
    % =========================================================================
    for t = 1:tIter

        % =============================================================
        % 1. 動態高斯擾動
        % =============================================================
        sigma_mu = sigma_start_mu - ...
            (sigma_start_mu - sigma_end_mu) * (t / tIter);

        sigma_alpha = sigma_start_alpha - ...
            (sigma_start_alpha - sigma_end_alpha) * (t / tIter);

        Mu = 0.5 + sigma_mu * randn();
        Alpha = 5.0 + sigma_alpha * randn();

        % 限制 Mu 與 Alpha，避免極端值
        Mu = min(max(Mu, 0.05), 0.95);
        Alpha = min(max(Alpha, 1.0), 10.0);

        % =============================================================
        % 2. MOP 與 MOA
        % =============================================================
        MOP = 1 - ((t)^(1 / Alpha) / (tIter)^(1 / Alpha));
        MOP_safe = max(MOP, 1e-3);

        % MOA 從 0.2 增加到 1.0
        % 前期探索機率高，後期開採機率高
        MOA = MOP_Min + t * ((MOP_Max - MOP_Min) / tIter);

        % =============================================================
        % 3. 更新每個粒子
        % =============================================================
        for i = 1:particleNum

            Xi = X(i, :);

            % ---------------------------------------------------------
            % 從目前候選解 Xi 指向歷史最佳候選解 gBest 的方向
            % ---------------------------------------------------------
            direction_vec = gBest - Xi;
            dist_to_gBest = norm(direction_vec);

            if dist_to_gBest < eps
                % 若 Xi 與 gBest 幾乎重合，給一個隨機方向避免停滯
                direction_unit = random_unit_vector(parmDim);
                dist_to_gBest = sqrt(parmDim) * (UB - LB);
            else
                direction_unit = direction_vec ./ dist_to_gBest;
            end

            % ---------------------------------------------------------
            % 宣告布林值：
            % true  -> 使用 4s 波函數，探索階段
            % false -> 使用 1s 波函數，開採階段
            % ---------------------------------------------------------
            r1 = rand();

            if r1 > MOA
                use4sWave = true;     % Exploration
            else
                use4sWave = false;    % Exploitation
            end

            % ---------------------------------------------------------
            % 根據布林值選擇波函數
            % ---------------------------------------------------------
            if use4sWave == true

                % =====================================================
                % Exploration：使用 4s 波函數
                % =====================================================
                r_sample = get_rsample(cdf_4s, r_vals_4s);

                % 正規化徑向距離
                q_radius = r_sample / R_4s_max;

                % 4s 探索步長較大
                wave_step_scale = 0.35;

                % AOA 探索項
                r2 = rand();

                if r2 > 0.5
                    AOA_center = gBest ./ MOP_safe .* ((UB - LB) * Mu + LB);
                else
                    AOA_center = gBest .* MOP .* ((UB - LB) * Mu + LB);
                end

                % 保留部分目前候選解，避免完全被 gBest 支配
                lambda = rand();
                base_position = lambda .* Xi + (1 - lambda) .* AOA_center;

            else

                % =====================================================
                % Exploitation：使用 1s 波函數
                % =====================================================
                r_sample = get_rsample(cdf_1s, r_vals_1s);

                % 正規化徑向距離
                q_radius = r_sample / R_1s_max;

                % 1s 開採步長較小
                wave_step_scale = 0.08;

                % AOA 開採項
                r3 = rand();

                if r3 > 0.5
                    AOA_center = gBest - MOP .* ((UB - LB) * Mu + LB);
                else
                    AOA_center = gBest + MOP .* ((UB - LB) * Mu + LB);
                end

                % 後期 MOA 越大，越靠近 gBest 附近
                lambda = rand() * (1 - MOA);
                base_position = lambda .* Xi + (1 - lambda) .* AOA_center;
            end

            % ---------------------------------------------------------
            % 由波函數生成徑向距離
            % 並且讓該徑向距離指向歷史最佳候選解 gBest
            % ---------------------------------------------------------
            radial_distance = wave_step_scale .* q_radius .* dist_to_gBest;

            Xnew(i, :) = base_position + radial_distance .* direction_unit;

            % ---------------------------------------------------------
            % 邊界處理
            % ---------------------------------------------------------
            Xnew(i, :) = max(Xnew(i, :), LB);
            Xnew(i, :) = min(Xnew(i, :), UB);

            % ---------------------------------------------------------
            % 計算新候選解 fitness
            % ---------------------------------------------------------
            [Y_output, pCnsqParm_new] = cFIS(H_train, Y_train, baseVarFuzzyN, Xnew(i, :));
            fNew = RMSE(Y_output, Y_train);

            % ---------------------------------------------------------
            % Greedy Selection
            % ---------------------------------------------------------
            if fNew < fitness(i)
                X(i, :) = Xnew(i, :);
                fitness(i) = fNew;
                pCnsqParm{i} = pCnsqParm_new;
            end
        end

        % =============================================================
        % 4. 更新歷史最佳候選解
        % =============================================================
        [minFit, minIdx] = min(fitness);

        if minFit < gBestVal
            gBestVal = minFit;
            gBest = X(minIdx, :);
            thenParm = pCnsqParm{minIdx};
        end

        yAll(t) = gBestVal;

        fprintf('DG-QAOA Iteration %d: Best RMSE = %.10f\n', t, gBestVal);

        % 提前停止
        if gBestVal < targetRMSE
            yAll = yAll(1:t);
            fprintf('Early stopping at iteration %d. Target RMSE reached.\n', t);
            break;
        end
    end
end

% =========================================================================
% Local Function 1：初始化氫原子波函數 CDF 查找表
% =========================================================================
function [cdf_1s, r_vals_1s, cdf_4s, r_vals_4s] = init_quantum_tables()
    % 初始化 1s 與 4s 波函數的 CDF 查找表
    %
    % P(r) = r^2 * |R_nl(r)|^2
    %
    % 1s 用於 exploitation
    % 4s 用於 exploration

    % -----------------------------
    % 1s wave function, n = 1, l = 0
    % -----------------------------
    P_1s = @(r) 4 .* (r.^2) .* exp(-2 .* r);

    % -----------------------------
    % 4s wave function, n = 4, l = 0
    % -----------------------------
    P_4s = @(r) (r.^2) .* 0.25 .* exp(-r ./ 2) .* ...
        (1 - 0.75 .* r + 0.125 .* (r.^2) - (1 / 192) .* (r.^3)).^2;

    % -----------------------------
    % 1s CDF
    % -----------------------------
    r_vals_1s = linspace(0, 10, 2000);
    pdf_1s = P_1s(r_vals_1s);

    cdf_1s = cumtrapz(r_vals_1s, pdf_1s);
    cdf_1s = cdf_1s ./ max(cdf_1s);

    % -----------------------------
    % 4s CDF
    % -----------------------------
    r_vals_4s = linspace(0, 70, 5000);
    pdf_4s = P_4s(r_vals_4s);

    cdf_4s = cumtrapz(r_vals_4s, pdf_4s);
    cdf_4s = cdf_4s ./ max(cdf_4s);

    % -----------------------------
    % 移除重複 CDF 值，避免 interp1 出錯
    % -----------------------------
    [cdf_1s, unique_idx_1s] = unique(cdf_1s, 'stable');
    r_vals_1s = r_vals_1s(unique_idx_1s);

    [cdf_4s, unique_idx_4s] = unique(cdf_4s, 'stable');
    r_vals_4s = r_vals_4s(unique_idx_4s);
end

% =========================================================================
% Local Function 2：反轉換採樣
% =========================================================================
function r_sample = get_rsample(cdf_table, r_vals)
    % 使用反轉換採樣法抽取徑向距離

    U = rand();

    U = min(max(U, cdf_table(1)), cdf_table(end));

    r_sample = interp1(cdf_table, r_vals, U, 'linear');

    if isnan(r_sample) || r_sample < 0
        r_sample = 0;
    end
end

% =========================================================================
% Local Function 3：產生隨機單位向量
% =========================================================================
function v = random_unit_vector(dim)
    % 產生 dim 維空間中的隨機單位向量

    v = randn(1, dim);
    n = norm(v);

    if n < eps
        v = ones(1, dim) ./ sqrt(dim);
    else
        v = v ./ n;
    end
end