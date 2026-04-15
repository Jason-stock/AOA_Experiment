function [Y_predict] = approximator_nonlinear(H, ifParm, cnsqParm, baseVarFuzzyN)
    % 初始化每條rule的係數與佔整體結果的權重
    HNum = size(H,1);
    Y_predict = zeros(HNum,1);
    ruleNum = prod(baseVarFuzzyN);
    nfs = zeros(HNum,ruleNum);

    HT = layer0(H);
    for i=1:HNum
        %If Part
        h = HT(:,i);
        mbrDeg = layer1(h, ifParm, baseVarFuzzyN);
        strength = layer2(mbrDeg, baseVarFuzzyN);
        nfs(i,:) = layer3(strength);

        %then part
        % 這裡呼叫的是下方定義的非線性版 layer4
        ruleOut = layer4_nonlinear(h, cnsqParm, nfs(i,:)); 
        Y_predict(i,1) = layer5(ruleOut);
    end
end

% --- 以下為 Local Functions ---

function [H] = layer0(H)
    H = transpose(H);
end

function [mbrDeg] = layer1(H, ifParm, fuzzyNForVar)
    numOfBaseVar = size(fuzzyNForVar,1);
    mbrDeg = cell(size(H,1), numOfBaseVar);
    for j = 1:numOfBaseVar
        for k = 1:fuzzyNForVar(j,1)
            [fuzzySigma, fuzzyMu, fuzzyLambda] = getFuzzyParm(ifParm, fuzzyNForVar, j, k);
            mbrDeg{j}(1,k) = cGMF(H(j,1), fuzzySigma, fuzzyMu, fuzzyLambda);
        end
    end
end

function [strength] = layer2(mbrDeg, fuzzyNForVar)
    strength = reshape(mbrDeg{1}(1,:),[1,fuzzyNForVar(1,1)]);
    for j = 2:size(fuzzyNForVar,1)
        strength = reshape(mbrDeg{j}(1,:),[fuzzyNForVar(j,1),1])*strength;
        sSize = size(strength);
        strength = reshape(strength,[1,sSize(1)*sSize(2)]);
    end
end

function [nfs] = layer3(strength)
    nfs = strength(1,:)/sum(strength(1,:));
end

function [ruleOut] = layer4_nonlinear(H, cnsqParm, nfs)
    % 核心修改：同步構造與訓練時相同的擴張輸入 [1; h1; h1^2; h2; h2^2; ...]
    h_ext = [1; reshape([H, H.^2]', [], 1)];
    ruleOut = nfs.*(transpose(h_ext)*cnsqParm);
end

function [Y_output] = layer5(ruleOut)
    Y_output = sum(ruleOut);
end

function [fuzzySigma, fuzzyMu, fuzzyLambda] = getFuzzyParm(parm, baseVarFuzzyN, baseVarNO, fuzzyNO)
    if(baseVarNO==1)
        startIdx = 1+(fuzzyNO-1)*3;
    else
        startIdx = ( sum(baseVarFuzzyN(1:baseVarNO-1,1))+(fuzzyNO-1) )*3+1;
    end
    fuzzySigma = parm(1,startIdx);
    fuzzyMu = parm(1,startIdx+1);
    fuzzyLambda = parm(1,startIdx+2);
end

function [cMbrDeg] = cGMF(h,m,s,l)
    r = gaussmf(h,m,s);
    w = -gaussmf(h,m,s)*(-(h-m)/s^2)*l;
    cMbrDeg = r*exp(w*1i);
end

function [mbrDeg] = gaussmf(h,m,s)
    mbrDeg = exp( -((h-m)*(h-m)')/(2*(s)^2) );
end