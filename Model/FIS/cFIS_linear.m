function [Y_output, cnsqParm] = cFIS_linear(H, Y, baseVarFuzzyN, ifParm)
    HNum = size(H,1);
    Y_output = zeros(HNum,1);

    %初始化每條rule的係數與佔整體結果的權重
    ruleNum = prod(baseVarFuzzyN);
    nfs = zeros(HNum,ruleNum);

    %initialize RLSE parameters
    numOfCoeff = size(H,2)+1;
    p = 10^8*eye(ruleNum*numOfCoeff);
    theta = zeros(ruleNum*numOfCoeff,1);

    HT = layer0(H);

    %If Part
    for i=1:HNum
        h = HT(:,i);
        mbrDeg = layer1(h, ifParm, baseVarFuzzyN);
        strength = layer2(mbrDeg, baseVarFuzzyN);
        nfs(i,:) = layer3(strength);
    end

    %RLSE
    for i=1:HNum
        h = HT(:,i);
        %將矩陣reshape成行向量
        b = reshape([1;h]*nfs(i,:),[ruleNum*numOfCoeff,1]);
        [theta,p] = RLSE(b, theta, p, Y(i,:));
    end

    %then part
    cnsqParm = reshape(theta, [numOfCoeff, ruleNum]);
    for i = 1:HNum
        h = HT(:,i);
        ruleOut = layer4(h,cnsqParm,nfs(i,:));
        Y_output(i,1) = layer5(ruleOut);
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

function [ruleOut] = layer4(H, cnsqParm, nfs)
    ruleOut = nfs.*([1, transpose(H) ]*cnsqParm);
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

function [theta, p] = RLSE(b, theta, p, Y)
    p = p - ( p*b*transpose(b)*p )/( 1+transpose(b)*p*b );
    theta = theta + p*b*( Y-transpose(b)*theta );
end