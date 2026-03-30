function [fuzzySigma, fuzzyMu, fuzzyLambda] = getFuzzyParm(parm, baseVarFuzzyN, baseVarNO, fuzzyNO)
%將parm拆解成fuzzy set參數
if(baseVarNO==1)
    startIdx = 1+(fuzzyNO-1)*3;
else
    startIdx = ( sum(baseVarFuzzyN(1:baseVarNO-1,1))+(fuzzyNO-1) )*3+1;
end

fuzzySigma = parm(1,startIdx);
fuzzyMu = parm(1,startIdx+1);
fuzzyLambda = parm(1,startIdx+2);
end