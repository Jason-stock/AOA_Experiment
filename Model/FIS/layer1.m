function [mbrDeg] = layer1(H, ifParm, fuzzyNForVar)
numOfBaseVar = size(fuzzyNForVar,1);
mbrDeg = cell(size(H,1), numOfBaseVar);

%calculate membership degree for each fuzzy sets
for j = 1:numOfBaseVar
    for k = 1:fuzzyNForVar(j,1)
        [fuzzySigma, fuzzyMu, fuzzyLambda] = getFuzzyParm(ifParm, fuzzyNForVar, j, k);
        mbrDeg{j}(1,k) = cGMF(H(j,1), fuzzySigma, fuzzyMu, fuzzyLambda);
    end
end
end