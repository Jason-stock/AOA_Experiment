function [Y_predict] = approximator(H, ifParm, cnsqParm, baseVarFuzzyN)
addpath(fullfile(pwd, 'Model', 'FIS'));

%初始化每條rule的係數與佔整體結果的權重
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
    ruleOut = layer4(h,cnsqParm,nfs(i,:));
    Y_predict(i,1) = layer5(ruleOut);
end

rmpath(fullfile(pwd, 'Model', 'FIS'));
end