function [Y_output, cnsqParm] = cFIS(H, Y, baseVarFuzzyN, ifParm)
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
    %將5*24矩陣reshape成120*1矩陣
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

