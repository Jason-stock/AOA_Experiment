function [ifParm, cnsqParm, baseVarFuzzyN, lossAll] = optimizer(H_train, Y_train, tIter)
addpath(fullfile(pwd, 'Model', 'LossFunc'), fullfile(pwd, 'Model', 'FIS'), fullfile(pwd, 'Model', 'Algo'));
particleNum = 60;
baseVarFuzzyN = [2;2;2;2];
[ifParm, cnsqParm, lossAll] = AOA_Fuzzy(tIter, H_train, Y_train, particleNum, baseVarFuzzyN);

rmpath(fullfile(pwd, 'Model', 'LossFunc'), fullfile(pwd, 'Model', 'FIS'), fullfile(pwd, 'Model', 'Algo'));
end