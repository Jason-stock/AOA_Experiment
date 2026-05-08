function [ifParm, cnsqParm, baseVarFuzzyN, lossAll] = optimizer(H_train, Y_train, tIter)
addpath(fullfile(pwd, 'Model', 'LossFunc'), fullfile(pwd, 'Model', 'FIS'), fullfile(pwd, 'Model', 'Algo'));
particleNum = 60;
baseVarFuzzyN = [4;4];
[ifParm, cnsqParm, lossAll] = DG_QAOA_Fuzzy_V2(tIter, H_train, Y_train, particleNum, baseVarFuzzyN);

rmpath(fullfile(pwd, 'Model', 'LossFunc'), fullfile(pwd, 'Model', 'FIS'), fullfile(pwd, 'Model', 'Algo'));
end