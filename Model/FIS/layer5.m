function [Y_output] = layer5(ruleOut)

%calculate fuzzy system output
Y_output = sum(ruleOut);
end