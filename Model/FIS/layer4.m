function [ruleOut] = layer4(H, cnsqParm, nfs)

%calculate consequence part
ruleOut = nfs.*([1, transpose(H) ]*cnsqParm);
end

