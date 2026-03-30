function [strength] = layer2(mbrDeg, fuzzyNForVar)

%calculate firing strengths of fuzzy set
strength = reshape(mbrDeg{1}(1,:),[1,fuzzyNForVar(1,1)]);
for j = 2:size(fuzzyNForVar,1)
    strength = reshape(mbrDeg{j}(1,:),[fuzzyNForVar(j,1),1])*strength;
    sSize = size(strength);
    strength = reshape(strength,[1,sSize(1)*sSize(2)]);
end
end

