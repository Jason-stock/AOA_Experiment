function [nfs] = layer3(strength)

%calculate normalized firing strengths
nfs = strength(1,:)/sum(strength(1,:));
end

