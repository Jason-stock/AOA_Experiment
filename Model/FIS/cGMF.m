function [cMbrDeg] = cGMF(h,m,s,l)
%h: base variable, s: sigma, m: mean, l:lambda
    r = gaussmf(h,m,s);
    
    %對m微分
    %w = -gaussmf(h,m,s)*((h-m)/s^2)*l;

    %對h微分
    w = -gaussmf(h,m,s)*(-(h-m)/s^2)*l;

    %對s微分
    %w = -gaussmf(h,m,s)*((h-m)/s^3)*l;

    cMbrDeg = r*exp(w*1i);
end