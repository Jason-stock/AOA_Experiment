function [mbrDeg] = gaussmf(h,m,s)
mbrDeg = exp( -((h-m)*(h-m)')/(2*(s)^2) );
end