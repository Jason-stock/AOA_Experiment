function [theta, p] = RLSE(b, theta, p, Y)
%p = p - ( p*transpose(b)*b*p )/( 1+b*p*transpose(b) );
%theta = theta + p*transpose(b)*( Y-b*theta );
p = p - ( p*b*transpose(b)*p )/( 1+transpose(b)*p*b );
theta = theta + p*b*( Y-transpose(b)*theta );
end