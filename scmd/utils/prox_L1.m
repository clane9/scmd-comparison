function A = prox_L1(X, lamb)
% prox_L1    Evaluate proximal operator of L1 norm
%
%   min_A lamb || A ||_1 + 1/2 || A - X ||_F^2
%
%   A = prox_frosqr(X, lamb)
A = sign(X).*max(abs(X) - lamb, 0);
end
