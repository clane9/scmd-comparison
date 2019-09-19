function A = prox_frosqr(X, lamb)
% prox_frosqr    Evaluate proximal operator of Frobenius squared norm
%
%   min_A lamb || A ||_F^2 + 1/2 || A - X ||_F^2
%
%   A = prox_frosqr(X, lamb)
A = (1/(1 + 2*lamb)) * X;
end
