function A = prox_L21(X, lamb)
% prox_L21    Evaluate proximal operator \ell_2,1 norm:
%   ||A||_{2,1} = \sum_i ||A_{.,i}||_2
%
%   min_A lamb || A ||_{2,1} + 1/2 || A - X ||_F^2
%
%   A = prox_L21(X, lamb)
colnorms = sqrt(sum(X.^2));
[D, N] = size(X);
coeffs = max(ones(1, N) - lamb./(colnorms+eps), 0);
A = repmat(coeffs, [D 1]).*X;
end
