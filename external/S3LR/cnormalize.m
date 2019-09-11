function [Y, Xnorm] = cnormalize(X, p)
% Normalize columns of a matrix to unit norm
% 
% [Y, Xnorm] = cnormalize(X, p) normalizes columns of matrix X to unit l_p
% norm, and returen the norm values to Xnorm and data to Y.

if ~exist('p', 'var')
    p = 2;
end

if p == Inf
    Xnorm = max(abs(X), [], 1);
else
    Xnorm = sum(abs(X) .^p, 1) .^(1/p);
end
Y = bsxfun(@rdivide, X, Xnorm + eps);