function Z = tsc(X, q)
% tsc   Threshold subspace clustering.
%
%   Z = tsc(X, q)
%
%   Args:
%     X: D x N data matrix
%     q: number of nearest-neigbors
%
%   Returns:
%     Z: N x N sparse symmetric affinity matrix
%
%   References:
%     Heckel, R., & Bolcskei, H. (2015). Robust subspace clustering via
%     thresholding. IEEE Transactions on Information Theory, 61(11), 6320-6342.
[D, N] = size(X);

% normalize and take dot products
X = X ./ repmat(sqrt(sum(X.^2))+eps, [D 1]);
G = abs(X'*X);

% construct weighted q-nn affinity
[topqG, topqI] = maxk(G, q);
topqJ = repmat(1:N, [q 1]);
Z = sparse(topqI(:), topqJ(:), exp(-2 * acos(topqG(:))), N, N, 2*q*N);
Z = Z + Z';
end
