function [X, groups, Omega] = generate_scmd_data(n, d, D, Ng, sigma, ell, seed)
% generate_scmd_data   Generate a synthetic union of subspaces dataset with
%   noise and missing data.
%
%   [X, groups, Omega] = generate_scmd_data_matrix(n, d, D, Ng, sigma, ...
%       ell, seed)
%
%   Complete data are sampled as
%
%   X = [U_1 V_1 ... U_n V_n] + E
%
%   where the U_i are random orthonormal bases of dimension D x d, V_i are
%   random d x Ng coefficients sampled from N(0, 1/d), and E is D x Ng*n dense
%   Gaussian noise sampled from N(0, sigma^2/D). Observed entries are sampled
%   uniformly at random at a rate ell/D.
%
%   Args:
%     n: number of subspaces
%     d: subspace dimension
%     D: ambient dimension
%     Ng: number of points per group
%     sigma: dense noise sigma
%     ell: expected number of observed entries per column
%     seed: seed for random generator [default: none]
%
%   Returns:
%     X: D x (Ng*n) complete, noisy data matrix
%     groups: (Ng*n) x 1 cluster assignment
%     Omega: D x (Ng*n) logical indicator matrix of observed entries, so that the
%       observed data Xobs = X.*Omega.
if nargin >= 7; rng(seed); end

if n <= 0 || d <= 0 || d > D || D <= 0
  error('ERROR: invalid synthetic setting.')
end

% sample noiseless data from n subspaces, with columns approximately unit norm
N = Ng*n;
X = zeros(D, N);
groups = zeros(N, 1);
for ii=1:n
  U = orth(randn(D, d));
  V = (1/sqrt(d)) * randn(d, Ng);
  startidx = (ii-1)*Ng + 1; stopidx = startidx + Ng - 1;
  groups(startidx:stopidx) = ii;
  X(:, startidx:stopidx) = U*V;
end

% add dense noise
if sigma > 0
  X = X + (sigma/sqrt(D)) * randn(D, N);
end

% sample ell observed entries per column (in expectation) uniformly at random
if ell < D && ell > 0; obs_rate = ell/D; else; obs_rate = 1; end
Omega = rand(D, N) <= obs_rate;
end
