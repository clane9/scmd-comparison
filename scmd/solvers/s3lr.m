function [groups, Y, history] = s3lr(X, Omega, n, params)
% s3lr    Stripped down version of S3LR function from Chun-Guang Li, to be
%   consistent with other algorithms. Approximately solves the following
%   problem by alternating LADMM and spectral clustering.
%
%   min_{C, Y, E, Q}  \sum_{j=1}^k || Y diag(q_j) ||_*
%                     + gamma ( alpha || Theta(Q) \odot C ||_1 + || C ||_1)
%                     + lambda || E ||_1
%               s.t.  Y = YC + E, diag(C) = 0, Q^T Q = I
%
%   where (Theta(Q))_{i j} = 0.5 * || Q_{i,.} - Q_{j,.} ||_2^2 for all i, j in
%   [N]. Note that compared to Li et al., we have changed variables C=Z, Y=D,
%   and parameters gamma*alpha=Gamma, gamma=gamma0.
%
%   [groups, Y, history] = s3lr(X, Omega, n, params)
%
%   Args:
%     X, Omega: D x N data matrix and observed entry mask
%     n: number of clusters
%     params: struct containing the following problem parameters.
%       Y0, C0: initial completion, self-expression.
%       gamma: structured sparse self-expressive penalty [default: 0.02]
%       alpha: structured sparse tradeoff parameter [default: 1]
%       lambda: sparse corruption penalty [default: 20]
%       ladmm_maxit, ladmm_tol: LADMM sub-problem max iters and stopping
%         tolerance [default: 1e6, 1e-5]
%       ladmm_mu, ladmm_rho, ladmm_mu_max: ADMM penalty parameter, increasing
%         rate, max value [default: (1.25 / || X ||_2), 1.1, 1e4]
%       maxit: maximum outer iterations [default: 50]
%       prtlevel, loglevel: [default: 0, 1]
%
%   Returns:
%     groups: N x 1 cluster assignment
%     Y: D x N data completion
%     history: struct containing the following information
%       iter, status, conv_cond: number of iterations, termination status,
%         convergence condition at termination.
%       rtime: total runtime in seconds
tstart = tic;
[~, N] = size(X);
Omega = logical(Omega);
Omegac = ~Omega;
X(Omegac) = 0;

if nargin < 4
  params = struct;
end
fields = {'init', 'gamma', 'alpha', 'lambda', 'ladmm_maxit', 'ladmm_tol', ...
    'ladmm_mu', 'ladmm_rho', 'ladmm_mu_max', 'maxit', 'tol', ...
    'prtlevel', 'loglevel'};
defaults = {'zf', 0.02, 1, 20, 1e6, 1e-5, NaN, 1.1, 1e4, 50, 0, 1};
params = set_default_params(params, fields, defaults);
% use of normest follows Li, although I don't think much time is saved vs
% just norm(X).
if isnan(params.ladmm_mu); params.ladmm_mu = 1.25 / normest(X, 0.1); end

% initialize C, Y, Theta, E
if isempty(params.C0); C = zeros(N); else; C = params.C0; end
if isempty(params.Y0); Y = X; else; Y = params.Y0; end
[Theta, E] = deal(zeros(N));

% set fixed ladmm parameters
tau = Inf;
relax = 1;
affine = 0;
opt_params = struct('tol', params.ladmm_tol, 'maxIter', params.ladmm_maxit, ...
    'mu', params.ladmm_mu, 'rho', params.ladmm_rho, 'mu_max', ...
    params.ladmm_mu_max, 'norm_sr', '1', 'norm_mc', '1');

history.status = 1;
for kk=1:params.maxit
  Y_old = Y;
  Theta_old = Theta;

  [C, Y, E] = SMC_StrSR(X, Omega, Theta, params.lambda, ...
      params.gamma*params.alpha, params.gamma, tau, opt_params, Y, C, ...
      relax, affine);

  % Li used a different, seemingly more complicated implementation for the
  % spectral clustering step. here we just use the same implementation used in
  % other algorithms.
  A = build_affinity(C);
  groups = SpectralClustering(A, n, 'Eig_Solver', 'eigs');
  Theta = flip_co_occur(groups);

  fc_norm = norm(Theta .* C, 'fro');
  Theta_update = infnorm(Theta - Theta_old);
  % this could be done just over unobserved entries, but following Li.
  Y_update = norm(Y - Y_old, 'fro') / norm(Y, 'fro');

  % compute objective only if needed, since very expensive (n svd's).
  if max(params.prtlevel, params.loglevel) > 0
    obj = s3lr_objective(C, Y, E, Theta, groups, params.lambda, params.gamma, ...
        params.alpha);
  end

  if params.prtlevel > 0
    fprintf('k=%d, obj=%.2e, fc_norm=%.2e, Theta_updt=%.0f, Y_updt=%.2e \n', ...
        kk, obj, fc_norm, Theta_update, Y_update);
  end
  if params.loglevel > 0
    history.obj(kk) = obj;
    history.fc_norm(kk) = fc_norm;
    history.Theta_update(kk) = Theta_update;
    history.Y_update(kk) = Y_update;
  end

  % note that the paper describes a different stopping condition that should be
  % much more strict (p. 7):
  %
  %   || Theta^k - Theta^{k-1} ||_\infty < 1 and
  %   || Y^k - Y^{k-1} ||_F / || Y^k ||_F < delta
  %
  % but here we follow Li's code.
  if min(fc_norm, Y_update) < params.tol
    history.status = 0;
    break
  end
end

history.C = C;
history.E = E;

if params.loglevel <= 0
  history.obj = s3lr_objective(C, Y, E, Theta, groups, params.lambda, ...
      params.gamma, params.alpha);
  history.fc_norm = fc_norm;
  history.Theta_update = Theta_update;
  history.Y_update = Y_update;
end

history.conv_cond = min(fc_norm, Y_update);
history.iter = kk; history.rtime = toc(tstart);
end


function Theta = flip_co_occur(groups)
% flip_co_occur    compute flipped co-occurrence matrix Theta from groups.
groups = reshape(groups, [], 1);
N = size(groups, 1);
Theta = repmat(groups, [1 N]) ~= repmat(groups, [N 1]);
end


function obj = s3lr_objective(C, Y, E, Theta, groups, lambda, gamma, alpha)
% sl3r_objective    evaluate S3LR objective.
slr_term = 0;
n = max(groups);
for ii=1:n
  slr_term = slr_term + sum(svd(Y(:, groups == ii)));
end

obj = slr_term + ...
    gamma * (alpha * sum(sum(abs(Theta .* C))) + sum(abs(C(:)))) + ...
    lambda * sum(abs(E(:)));
end
