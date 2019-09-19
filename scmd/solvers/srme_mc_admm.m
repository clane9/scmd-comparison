function [groups, Y, history] = srme_mc_admm(X, Omega, n, params)
% srme_mc_admm    subspace clustering and matrix completion by sparse
%   representation with missing entries and matrix completion (SRME-MC) (Fan &
%   Chow, 2017). Solves the following problem formulation by ADMM
%
%   min_{C, Y, E, L, A}  || C ||_1 + alpha || L ||_ell + lambda || E ||_{ell'}
%                  s.t.  Y = YA + E, P_Omega(Y - X) = 0, L = Y,
%                        diag(C) = 0, A = C
%
%   Note that compared to Fan & Chow, we have changed variables Y=X, L=Y, C=C,
%   A=A, E=E.
%
%   [groups, Y, history] = srme_mc_admm(X, Omega, n, params)
%
%   Args:
%     X: D x N incomplete data matrix
%     Omega: D x N logical indicator of observed entries
%     n: number of clusters
%     params: struct containing the following problem parameters.
%       init: initialization method ('zf', 'lrmc', 'ladmc', 'pzf-ensc+lrmc')
%         [default: 'zf']
%       L_ell: L norm ('frosquared', 'nuc') [default: 'nuc']
%       E_ell: E norm ('frosquared', 'L1') [default: 'frosquared']
%       alpha: L penalty parameter [default: 0.1]
%       lambda: E penalty [default: 10]
%       mu, rho, mu_max: ADMM penalty parameter, increasing
%         rate, max value [default: 5/||X||_2, 1.01, 1e10]
%       maxit: maximum admm iterations [default: 800]
%       tol: stopping tolerance [default: 1e-5]
%       prtlevel, loglevel: [default: 0, 0]
%
%   Returns:
%     groups: N x 1 cluster assignment
%     Y: D x N data completion
%     history: struct containing the following information
%       C: N x N self-expression matrix, if loglevel > 0
%       E: D x N corruption matrix, if loglevel > 0
%       init_history: history from initialization
%       obj, feas, update: objective, constraint norm, parameter update, per
%         iteration if loglevel > 0.
%       iter, status, conv_cond: number of iterations, termination status,
%         convergence condition at termination.
%       rtime: total runtime in seconds
%
%   References:
%     Fan, J., & Chow, T. W. (2017). Sparse subspace clustering for data with
%     missing entries and high-rank matrix completion. Neural Networks, 93,
%     36-44.
%
%   Written by Jicong Fan.
%
%   Rewritten by Connor Lane. Mostly edits for consistency and obsessive
%   changes to style, although update to Y (ie X) was incorrect and now fixed.
tstart = tic;
[D, N] = size(X);
Omega = logical(Omega);
Omegac = ~Omega;
X(Omegac) = 0;

if ~any(Omega(:))
  error('ERROR: no observed entries given.')
end

if nargin < 4; params = struct; end
fields = {'init', 'L_ell', 'E_ell', 'alpha', 'lambda', 'mu', 'rho', 'mu_max', ...
    'maxit', 'tol', 'prtlevel', 'loglevel'};
defaults = {'zf', 'nuc', 'frosquared', 0.1, 10, NaN, 1.01, 1e10, 800, 1e-5, ...
    0, 0};
params = set_default_params(params, fields, defaults);

if strcmpi(params.E_ell, 'L1')
  prox_E = @prox_L1; norm_E = @(E) (sum(abs(E(:))));
else
  prox_E = @prox_frosqr; norm_E = @(E) (sum(E(:).^2));
end
if strcmpi(params.L_ell, 'frosquared')
  prox_L = @prox_frosqr; norm_L = @(L) (sum(L(:).^2));
else
  prox_L = @prox_nuc; norm_L = @(L) (sum(svd(L)));
end
srme_objective = @(C, L, E) (sum(abs(C(:))) + ...
    params.alpha*norm_L(L) + params.lambda*norm_E(E));

% initialize C, Y, E, L, A
if any(strcmpi(params.init, {'lrmc', 'ladmc'}))
  init_params = struct('maxit', 500, 'tol', 1e-5, 'prtlevel', ...
      params.prtlevel-1, 'loglevel', params.loglevel-1);
  if strcmpi(params.init, 'lrmc')
    [Y, history.init_history] = alm_mc(X, Omega, [], init_params);
  else
    [Y, history.init_history] = ladmc(X, Omega, [], init_params);
  end
  C = zeros(N);
  mu0 = min(1/(0.2*norm(Y)), 10);
elseif strcmpi(params.init, 'pzf-ensc+lrmc')
  init_params = struct('init', 'zf', 'sc_method', 'ensc', 'ensc_pzf', 1, ...
      'mc_method', 'lrmc', 'maxit', 1, 'prtlevel', params.prtlevel-1, ...
      'loglevel', 1);
  [~, Y, history.init_history] = alt_sc_mc(X, Omega, n, init_params);
  C = full(history.init_history.C);
  mu0 = min(1/(0.2*infnorm(C)), 10);
else
  Y = X; C = zeros(N);
  history.init_history = struct;
  mu0 = min(1/(0.2*norm(X)), 10);
end
L = Y; A = C;
E = zeros(D, N);

Q1 = zeros(D, N); Q2 = zeros(N); Q3 = zeros(D, N);

SE_Res = Y - Y*A;
mu = params.mu;
if isnan(mu); mu = mu0; end

I = eye(N);
relthr = norm(X, 'fro');

history.status = 1;
for kk=1:params.maxit
  C_old = C; Y_old = Y; A_old = A; E_old = E;
  U1 = Q1/mu; U2 = Q2/mu; U3 = Q3/mu;
  
  % E = arg min  lambda || E ||_ell' + mu/2 || Y - YA - E + U ||_F^2
  E = prox_E(SE_Res + U1, params.lambda/mu);
  % A = arg min  mu/2 || Y - YA - E + U1 ||_F^2
  %            + mu/2 || A - C + U2 ||_F^2
  A = (Y'*Y + I) \ (Y'*(Y - E + U1) + (C - U2));
  % C = arg min  || C ||_1 + mu/2 || A - C + U2 ||_F^2
  %        s.t.  diag(C) = 0
  C = prox_L1(A + U2, 1/mu);
  C(1:N+1:end) = 0;
  % L = arg min  alpha || L ||_ell + mu/2 || L - Y + U3 ||_F^2
  L = prox_L(Y - U3, params.alpha/mu);
  % Y = arg min  mu/2 || Y - YA - E + U1 ||_F^2
  %            + mu/2 || L - Y + U3 ||_F^2
  %        s.t.  P_Omega(Y - X) = 0
  Y = lsqrmd(X, Omega, I - A, 1, E - U1, L + U3);

  % update multipliers
  SE_Res = Y - Y*A;
  Feas1 = SE_Res - E;
  Feas2 = A - C;
  Feas3 = L - Y;
  Q1 = Q1 + mu*Feas1;
  Q2 = Q2 + mu*Feas2;
  Q3 = Q3 + mu*Feas3;

  feas = max([norm(Feas1, 'fro') norm(Feas2, 'fro') ...
      norm(Feas3, 'fro')]) / relthr;
  update = max([norm(A - A_old, 'fro'), norm(C - C_old, 'fro'), ...
      norm(Y - Y_old, 'fro'), norm(E - E_old, 'fro')]) / relthr;

  % compute objective only when required
  if max(params.prtlevel, params.loglevel) > 0
    obj = srme_objective(C, L, E);
  end

  if params.prtlevel > 0
    fprintf('k=%d, mu=%.2e, obj=%.2e, feas=%.2e, updt=%.2e \n', kk, mu, ...
        obj, feas, update);
  end
  if params.loglevel > 0
    history.obj(kk) = obj;
    history.feas(kk) = feas;
    history.update(kk) = update;
  end
  
  if min(feas, update) < params.tol
    history.status = 0;
    break
  end

  mu = min(params.mu_max, params.rho*mu);
end
  
W = build_affinity(C);
groups = SpectralClustering(W, n, 'Eig_Solver', 'eigs');

if params.loglevel > 0
  history.C = C; history.E = E;
else
  [history.C, history.E] = deal([]);
  history.obj = srme_objective(C, L, E);
  history.feas = feas;
  history.update = update;
end

history.conv_cond = max(feas, update);
history.iter = kk; history.rtime = toc(tstart);
end
