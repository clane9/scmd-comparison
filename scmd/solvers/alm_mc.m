function [Y, history] = alm_mc(X, Omega, Y0, params)
% alm_mc    Inexact ALM algorithm for matrix completion from Lin et al 2009.
%   Solves formulation
%
%   min_Y ||Y||_* s.t. X = Y + E, P_Omega(E) = 0
%
%   [Y, history] = alm_mc(X, Omega, Y0, params)
%
%   Args:
%     X: D x N incomplete data matrix
%     Omega: D x N logical indicator of observed entries
%     Y0: D x N initial guess for low-rank completion.
%     params: Struct containing problem parameters
%       mu: ALM penalty parameter [default: 1/||X||_{2,1}]
%       alpha: rate for increasing mu after each iteration [default: 1.1]
%       mu_max: maximum mu [default: 1e4]
%       maxit: maximum iterations [default: 500]
%       tol: stopping tolerance [default: 1e-5]
%       prtlevel: 1=basic per-iteration output [default: 0]
%       loglevel: 0=basic summary info, 1=detailed per-iteration info
%         [default: 0]
%
%   Returns:
%     Y: D x N low-rank completion.
%     history: struct containing the following information
%       obj, feas, rnk: objective, feasibility, rank; per iteration if loglevel > 0.
%       iter, status, conv_cond: number of iterations, termination status,
%         convergence condition at termination.
%       rtime: total runtime in seconds
%
%   References:
%     Lin, Z., Chen, M., & Ma, Y. (2010). The augmented lagrange multiplier
%     method for exact recovery of corrupted low-rank matrices. arXiv preprint
%     arXiv:1009.5055.
%
%     Chen, Y., Xu, H., Caramanis, C., & Sanghavi, S. (2011). Robust matrix
%     completion and corrupted columns. In Proceedings of the 28th
%     International Conference on Machine Learning (ICML-11) (pp. 873-880).
tstart = tic;
[D, N] = size(X);
Omega = logical(Omega);
Omegac = ~Omega;
X(Omegac) = 0;

if nargin < 4; params = struct; end
fields = {'mu', 'alpha', 'mu_max', 'maxit', 'tol', 'prtlevel', 'loglevel'};
defaults = {NaN, 1.1, 1e4, 500, 1e-5, 0, 0};
params = set_default_params(params, fields, defaults);

if isempty(Y0); Y = X; else; Y = Y0; end
U = zeros(D, N);

% set default 1/mu = 0.2 || Y ||_2 to get some reasonable but not excessive
% singular value thresholding. if too strong, then get no benefit of warm
% start.
if isnan(params.mu); mu = 1/(0.2 * norm(Y)); else; mu = params.mu; end

relthr = max(infnorm(X(Omega)), 1e-3);
history.status = 1;
for kk=1:params.maxit
  E = Omegac.*(X - Y + U);
  [Y, ~, sthr] = prox_nuc(X - E + U, 1/mu);
  Con = X - Y - E;
  U = U + Con;
  feas = infnorm(Con) / relthr;
  obj = sum(sthr);
  rnk = sum(sthr > 1e-3*sthr(1));

  if params.prtlevel > 0
    fprintf('k=%d, mu=%.2e, obj=%.2e, rnk=%d, feas=%.2e \n', kk, mu, obj, rnk, feas);
  end
  if params.loglevel > 0
    history.feas(kk) = feas;
    history.obj(kk) = obj;
    history.rnk(kk) = rnk;
  end
  if feas < params.tol
    history.status = 0;
    break
  end

  mu = min(params.mu_max, params.alpha * mu);
end

% ensure constraint satisfied exactly
Y(Omega) = X(Omega);

if params.loglevel <= 0
  history.feas = feas;
  history.obj = obj;
  history.rnk = rnk;
end
history.conv_cond = feas; history.iter = kk; history.rtime = toc(tstart);
end
