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
%       maxit: maximum iterations [default: 500]
%       tol: stopping tolerance [default: 1e-5]
%       prtlevel: 1=basic per-iteration output [default: 0]
%       loglevel: 0=basic summary info, 1=detailed per-iteration info
%         [default: 0]
%
%   Returns:
%     Y: D x N low-rank completion.
%     history: struct containing the following information
%       feas: feasibility, per iteration if loglevel > 0.
%       obj: objective, per iteration if loglevel > 0.
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

if nargin < 3
  params = struct;
end
% set defaults
% see page 6 in (Chen et al., 2011) for mu/alpha default justification
% https://pdfs.semanticscholar.org/1556/6adc0e6312b72290c31f891bdb080c06d997.pdf
fields = {'mu', 'alpha', 'maxit', 'tol', 'prtlevel', 'loglevel'};
defaults = {1/sum(sqrt( sum(X.^2, 1) )), 1.1, 500, 1e-5, 0, 0};
params = set_default_params(params, fields, defaults);

if isempty(Y0); Y = X; else; Y = Y0; end
U = zeros(D, N);

mu = params.mu;
relthr = max(infnorm(X(Omega)), 1e-3);
history.status = 1;
for kk=1:params.maxit
  E = Omegac.*(X - Y + U);
  [Y, ~, sthr] = prox_nuc(X - E + U, 1/mu);
  Con = X - Y - E;
  U = U + Con;
  feas = infnorm(Con) / relthr;
  obj = sum(sthr);

  if params.prtlevel > 0
    fprintf('k=%d, obj=%.2e, feas=%.2e \n', kk, obj, feas);
  end
  if params.loglevel > 0
    history.feas(kk) = feas;
    history.obj(kk) = obj;
  end
  if feas < params.tol
    history.status = 0;
    break
  end

  mu = params.alpha * mu;
end

% ensure constraint satisfied exactly
Y(Omega) = X(Omega);

if params.loglevel <= 0
  history.feas = feas;
  history.obj = obj;
end
history.conv_cond = feas; history.iter = kk; history.rtime = toc(tstart);
end
