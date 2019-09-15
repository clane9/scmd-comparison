function [groups, Y, history] = gssc(X, Omega, n, r, params)
% gssc    Run (LR-)GSSC for joint subspace clustering and completion. All
%   substance of GSSC algorithm from D. Pimentel. Solves the formulation
%
%   min_{U, V} || P_{\Omega}(X - U V^T) ||_F^2
%              + lambda \sum_{j,k=1}^{N,n} || v_{j k} ||_2
%              + \Theta( U )
%
%   where v_{j k} in R^r and 
%
%   \Theta_{GSSC}(U) = \delta_{|| U ||_F <= 1}(U)
%   \Theta_{LR-GSSC}(U) = gamma || U ||_{2,1}
%
%   [groups, Y, history] = gssc(X, Omega, n, r, params)
%
%   Args:
%     X: D x N incomplete data matrix
%     Omega: D x N logical indicator of observed entries
%     n: number of clusters
%     r: subspace dimension
%     params: struct containing the following problem parameters.
%       squared: use Frobenius squared vs unsquared loss. D.P.A's paper says
%         squared, but code uses unsquared. problems are equivalent up to
%         choice of lambda [default: 0].
%       lr_mode: solve LR-GSSC formulation [default: 0].
%       lrmc_final: compute final completion by group-wise LRMC [default: 1].
%       lambda: group sparse V penalty parameter [default: 1e-3].
%       gamma: column sparse U penalty parameter for LR-GSSC [default: 1e-3].
%       maxit: maximum iterations [default: 100]
%       tol: stopping tolerance [default: 1e-3]
%       prtlevel, loglevel: [default: 0, 1]
%
%   Returns:
%     groups: N x 1 cluster assignment
%     Y: D x N data completion
%     history: struct containing the following information
%       init_history: history from pzf-ensc+lrmc initialization.
%       obj, obj_update, U_update: either per iteration or at termination.
%       mc_history: history from group lrmc completion.
%       iter, status, conv_cond: number of iterations, termination status,
%         convergence condition at termination.
%       rtime: total runtime in seconds
tstart = tic;
[D, ~] = size(X);
Omega = logical(Omega);
Omegac = ~Omega;
X(Omegac) = 0;

if nargin < 5
  params = struct;
end
fields = {'squared', 'lr_mode', 'lrmc_final', 'lambda', 'gamma', ...
    'maxit', 'tol', 'prtlevel', 'loglevel'};
defaults = {0, 0, 1, 1e-3, 1e-3, 100, 1e-3, 0, 1};
params = set_default_params(params, fields, defaults);

% initialize U by first finding clustering and completion with pzf-ssc+lrmc.
% then for each group i initialize U_i by svd.
init_params = struct('init', 'zf', 'pzf', 1, 'maxit', 1, 'mc_method', ...
    'lrmc', 'lrmc_maxit', 500, 'lrmc_tol', 1e-5, 'prtlevel', 0, ...
    'loglevel', 0);
[groups, Y, history.init_history] = alt_ensc_mc(X, Omega, n, init_params);
% any unused columns will be initialized randomly
U = (1/sqrt(D)) * randn(D, r*n);
for ii=1:n
  maski = groups == ii;
  ri = min(sum(maski), r);
  if ri > 0
    startidx = r*(ii-1) + 1;
    [U(:, startidx:(startidx+ri)), ~, ~] = svds(Y(:, maski), ri);
  end
end
U = (1/norm(U, 'fro')) * U;

objprev = Inf;
Uprev = U;
history.status = 1;
for kk=1:params.maxit
  V = gssc_Vmin(X, Omega, U, n, r, params.squared, params.lambda);
  U = gssc_Umin(X, Omega, V, n, r, params.squared, params.lr_mode, ...
      params.gamma);
  obj = gssc_objective(X, Omega, U, V, n, r, params.squared, params.lr_mode, ...
      params.lambda, params.gamma);

  obj_update = objprev - obj;
  U_update = infnorm(Uprev - U) / max(infnorm(U), 1e-3);
  objprev = obj;

  if params.prtlevel > 0
    fprintf('k=%d, obj=%.2e, obj_updt=%.2e, U_updt=%.2e \n', kk, obj, ...
        obj_update, U_update);
  end
  if params.loglevel > 0
    history.obj(kk) = obj;
    history.obj_update(kk) = obj_update;
    history.U_update(kk) = U_update;
  end
  if obj_update < params.tol
    history.status = 0;
    break
  end
end

groups = assign_groups(V, n, r);

if params.lrmc_final
  lrmc_params = struct('maxit', 500, 'tol', 1e-5, 'prtlevel', 0, ...
      'loglevel', 0);
  [Y, history.mc_history] = group_alm_mc(X, Omega, groups, [], lrmc_params);
else
  % note that this completion should experience some shrinkage compared to
  % exact lrmc.
  Y = U*V';
  history.mc_history = struct;
end

if params.loglevel <= 0
  history.obj = obj;
  history.obj_update = obj_update;
  history.U_update = U_update;
end
history.conv_cond = obj_update; history.iter = kk;
history.rtime = toc(tstart);
end


function obj = gssc_objective(X, Omega, U, V, n, r, squared, lr_mode, ...
    lambda, gamma)
% gssc_objective    evaluate (LR-)GSSC objective.
N = size(X, 2);
obj = sum(sum((Omega.*(U*V' - X)).^2));
if ~squared
  obj = sqrt(obj);
end
obj = obj + lambda * sum(sqrt(sum(reshape(V',[r,N*n]).^2)));
if lr_mode
  obj = obj + gamma * sum(sqrt(sum(U.^2)));
end
end


function V = gssc_Vmin(X, Omega, U, n, r, squared, lambda)
% gssc_Vmin   minimize wrt V with U fixed.
N = size(X, 2);
cvx_begin quiet
cvx_precision low
variable V(N,n*r);
% squared and un-squared forms share the same solution path, but solutions for
% particular lambda will be different.
if squared
  func = sum_square(Omega.*(U*V'-X));
else
  func = norm(Omega.*(U*V'-X),'fro');
end
pnlty = lambda * sum(norms(reshape(V',[r,N*n]),2,1));
minimize(func + pnlty);
cvx_end
end


function U = gssc_Umin(X, Omega, V, n, r, squared, lr_mode, gamma)
% gssc_Umin   minimize wrt U with V fixed.
D = size(X, 1);
cvx_begin quiet
cvx_precision low
variable U(D,n*r);
if squared
  func = sum_square(Omega.*(U*V'-X));
else
  func = norm(Omega.*(U*V'-X),'fro');
end
if lr_mode
  pnlty = gamma * sum(norms(U, 2, 1));
  minimize(func + pnlty);
else
  minimize(func);
  norm(U,'fro')<=1;
end
cvx_end
end


function groups = assign_groups(V, n, r)
% assign_groups   assign data points to group with largest coefficients.
N = size(V, 1);
[~, groups] = max(sum(reshape(V', [r n N]).^2, 1));
groups = squeeze(groups);
end
