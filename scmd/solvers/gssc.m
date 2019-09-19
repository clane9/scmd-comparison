function [groups, Y, history] = gssc(X, Omega, n, params)
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
%   [groups, Y, history] = gssc(X, Omega, n, params)
%
%   Args:
%     X: D x N incomplete data matrix
%     Omega: D x N logical indicator of observed entries
%     n: number of clusters
%     params: struct containing the following problem parameters.
%       r: subspace dimension (required)
%       init: initialization method ('random', 'pzf-ensc+lrmc') [default:
%         pzf-ensc+lrmc]
%       optim: optimization method ('cvx', 'apg') [default: 'apg']
%       squared: use Frobenius squared vs unsquared loss. D.P.A's paper says
%         squared, but code uses unsquared. problems are equivalent up to
%         choice of lambda [default: 1].
%       lr_mode: solve LR-GSSC formulation [default: 0].
%       lrmc_final: compute final completion by group-wise LRMC [default: 0].
%       lambda: group sparse V penalty parameter [default: 1e-3].
%       gamma: column sparse U penalty parameter for LR-GSSC [default: 1].
%       maxit: maximum iterations [default: 100]
%       tol: stopping tolerance [default: 1e-5]
%       prtlevel, loglevel: [default: 0, 0]
%
%   Returns:
%     groups: N x 1 cluster assignment
%     Y: D x N data completion
%     history: struct containing the following information
%       U, V: final iterates, if loglevel > 0.
%       obj, obj_update, U_update: either per iteration or at termination.
%       init_history: history from initialization
%       sp_history: alt min sub-problem history
%       mc_history: history from final group lrmc completion.
%       iter, status, conv_cond: number of iterations, termination status,
%         convergence condition at termination.
%       rtime: total runtime in seconds
tstart = tic;
[D, N] = size(X);
Omega = logical(Omega);
Omegac = ~Omega;
X(Omegac) = 0;

if ~any(Omega(:))
  error('ERROR: no observed entries given.')
end

if nargin < 4; params = struct; end
fields = {'r', 'init', 'optim' 'squared', 'lr_mode', 'lrmc_final', ...
    'lambda', 'gamma', 'maxit', 'tol', 'prtlevel', 'loglevel'};
defaults = {NaN, 'pzf-ensc+lrmc', 'apg', 0, 0, 0, 1e-3, 1, 100, 1e-5, 0, 0};
params = set_default_params(params, fields, defaults);
if isnan(params.r); error('ERROR: subspace dimension r is required.'); end
r = params.r;

if strcmpi(params.optim, 'cvx')
  gssc_Umin = @(U, V) gssc_Umin_cvx(X, Omega, U, V, params.r, ...
      params.squared, params.lr_mode, params.gamma);
  gssc_Vmin = @(U, V) gssc_Vmin_cvx(X, Omega, U, V, params.r, ...
      params.squared, params.lambda);
else
  apg_params = struct('accel', 1, 'maxit', 500, 'tol', 1e-6, ...
      'prtlevel', params.prtlevel-1, 'loglevel', params.loglevel-1);
  gssc_Umin = @(U, V) gssc_Umin_apg(X, Omega, U, V, ...
      params.lr_mode, params.gamma, apg_params);
  gssc_Vmin = @(U, V) gssc_Vmin_apg(X, Omega, U, V, params.r, ...
      params.lambda, apg_params);
end

function obj = gssc_objective(U, V)
  obj = sum(sum((Omega.*(U*V' - X)).^2));
  if ~params.squared
    obj = sqrt(obj);
  end
  Vflat = reshape(V', params.r, []);
  obj = obj + params.lambda * sum(sqrt(sum(Vflat.^2)));
  if params.lr_mode
    obj = obj + params.gamma * sum(sqrt(sum(U.^2)));
  end
end

% if Y0, groups0 both provided, initialize U_i by svd. otherwise initialize
% randomly.
U = (1/sqrt(D)) * randn(D, r*n);
if strcmpi(params.init, 'pzf-ensc+lrmc')
  init_params = struct('init', 'zf', 'sc_method', 'ensc', 'ensc_pzf', 1, ...
      'mc_method', 'lrmc', 'maxit', 1, 'prtlevel', params.prtlevel-1, ...
      'loglevel', params.loglevel-1);
  [groups, Y, history.init_history] = alt_sc_mc(X, Omega, n, init_params);
  for ii=1:n
    maski = groups == ii;
    % we initialize randomly and then overwrite to ensure unused columns are
    % non-zero.
    ri = min(sum(maski), r);
    if ri > 0
      startidx = r*(ii-1) + 1;
      [U(:, startidx:(startidx+ri-1)), ~, ~] = svds(Y(:, maski), ri);
    end
  end
end
U = (1/norm(U, 'fro')) * U;
V = gssc_Vmin(U, 1e-3*randn(N, r*n));

obj = gssc_objective(U, V);
history.status = 1;
for kk=1:params.maxit
  objprev = obj;
  Uprev = U;

  [U, history.sp_history{kk, 1}] = gssc_Umin(U, V);
  [V, history.sp_history{kk, 2}] = gssc_Vmin(U, V);
  obj = gssc_objective(U, V);

  obj_update = (objprev - obj)/(objprev + eps);
  U_update = infnorm(Uprev - U) / max(infnorm(U), 1e-3);

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

if params.loglevel > 0
  history.U = U; history.V = V;
else
  history.U = []; history.V = [];
  history.obj = obj;
  history.obj_update = obj_update;
  history.U_update = U_update;
end
history.conv_cond = obj_update; history.iter = kk;
history.rtime = toc(tstart);
end


function [V, history] = gssc_Vmin_cvx(X, Omega, U, ~, r, squared, lambda)
% gssc_Vmin_cvx   Solve V minimization using CVX 
N = size(X, 2);
n = size(U, 2) / r;
cvx_begin quiet
cvx_precision low
variable V(N,n*r);
% squared and un-squared forms share the same solution path, but solutions for
% particular lambda will be different.
if squared
  func = sum(sum_square(Omega.*(U*V'-X)));
else
  func = norm(Omega.*(U*V'-X),'fro');
end
pnlty = lambda * sum(norms(reshape(V',[r,N*n]),2,1));
minimize(func + pnlty);
cvx_end
history = struct;
end


function [V, history] = gssc_Vmin_apg(X, Omega, U, V0, r, lambda, apg_params)
% gssc_Vmin_apg   Solve V minimization using accelerated proximal gradient
N = size(V0, 1);
function [f, G] = vmin_ffun(V)
  Res = Omega.*(U*V' - X);
  f = sum(Res(:).^2);
  if nargout > 1
    G = 2*Res'*U;
  end
end
function [theta, Z] = vmin_rfun(V, alpha)
  Vflat = reshape(V', r, []);
  theta = lambda * sum(sqrt(sum(Vflat).^2));
  if nargout > 1
    Zflat = prox_L21(Vflat, lambda*alpha);
    Z = reshape(Zflat, [], N)';
  end
end
[V, history] = apg(V0, @vmin_ffun, @vmin_rfun, apg_params);
end


function [U, history] = gssc_Umin_cvx(X, Omega, ~, V, r, squared, lr_mode, ...
    gamma)
% gssc_Umin_cvx   Solve U minimization using CVX 
D = size(X, 1);
n = size(V, 2) / r;
cvx_begin quiet
cvx_precision low
variable U(D,n*r);
if squared
  func = sum(sum_square(Omega.*(U*V'-X)));
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
history = struct;
end


function [U, history] = gssc_Umin_apg(X, Omega, U0, V, lr_mode, gamma, ...
    apg_params)
% gssc_Umin_apg   Solve U minimization using accelerated proximal gradient
function [f, G] = umin_ffun(U)
  Res = Omega.*(U*V' - X);
  f = sum(Res(:).^2);
  if nargout > 1
    G = 2*Res*V;
  end
end
function [theta, Z] = umin_rfun(U, alpha)
  if lr_mode
    theta = gamma * sum(sqrt(sum(U.^2)));
  else
    theta = 0;
  end
  if nargout > 1
    if lr_mode
      Z = prox_L21(U, gamma*alpha);
    else
      Z = (1/(max(1, norm(U, 'fro'))+eps))*U;
    end
  end
end
[U, history] = apg(U0, @umin_ffun, @umin_rfun, apg_params);
end


function groups = assign_groups(V, n, r)
% assign_groups   assign data points to group with largest coefficients.
N = size(V, 1);
[~, groups] = max(sum(reshape(V', [r n N]).^2, 1));
groups = squeeze(groups);
end
