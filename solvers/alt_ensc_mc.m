function [groups, Y, C, history] = alt_ensc_mc(X, Omega, n, params)
% alt_ensc_mc   Alternate between elastic-net subspace clustering and matrix
%   completion, either by LRMC per group or using self-expression.
%
%   [groups, Y, C, history] = alt_ensc_mc(X, Omega, n, params)
%
%   Args:
%     X, Omega: D x N data matrix and observed entry mask
%     n: number of clusters
%     params: struct containing the following problem parameters.
%       Y_init: how to initialize completion ('zf', 'pzf', 'lrmc',
%         'pzf-ensc+lrmc') [default: 'pzf']
%       lambda0: least-squares penalty relative to minimum lambda required to
%         guarantee non-zero self-expressions [default: 20]
%       gamma: elastic net tradeoff parameter [default: 0.9]
%       mc_method: which mc method ('lrmc', 'semc') [default: 'lrmc']
%       semc_eta: l2 squared regularization parameter in semc [default: 0]
%       lrmc_maxit: maximum iterations in alm_mc [default: 500]
%       lrmc_tol: stopping tolerance in alm_mc [default: 1e-5]
%       maxit: maximum iterations [default: 5]
%       tol: stopping tolerance [default: 1e-5]
%       prtlevel, loglevel: [default: 0, 1]
%
%   Returns:
%     groups: N x 1 cluster assignment
%     Y: D x N data completion
%     C: N x N self-expression matrix
%     history: struct containing the following information
%       init_history: initialization subproblem history
%       ensc_history: ensc subproblem histories
%       mc_history: mc subproblem histories
%       update: infnorm update to unobserved entries in Y.
%       iter, status: number of iterations, termination status
%       rtime: total runtime in seconds
tstart = tic;
[D, N] = size(X);
Omega = logical(Omega);
Omegac = ~Omega;
X(Omegac) = 0;

if nargin < 4
  params = struct;
end
fields = {'Y_init', 'lambda0', 'gamma', 'mc_method', 'semc_eta', ...
    'lrmc_maxit', 'lrmc_tol', 'maxit', 'tol', 'prtlevel', 'loglevel'};
defaults = {'pzf', 20, 0.9, 'lrmc', 0, 500, 1e-5, 5, 1e-5, 0, 1};
params = set_default_params(params, fields, defaults);
[ensc_params, lrmc_params] = parse_subprob_params(params);

if strcmpi(params.Y_init, 'lrmc')
  [Y, history.init_history] = alm_mc(X, Omega, [], lrmc_params);
elseif strcmpi(params.Y_init, 'pzf-ensc+lrmc')
  % initialize by doing 1 iteration of pzf-ensc+lrmc
  % only potentially useful when doing semc
  init_params = params;
  init_params.Y_init = 'pzf';
  init_params.mc_method = 'lrmc';
  init_params.maxit = 1;
  init_params.prtlevel = 0;
  init_params.loglevel = 0;
  [~, Y, ~, history.init_history] = alt_ensc_mc(X, Omega, n, init_params);
else
  Y = X;
  history.init_history = struct;
end

if strcmpi(params.Y_init, 'pzf')
  W = Omega;
else
  W = [];
end

relthr = max(infnorm(X(Omega)), 1e-3);
history.status = 1;
for kk=1:params.maxit
  Ynorm = sqrt(sum(Y.^2));
  % correct norm bias due to zero-filled missing entries
  if kk == 1 && any(strcmpi(params.Y_init, {'pzf', 'zf'}))
    Ynorm = sqrt(D ./ (sum(Omega)+eps)) .* Ynorm;
  end
  unitY = Y ./ repmat(Ynorm+eps, [D 1]);

  [C, history.ensc_history{kk}] = weight_ensc_spams(unitY, W, ensc_params);
  A = build_affinity(C);
  groups = SpectralClustering(A, n, 'Eig_Solver', 'eigs');

  yunobs = Y(Omegac);
  if strcmpi(params.mc_method, 'semc')
    % semc formulation has no penalty on least-squares self-expression term, so
    % need to re-scale. note that this lambda might vary across iterations.
    eta = params.semc_eta / history.ensc_history{kk}.lambda;
    [Y, history.mc_history{kk}] = semc(X, Omega, C, eta);
  else
    [Y, history.mc_history{kk}] = group_lrmc(X, Omega, Y, groups, n, ...
        lrmc_params);
  end
  history.update(kk) = infnorm(Y(Omegac) - yunobs) / relthr;

  if history.update(kk) <= params.tol
    history.status = 0;
    break
  end

  W = [];
end
history.iter = kk; history.rtime = toc(tstart);
end


function [ensc_params, lrmc_params] = parse_subprob_params(params)
% parse_subprob_params    Parse parameters for sub-problems.
lrmc_params.maxit = params.lrmc_maxit;
lrmc_params.tol = params.lrmc_tol;
lrmc_params.prtlevel = params.prtlevel - 1;
lrmc_params.loglevel = params.loglevel - 1;

ensc_params.lambda0 = params.lambda0;
ensc_params.gamma = params.gamma;
ensc_params.lambda_method = 'fixed';
ensc_params.normalize = 0;
end


function [Y, history] = group_lrmc(X, Omega, Y0, groups, n, lrmc_params)
% group_lrmc    Do low-rank matrix completion on each group.
Y = zeros(size(X));
for ii=1:n
  Idx = find(groups == ii);
  [Y(:, Idx), grp_history] = alm_mc(X(:, Idx), Omega(:, Idx), Y0(:, Idx), ...
      lrmc_params);

  history.iter(ii) = grp_history.iter;
  history.status(ii) = grp_history.status;
  history.feas(ii) = grp_history.feas(end);
  history.rtime(ii) = grp_history.rtime;
end
end
