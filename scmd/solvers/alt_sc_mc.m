function [groups, Y, history] = alt_sc_mc(X, Omega, n, params)
% alt_sc_mc   Alternate between subspace clustering and matrix completion.
%
%   [groups, Y, history] = alt_sc_mc(X, Omega, n, params)
%
%   Args:
%     X: D x N incomplete data matrix
%     Omega: D x N logical indicator of observed entries
%     n: number of clusters
%     params: struct containing the following problem parameters. various
%       combinations correspond to different algorithms, e.g.
%       (LRMC|LADMC)+(EnSC|TSC), (Alt) (P)ZF-EnSC+LRMC, EnSC-SEMC.
%       init: initialization method ('zf', 'lrmc', 'ladmc', 'pzf-ensc+lrmc')
%         [default: 'zf']
%       sc_method: which sc method ('ensc', 'tsc') [default: 'ensc']
%       ensc_pzf: whether to discount self-expressive errors on unobserved
%         entries [default: 1]
%       ensc_lambda0: least-squares penalty relative to minimum lambda required
%         to guarantee non-zero self-expressions [default: 20]
%       ensc_gamma: elastic net tradeoff parameter [default: 0.9]
%       tsc_qfrac: number of TSC nearest neighbors [default: .05]
%       mc_method: which mc method ('lrmc', 'semc') [default: 'lrmc']
%       semc_eta: l2 squared regularization parameter in semc [default: 0]
%       lrmc_maxit: maximum iterations in alm_mc [default: 500]
%       lrmc_tol: stopping tolerance in alm_mc [default: 1e-5]
%       maxit: maximum iterations. can specify non-integer maxit (e.g. 0.5) to
%         terminate on clustering and skip final completion. [default: 5]
%       tol: stopping tolerance [default: 1e-5]
%       prtlevel, loglevel: values > 0 indicate more diagnostic
%         printing/logging [default: 0, 0]
%
%   Returns:
%     groups: N x 1 cluster assignment
%     Y: D x N data completion
%     history: struct containing the following information
%       C: N x N self-expression matrix, if loglevel > 0
%       init_history: history from initialization
%       sc_history: ensc subproblem histories
%       mc_history: mc subproblem histories, detailed per-iteration info if
%         loglevel > 1
%       groups_update: cluster assignment update, per iteration if loglevel > 0
%       Y_update: infnorm update to unobserved entries in Y, per iteration if
%         loglevel > 0.
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
iscomplete = all(Omega(:));

% parse params and minimal checking
if nargin < 4; params = struct; end
fields = {'init', 'sc_method', 'ensc_pzf', 'ensc_lambda0', 'ensc_gamma', ...
    'tsc_qfrac', 'mc_method', 'semc_eta', 'lrmc_maxit', 'lrmc_tol', ...
    'maxit', 'tol', 'prtlevel', 'loglevel'};
defaults = {'zf', 'ensc', 1, 20, 0.9, 0.05, 'lrmc', 0, 500, 1e-5, 1, 1e-5, ...
    0, 0};
params = set_default_params(params, fields, defaults);
params.tsc_q = max(3, ceil(params.tsc_qfrac * N/n));

semc_mode = strcmpi(params.mc_method, 'semc');
if strcmpi(params.sc_method, 'tsc') && semc_mode
  error('ERROR: TSC incompatible with SEMC.')
end

lrmc_params.maxit = params.lrmc_maxit;
lrmc_params.tol = params.lrmc_tol;
lrmc_params.prtlevel = params.prtlevel-1;
lrmc_params.loglevel = params.loglevel-1;

ensc_params.lambda0 = params.ensc_lambda0;
ensc_params.gamma = params.ensc_gamma;
% use a fixed min lambda for all data points, to guarantee all c_j non-zero
ensc_params.lambda_method = 'fixed';
ensc_params.normalize = 0;

if any(strcmpi(params.init, {'lrmc', 'ladmc'}))
  init_params = struct('maxit', 500, 'tol', 1e-5, 'prtlevel', ...
    params.prtlevel-1, 'loglevel', params.loglevel-1);
  if strcmpi(params.init, 'lrmc')
    [Y, history.init_history] = alm_mc(X, Omega, [], init_params);
  else
    [Y, history.init_history] = ladmc(X, Omega, [], init_params);
  end
elseif strcmpi(params.init, 'pzf-ensc+lrmc')
  % initialize by doing 1 iteration of pzf-ensc+lrmc. only potentially useful
  % when doing semc
  init_params = struct('init', 'zf', 'sc_method', 'ensc', 'ensc_pzf', 1, ...
      'mc_method', 'lrmc', 'maxit', 1, 'prtlevel', params.prtlevel-1, ...
      'loglevel', params.loglevel-1);
  [~, Y, history.init_history] = alt_sc_mc(X, Omega, n, init_params);
else
  Y = X;
  history.init_history = struct;
end

if params.ensc_pzf; W = Omega; else; W = []; end

groups = ones(N, 1); C = sparse(N, N);
relthr = max(infnorm(X(Omega)), 1e-3);
history.status = 1;
kk = 0;
while kk < params.maxit
  kk = kk + 1;

  % subspace clustering
  groups_prev = groups;
  C_prev = C;
  if strcmpi(params.sc_method, 'tsc')
    % store affinity as C also
    A = tsc(Y, params.tsc_q); C = A;
    history.sc_history{kk} = struct;
  else
    unitY = Y ./ repmat(sqrt(sum(Y.^2))+eps, [D 1]);
    [C, history.sc_history{kk}] = weight_ensc_spams(unitY, W, ensc_params);
    A = build_affinity(C);
  end
  groups = SpectralClustering(A, n);
  % use hungarian algorithm to measure change in clustering.
  groups_update = 1 - evalAccuracy(groups_prev, groups);
  C_update = full(infnorm(C - C_prev) / relthr);

  % matrix completion
  if kk > params.maxit
    % skip final completion for non-integer maxit. needed for algorithms
    % (LRMC|LADMC)+(EnSC|TSC).
    Y_update = NaN;
    history.mc_history{kk} = struct;
  elseif iscomplete
    Y = X; Y_update = 0;
    history.mc_history{kk} = struct;
  else
    Yunobs_prev = Y(Omegac);
    if semc_mode
      % semc formulation has no penalty on least-squares self-expression term, so
      % need to re-scale. note that this lambda might vary across iterations.
      eta = params.semc_eta / history.sc_history{kk}.lambda;
      [Y, history.mc_history{kk}] = semc(X, Omega, C, eta);
    else
      [Y, history.mc_history{kk}] = group_alm_mc(X, Omega, groups, [], ...
          lrmc_params);
    end
    Y_update = infnorm(Y(Omegac) - Yunobs_prev) / relthr;
  end

  if params.prtlevel > 0
    fprintf('k=%d, groups_updt=%.3f, C_updt=%.2e, Y_updt=%.2e \n', kk, ...
        groups_update, C_update, Y_update);
  end
  if params.loglevel > 0
    history.groups_update(kk) = groups_update;
    history.C_update(kk) = C_update;
    history.Y_update(kk) = Y_update;
  end

  % stop when clustering doesn't change if doing lrmc, or when C or Y
  % stop changing.
  conv_cond = min([max(semc_mode, groups_update) ~= 0, C_update, Y_update]);
  if conv_cond < params.tol
    history.status = 0;
    break
  elseif kk > params.maxit
    kk = kk - 0.5;
    break
  end
end

if params.loglevel > 0
  history.C = C;
end
if params.loglevel <= 0
  history.groups_update = groups_update;
  history.Y_update = Y_update;
end
history.conv_cond = conv_cond;
history.iter = kk; history.rtime = toc(tstart);
end
