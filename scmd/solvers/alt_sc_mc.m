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
%       Y0: D x N initial completion guess [default: X]
%       sc_method: which sc method ('ensc', 'tsc') [default: 'ensc']
%       ensc_pzf: whether to discount self-expressive errors on unobserved
%         entries [default: 1]
%       ensc_lambda0: least-squares penalty relative to minimum lambda required
%         to guarantee non-zero self-expressions [default: 20]
%       ensc_gamma: elastic net tradeoff parameter [default: 0.9]
%       tsc_q: number of TSC nearest neighbors [default: max(3, N/(20 n))]
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

% parse params and minimal checking
if nargin < 4
  params = struct;
end
fields = {'Y0', 'sc_method', 'ensc_pzf', 'ensc_lambda0', 'ensc_gamma', ...
    'tsc_q', 'mc_method', 'semc_eta', 'lrmc_maxit', 'lrmc_tol', ...
    'maxit', 'tol', 'prtlevel', 'loglevel'};
defaults = {[], 'ensc', 1, 20, 0.9, NaN, 'lrmc', 0, 500, 1e-5, 1, 1e-5, 0, 0};
params = set_default_params(params, fields, defaults);
if isnan(params.tsc_q); params.tsc_q = max(3, ceil(N/(20*n))); end
if strcmpi(params.sc_method, 'tsc') && strcmpi(params.mc_method, 'semc')
  error('ERROR: TSC incompatible with SEMC.')
end

lrmc_params.maxit = params.lrmc_maxit;
lrmc_params.tol = params.lrmc_tol;
lrmc_params.prtlevel = params.prtlevel - 1;
lrmc_params.loglevel = params.loglevel - 1;

ensc_params.lambda0 = params.ensc_lambda0;
ensc_params.gamma = params.ensc_gamma;
% use a fixed min lambda for all data points, to guarantee all c_j non-zero
ensc_params.lambda_method = 'fixed';
ensc_params.normalize = 0;

if isempty(params.Y0); Y = X; else; Y = params.Y0; end
if params.ensc_pzf; W = Omega; else; W = []; end

groups = ones(N, 1);
relthr = max(infnorm(X(Omega)), 1e-3);
history.status = 1;
kk = 0;
while kk < params.maxit
  kk = kk + 1;

  % subspace clustering
  groups_prev = groups;
  if strcmpi(params.sc_method, 'tsc')
    % store affinity as C also
    A = tsc(Y, params.tsc_q); C = A;
    history.sc_history{kk} = struct;
  else
    unitY = Y ./ repmat(sqrt(sum(Y.^2))+eps, [D 1]);
    [C, history.sc_history{kk}] = weight_ensc_spams(unitY, W, ensc_params);
    A = build_affinity(C);
  end
  groups = SpectralClustering(A, n, 'Eig_Solver', 'eigs');
  % use hungarian algorithm to measure change in clustering.
  groups_update = 1 - evalAccuracy(groups_prev, groups);

  % skip final completion for non-integer maxit. needed for algorithms
  % (LRMC|LADMC)+(EnSC|TSC).
  if kk > params.maxit
    kk = kk + 0.5;
    break
  end

  % matrix completion
  Yunobs_prev = Y(Omegac);
  if strcmpi(params.mc_method, 'semc')
    % semc formulation has no penalty on least-squares self-expression term, so
    % need to re-scale. note that this lambda might vary across iterations.
    eta = params.semc_eta / history.sc_history{kk}.lambda;
    [Y, history.mc_history{kk}] = semc(X, Omega, C, eta);
  else
    [Y, history.mc_history{kk}] = group_alm_mc(X, Omega, groups, Y, ...
        lrmc_params);
  end
  Y_update = infnorm(Y(Omegac) - Yunobs_prev) / relthr;

  if params.prtlevel > 0
    fprintf('k=%d, groups_updt=%.3f, Y_updt=%.2e \n', kk, ...
        groups_update, Y_update);
  end
  if params.loglevel > 0
    history.groups_update(kk) = groups_update;
    history.Y_update(kk) = Y_update;
  end

  if Y_update < params.tol || groups_update == 0
    history.status = 0;
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
history.conv_cond = min(Y_update, groups_update);
history.iter = kk; history.rtime = toc(tstart);
end
