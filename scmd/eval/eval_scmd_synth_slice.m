function [ell_thr, cluster_errs, ...
    summary, histories] = eval_scmd_synth_slice(n, d, D, Ng, sigma, ...
    method_name, solver, params, opts)
% eval_scmd_synth_slice   Evaluate one "slice" of synthetic SCMD phase
%   transition plot by doing a bisection search over the number of missing
%   entries ell.
%
%   [ell_thr, cluster_errs, ...
%       summary, histories] = eval_scmd_synth_slice(n, d, D, Ng, sigma, ...
%       method_name, solver, params, opts)
if nargin < 9; opts = struct; end
fields = {'target_err', 'bisect_maxit', 'center_window', 'seed_start', ...
    'ntrial', 'paridx', 'out_summary', 'out_mat', 'prtlevel'};
defaults = {0.05, 10, 5, 1e5, 20, 0, 0, '', '', 0};
opts = set_default_params(opts, fields, defaults);
opts.target_err = max(opts.target_err, 5/(n*Ng));
trial_opts = struct('paridx', opts.paridx, 'out_summary', opts.out_summary, ...
    'quiet', 1);

% overwrite output summary and write header
if ~isempty(opts.out_summary)
  fid = fopen(opts.out_summary, 'w');
  fprintf(fid, ['n,d,D,Ng,sigma,ell,seed,method.name,paridx,cluster.err,' ...
      'comp.err,iter,status,conv.cond,rtime\n']);
  fclose(fid);
end

ell_thr = NaN; center_idx = NaN;
[summary, histories] = deal(cell(0));
ME = [];

% will consider complete range of ell values, but should only evaluate a narrow
% sub-interval, even for very large D
ells = d:D; nell = length(ells);
cluster_errs = NaN(nell, 1);
seed = opts.seed_start;

try
  % first compute errors at extremes
  for idx=[1 nell]
    [cluster_errs(idx), seed, status, ...
        summary, histories] = eval_scmd_synth_run(n, d, D, Ng, sigma, ...
        ells(idx), seed, method_name, solver, params, opts.ntrial, ...
        summary, histories, trial_opts); check_status(status);
  end

  % set target error relative to that observed for complete data
  opts.target_err = 1 - (1-opts.target_err)*(1-cluster_errs(nell));

  % find "center index" corresponding to largest ell s.t. error <= target for all
  % ell' > ell, using bisection.
  low_idx = 1; high_idx = nell;
  for kk=1:opts.bisect_maxit
    tic;
    center_idx = floor(0.5*(low_idx + high_idx));
    ell = ells(center_idx);

    [cluster_err, seed, status, ...
        summary, histories] = eval_scmd_synth_run(n, d, D, Ng, sigma, ...
        ell, seed, method_name, solver, params, opts.ntrial, ...
        summary, histories, trial_opts); check_status(status);
    cluster_errs(center_idx) = cluster_err;

    if opts.prtlevel > 0
      fprintf('k=%d, low=%d, idx=%d, hi=%d, err=%.4f, rt=%.3f \n', kk, low_idx, ...
          center_idx, high_idx, cluster_err, toc);
    end

    if cluster_err <= opts.target_err
      high_idx = center_idx;
    else
      low_idx = center_idx;
    end

    % bisection convergence
    if high_idx == low_idx + 1
      center_idx = low_idx;
      break
    end
  end

  % now fill in any gaps within narrow window arround the center
  scale_window = max(opts.center_window, ceil(d/5));
  Idx = floor(linspace(center_idx-scale_window, center_idx+scale_window, ...
      2*opts.center_window+1));
  % shift to fit in range
  if Idx(1) < 1
    Idx = Idx + (1-Idx(1));
  elseif Idx(end) > nell
    Idx = Idx - (Idx(end) - nell);
  end
  % possibly truncate if window is too large, and evaluate previously untested
  Idx = Idx(Idx >= 1 & Idx <= nell);
  for idx=Idx(isnan(cluster_errs(Idx)))
    tic;
    kk = kk + 1;
    [cluster_errs(idx), seed, status, ...
        summary, histories] = eval_scmd_synth_run(n, d, D, Ng, sigma, ...
        ells(idx), seed, method_name, solver, params, opts.ntrial, ...
        summary, histories, trial_opts); check_status(status);
    if opts.prtlevel > 0
      fprintf('k=%d, idx=%d, err=%.4f, rt=%.3f \n', kk, idx, ...
          cluster_errs(idx), toc);
    end
  end

  % update center after finer evaluation and get ell threshold
  center_idx = find(cluster_errs > opts.target_err, 1, 'last');
  ell_thr = ells(center_idx);
catch ME
  warning('many errors for %s (n=%d,d=%d,D=%d,Ng=%d,sigma=%.3e)', ...
      method_name, n, d, D, Ng, sigma);
end

cluster_errs = [ells' cluster_errs];

if ~isempty(opts.out_mat)
  save(opts.out_mat, 'n', 'd', 'D', 'Ng', 'sigma', ...
      'method_name', 'solver', 'params', 'opts', 'cluster_errs', ...
      'center_idx', 'ell_thr', 'summary', 'histories', 'ME');
end

if isempty(ME) && opts.prtlevel > 1
  plot(cluster_errs(Idx, 1), cluster_errs(Idx, 2), 'ko-', ...
       cluster_errs(Idx, 1), opts.target_err*ones(length(Idx), 1), 'b-', ...
      'MarkerSize', 5, 'LineWidth', 2);
end
end


function [cluster_err, seed, status, ...
    summary, histories] = eval_scmd_synth_run(n, d, D, Ng, sigma, ...
    ell, seed, method_name, solver, params, ntrial, summary, ...
    histories, trial_opts)
% eval_scmd_synth_run   evaluate a "run" of ntrial trials
cluster_errs = zeros(ntrial, 1);
trialidx = size(summary, 1);
for ii=1:ntrial
  seed = seed + 1;
  trialidx = trialidx + 1;
  [cluster_errs(ii), ~, summary(trialidx, :), ...
      histories{trialidx}] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ell, ...
      seed, method_name, solver, params, trial_opts);
end
status = mean(isnan(cluster_errs)) > 0.2;
cluster_err = nanmedian(cluster_errs);
end


function check_status(status)
if status ~= 0
  error('too many failures')
end
end
