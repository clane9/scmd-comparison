function [cluster_err, comp_err, summary, history] = eval_scmd_synth_trial(...
    n, d, D, Ng, sigma, ell, seed, method_name, solver, params, opts)
% eval_scmd_synth_trial   Evaluate one trial of synthetic SCMD
%
%   [cluster_err, comp_err, history] = eval_scmd_synth_trial(n, ...
%       d, D, Ng, sigma, ell, seed, method_name, solver, params, opts)
if nargin < 11; opts = struct; end
fields = {'paridx', 'out_summary'};
defaults = {0, ''};
opts = set_default_params(opts, fields, defaults);

history.args = {n, d, D, Ng, sigma, ell, seed, ...
    method_name, solver, params, opts};

try
  [X, groups_true, Omega] = generate_scmd_data(n, d, D, Ng, sigma, ell, seed);
  
  Omegac = ~Omega;
  Xunobs = X(Omegac);
  Xunobs_norm = norm(Xunobs);
  
  if ischar(solver); solver = str2func(solver); end
  [groups, Y, solver_history] = solver(X, Omega, n, params);
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / (Xunobs_norm+eps);
catch ME
  [cluster_err, comp_err] = deal(NaN);
  [solver_history.iter, solver_history.status] = deal(NaN);
  [solver_history.conv_cond, solver_history.rtime] = deal(NaN);
  history.ME = ME;
end

history.solver_history = solver_history;
history.cluster_err = cluster_err;
history.comp_err = comp_err;

summary = {n, d, D, Ng, sigma, ell, seed, method_name, opts.paridx, ...
      cluster_err, comp_err, solver_history.iter, solver_history.status, ...
      solver_history.conv_cond, solver_history.rtime};

if ~isempty(opts.out_summary)
  fid = fopen(opts.out_summary, 'a');
  % columns:
  %
  % n, d, D, Ng, sigma, ell, seed, method_name, paridx,
  % cluster_err, comp_err, iter, status, conv_cond, rtime
  fprintf(fid, '%d,%d,%d,%d,%.9e,%d,%d,%s,%d,%.9f,%.9e,%d,%d,%.9e,%.9f\n', ...
      summary{:});
  fclose(fid);
end
end
