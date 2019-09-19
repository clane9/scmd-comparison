function test_scmd_methods
% test_scmd_methods   test all scmd methods on relatively easy synthetic data
%   example

% relatively easy synthetic data example
n = 5; d = 5; D = 25; Ng = 50; sigma = 0.0; ell = 20; seed = 2001;

test_all = 1;
test_methods = {'lrmc+ssc'};
prtlevel = 0;
loglevel = 0;

% ===============================================================================
% 1. LRMC+SSC
% ===============================================================================
method_name = 'LRMC+SSC';
if test_all || any(strcmpi(method_name, test_methods))
  tstart = tic;
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'lrmc', 'sc_method', 'ensc', 'ensc_pzf', 0, ...
      'ensc_lambda0', 20, 'ensc_gamma', 0.9, 'maxit', 0.5, ...
      'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'alt_sc_mc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.0000 7.146e-01]);
end

% ===============================================================================
% 2. ZF+SSC
% ===============================================================================
method_name = 'ZF-SSC+LRMC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'zf', 'sc_method', 'ensc', 'ensc_pzf', 0, ...
      'ensc_lambda0', 20, 'ensc_gamma', 0.9, 'maxit', 1, ...
      'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'alt_sc_mc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.0040 1.220e-01]);
end

% ===============================================================================
% 3. PZF-SSC+LRMC
% ===============================================================================
method_name = 'PZF-SSC+LRMC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'zf', 'sc_method', 'ensc', 'ensc_pzf', 1, ...
      'ensc_lambda0', 20, 'ensc_gamma', 0.9, 'maxit', 1, ...
      'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'alt_sc_mc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.0000 1.672e-05]);
end

% ===============================================================================
% 4. Alt ZF-SSC+LRMC
% ===============================================================================
method_name = 'Alt ZF-SSC+LRMC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'zf', 'sc_method', 'ensc', 'ensc_pzf', 0, ...
      'ensc_lambda0', 20, 'ensc_gamma', 0.9, 'maxit', 10, ...
      'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'alt_sc_mc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.0000 1.672e-05]);
end

% ===============================================================================
% 4. Alt PZF-SSC+LRMC
% ===============================================================================
method_name = 'Alt PZF-SSC+LRMC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'zf', 'sc_method', 'ensc', 'ensc_pzf', 1, ...
      'ensc_lambda0', 20, 'ensc_gamma', 0.9, 'maxit', 10, ...
      'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'alt_sc_mc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.0000 1.672e-05]);
end

% ===============================================================================
% 5. SSC-SEMC
% ===============================================================================
method_name = 'SSC-SEMC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'pzf-ensc+lrmc', 'sc_method', 'ensc', 'ensc_pzf', 0, ...
      'ensc_lambda0', 10, 'ensc_gamma', 1, 'mc_method', 'semc', ...
      'semc_eta', 10, 'maxit', 10, 'prtlevel', prtlevel, ...
      'loglevel', loglevel);
  solver = 'alt_sc_mc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.0000 8.469e-01]);
end

% ===============================================================================
% 6. LADMC+SSC
% ===============================================================================
method_name = 'LADMC+SSC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'ladmc', 'sc_method', 'ensc', 'ensc_pzf', 0, ...
      'ensc_lambda0', 20, 'ensc_gamma', 0.9, 'maxit', 0.5, ...
      'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'alt_sc_mc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.1600 1.244e-00]);
end

% ===============================================================================
% 7. TSC+LRMC
% ===============================================================================
method_name = 'TSC+LRMC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'zf', 'sc_method', 'tsc', 'maxit', 1, ...
      'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'alt_sc_mc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.0240 1.218e-01]);
end

% ===============================================================================
% 8. Alt TSC+LRMC
% ===============================================================================
method_name = 'Alt TSC+LRMC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'zf', 'sc_method', 'tsc', 'maxit', 10, ...
      'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'alt_sc_mc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.0080 1.029e-01]);
end

% ===============================================================================
% 9. S3LR
% ===============================================================================
method_name = 'S3LR';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'pzf-ensc+lrmc', 'maxit', 10, ...
      'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 's3lr';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.0000 2.329e-05]);
end

% ===============================================================================
% 10. GSSC
% ===============================================================================
method_name = 'GSSC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('r', 2*d, 'init', 'pzf-ensc+lrmc', 'optim', 'apg', 'squared', 1, ...
      'lr_mode', 0, 'lambda', 1e-3, 'gamma', 1e-3, 'lrmc_final', 0, ...
      'maxit', 50, 'tol', 1e-5, 'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'gssc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.0000 2.122e-01]);
end

% ===============================================================================
% 11. LR-GSSC
% ===============================================================================
method_name = 'LR-GSSC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('r', 2*d, 'init', 'pzf-ensc+lrmc', 'optim', 'apg', 'squared', 1, ...
      'lr_mode', 1, 'lambda', 1e-3, 'gamma', 1, 'lrmc_final', 0, ...
      'maxit', 50, 'tol', 1e-5, 'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'gssc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.0000 1.199e-02]);
end

% ===============================================================================
% 12. SRME-MC
% ===============================================================================
method_name = 'SRME-MC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'pzf-ensc+lrmc', 'alpha', 1.0, 'lambda', 10, ...
      'L_ell', 'frosquared', 'maxit', 100, 'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'srme_mc_admm';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
  check_isclose(method_name, [cluster_err comp_err], [0.0000 2.539e-01]);
end

end


function check_isclose(method_name, a, b, tol)
if nargin < 4; tol = 1e-3; end
if any(abs(a - b) ./ max(abs(b), 1e-3) > tol)
  warning('unexpected outputs for %s', method_name);
end
end
