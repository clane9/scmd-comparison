function test_scmd_methods
% test_scmd_methods   test all scmd methods on relatively easy synthetic data
%   example

% (Expected output -- 9/16/19)
%
% Testing LRMC+SSC... 
% 
% k=1, groups_updt=0.800, C_updt=7.77e-01, Y_updt=NaN 
% 
% (LRMC+SSC) cluster=0.0000, comp=7.128e-01, rtime=1.0203 
% 
% Testing ZF-SSC+LRMC... 
% 
% k=1, groups_updt=0.796, C_updt=8.08e-01, Y_updt=7.70e-01 
% 
% (ZF-SSC+LRMC) cluster=0.0040, comp=1.222e-01, rtime=0.1810 
% 
% Testing PZF-SSC+LRMC... 
% 
% k=1, groups_updt=0.800, C_updt=8.62e-01, Y_updt=7.70e-01 
% 
% (PZF-SSC+LRMC) cluster=0.0000, comp=7.174e-06, rtime=0.1691 
% 
% Testing Alt ZF-SSC+LRMC... 
% 
% k=1, groups_updt=0.796, C_updt=8.08e-01, Y_updt=7.70e-01 
% k=2, groups_updt=0.004, C_updt=8.20e-01, Y_updt=6.98e-01 
% k=3, groups_updt=0.000, C_updt=4.35e-01, Y_updt=0.00e+00 
% 
% (Alt ZF-SSC+LRMC) cluster=0.0000, comp=7.174e-06, rtime=0.4552 
% 
% Testing Alt PZF-SSC+LRMC... 
% 
% k=1, groups_updt=0.800, C_updt=8.62e-01, Y_updt=7.70e-01 
% k=2, groups_updt=0.000, C_updt=7.38e-01, Y_updt=0.00e+00 
% 
% (Alt PZF-SSC+LRMC) cluster=0.0000, comp=7.174e-06, rtime=0.3033 
% 
% Testing SSC-SEMC... 
% 
% k=1, groups_updt=0.800, C_updt=8.91e-01, Y_updt=4.39e-01 
% k=2, groups_updt=0.000, C_updt=5.43e-01, Y_updt=2.99e-01 
% k=3, groups_updt=0.000, C_updt=3.59e-01, Y_updt=3.52e-01 
% k=4, groups_updt=0.008, C_updt=3.21e-01, Y_updt=3.23e-01 
% k=5, groups_updt=0.000, C_updt=3.09e-01, Y_updt=3.62e-01 
% k=6, groups_updt=0.004, C_updt=2.64e-01, Y_updt=2.95e-01 
% k=7, groups_updt=0.008, C_updt=2.92e-01, Y_updt=3.79e-01 
% k=8, groups_updt=0.000, C_updt=2.37e-01, Y_updt=4.21e-01 
% k=9, groups_updt=0.004, C_updt=2.39e-01, Y_updt=3.32e-01 
% k=10, groups_updt=0.008, C_updt=3.53e-01, Y_updt=2.83e-01 
% 
% (SSC-SEMC) cluster=0.0200, comp=1.300e+00, rtime=0.8667 
% 
% Testing LADMC+SSC... 
% 
% k=1, groups_updt=0.640, C_updt=9.66e-01, Y_updt=NaN 
% 
% (LADMC+SSC) cluster=0.1600, comp=1.245e+00, rtime=2.5300 
% 
% Testing TSC+LRMC... 
% 
% k=1, groups_updt=0.792, C_updt=9.87e-01, Y_updt=7.70e-01 
% 
% (TSC+LRMC) cluster=0.0240, comp=1.217e-01, rtime=0.1246 
% 
% Testing Alt TSC+LRMC... 
% 
% k=1, groups_updt=0.792, C_updt=9.87e-01, Y_updt=7.70e-01 
% k=2, groups_updt=0.096, C_updt=1.28e+00, Y_updt=2.86e-01 
% k=3, groups_updt=0.012, C_updt=5.94e-01, Y_updt=5.55e-02 
% k=4, groups_updt=0.092, C_updt=3.03e-01, Y_updt=2.34e-02 
% k=5, groups_updt=0.000, C_updt=2.77e-01, Y_updt=0.00e+00 
% 
% (Alt TSC+LRMC) cluster=0.0080, comp=1.030e-01, rtime=0.6027 
% 
% Testing S3LR... 
% 
% k=1, obj=1.03e+02, fc_norm=1.38e+00, Theta_updt=1, Y_updt=1.49e-01 
% k=2, obj=8.14e+01, fc_norm=8.59e-02, Theta_updt=0, Y_updt=1.43e-01 
% k=3, obj=8.11e+01, fc_norm=7.30e-05, Theta_updt=0, Y_updt=4.35e-04 
% k=4, obj=8.10e+01, fc_norm=6.10e-05, Theta_updt=0, Y_updt=2.73e-06 
% 
% (S3LR) cluster=0.0000, comp=2.329e-05, rtime=1.6884 
% 
% Testing GSSC... 
% 
% k=1, obj=1.59e+00, obj_updt=-2.79e-09, U_updt=9.99e-06 
% 
% (GSSC) cluster=0.0000, comp=3.221e-01, rtime=4.1514 
% 
% Testing LR-GSSC... 
% 
% k=1, obj=8.56e+00, obj_updt=7.17e+00, U_updt=1.24e+00 
% k=2, obj=8.56e+00, obj_updt=1.27e-05, U_updt=3.34e-05 
% 
% (LR-GSSC) cluster=0.0000, comp=7.971e-02, rtime=6.4683

% relatively easy synthetic data example
n = 5; d = 5; D = 25; Ng = 50; sigma = 0.0; ell = 25; seed = 2001;

test_all = 1;
test_methods = {'gssc', 'lr-gssc'};
prtlevel = 1;
loglevel = 1;

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
end

% ===============================================================================
% 5. SSC-SEMC
% ===============================================================================
method_name = 'SSC-SEMC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'pzf-ensc+lrmc', 'sc_method', 'ensc', 'ensc_pzf', 0, ...
      'ensc_lambda0', 20, 'ensc_gamma', 0.9, 'mc_method', 'semc', ...
      'semc_eta', 0, 'maxit', 10, 'prtlevel', prtlevel, ...
      'loglevel', loglevel);
  solver = 'alt_sc_mc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
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
end

% ===============================================================================
% 9. S3LR
% ===============================================================================
method_name = 'S3LR';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('init', 'pzf-ensc+lrmc', 'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 's3lr';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
end

% ===============================================================================
% 10. GSSC
% ===============================================================================
method_name = 'GSSC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('r', 2*d, 'init', 'pzf-ensc+lrmc', 'squared', 0, ...
      'lr_mode', 0, 'lambda', 1e-3, 'gamma', 1e-3, 'lrmc_final', 0, ...
      'maxit', 50, 'tol', 1e-3, 'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'gssc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
end

% ===============================================================================
% 11. LR-GSSC
% ===============================================================================
method_name = 'LR-GSSC';
tstart = tic;
if test_all || any(strcmpi(method_name, test_methods))
  fprintf('Testing %s... \n\n', method_name);
  params = struct('r', 2*d, 'init', 'pzf-ensc+lrmc', 'squared', 0, ...
      'lr_mode', 1, 'lambda', 1e-3, 'gamma', 2, 'lrmc_final', 0, ...
      'maxit', 50, 'tol', 1e-3, 'prtlevel', prtlevel, 'loglevel', loglevel);
  solver = 'gssc';
  [cluster_err, comp_err, ~] = eval_scmd_synth_trial(n, d, D, Ng, sigma, ...
      ell, seed, method_name, solver, params);
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
end

end
