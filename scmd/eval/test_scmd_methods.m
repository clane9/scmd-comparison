function test_scmd_methods
% test_scmd_methods   test all scmd methods on relatively easy synthetic data
%   example

% (Expected output -- 9/16/19)
%
% Testing LRMC+SSC...
%
% k=1, groups_updt=1.000, C_updt=8.34e-01, Y_updt=NaN
%
% (LRMC+SSC) cluster=0.0000, comp=7.242e-01, rtime=0.1089
%
% Testing ZF-SSC+LRMC...
%
% k=1, groups_updt=1.000, C_updt=7.65e-01, Y_updt=8.13e-01
%
% (ZF-SSC+LRMC) cluster=0.0120, comp=1.755e-01, rtime=0.1790
%
% Testing PZF-SSC+LRMC...
%
% k=1, groups_updt=1.000, C_updt=1.09e+00, Y_updt=8.13e-01
%
% (PZF-SSC+LRMC) cluster=0.0000, comp=1.080e-05, rtime=0.1601
%
% Testing Alt ZF-SSC+LRMC...
%
% k=1, groups_updt=1.000, C_updt=7.65e-01, Y_updt=8.13e-01
% k=2, groups_updt=0.008, C_updt=8.13e-01, Y_updt=7.07e-01
% k=3, groups_updt=0.000, C_updt=7.69e-01, Y_updt=0.00e+00
%
% (Alt ZF-SSC+LRMC) cluster=0.0040, comp=6.933e-02, rtime=0.4457
%
% Testing Alt PZF-SSC+LRMC...
%
% k=1, groups_updt=1.000, C_updt=1.09e+00, Y_updt=8.13e-01
% k=2, groups_updt=0.000, C_updt=8.21e-01, Y_updt=0.00e+00
%
% (Alt PZF-SSC+LRMC) cluster=0.0000, comp=1.080e-05, rtime=0.2972
%
% Testing SSC-SEMC...
%
% k=1, groups_updt=1.000, C_updt=8.34e-01, Y_updt=2.54e-01
% k=2, groups_updt=0.000, C_updt=5.01e-01, Y_updt=3.39e-01
% k=3, groups_updt=0.008, C_updt=4.82e-01, Y_updt=3.78e-01
% k=4, groups_updt=0.020, C_updt=6.97e-01, Y_updt=3.09e-01
% k=5, groups_updt=0.008, C_updt=4.69e-01, Y_updt=3.40e-01
% k=6, groups_updt=0.000, C_updt=2.68e-01, Y_updt=2.65e-01
% k=7, groups_updt=0.004, C_updt=2.75e-01, Y_updt=1.87e-01
% k=8, groups_updt=0.000, C_updt=3.03e-01, Y_updt=1.70e-01
% k=9, groups_updt=0.016, C_updt=2.09e-01, Y_updt=1.64e-01
% k=10, groups_updt=0.024, C_updt=1.54e-01, Y_updt=1.66e-01
%
% (SSC-SEMC) cluster=0.0560, comp=1.338e+00, rtime=0.7992
%
% Testing LADMC+SSC...
%
% k=1, groups_updt=1.000, C_updt=9.19e-01, Y_updt=NaN
%
% (LADMC+SSC) cluster=0.1720, comp=1.220e+00, rtime=2.6129
%
% Testing TSC+LRMC...
%
% k=1, groups_updt=1.000, C_updt=9.90e-01, Y_updt=8.13e-01
%
% (TSC+LRMC) cluster=0.0200, comp=1.543e-01, rtime=0.1243
%
% Testing Alt TSC+LRMC...
%
% k=1, groups_updt=1.000, C_updt=9.90e-01, Y_updt=8.13e-01
% k=2, groups_updt=0.008, C_updt=1.28e+00, Y_updt=1.94e-01
% k=3, groups_updt=0.000, C_updt=4.78e-01, Y_updt=0.00e+00
%
% (Alt TSC+LRMC) cluster=0.0120, comp=1.474e-01, rtime=0.4063
%
% Testing S3LR...
%
% k=1, obj=1.43e+02, fc_norm=2.09e+00, Theta_updt=1, Y_updt=3.76e-01
% k=2, obj=8.14e+01, fc_norm=4.56e-02, Theta_updt=0, Y_updt=2.25e-01
% k=3, obj=8.09e+01, fc_norm=2.85e-03, Theta_updt=0, Y_updt=4.00e-04
% k=4, obj=8.07e+01, fc_norm=2.38e-03, Theta_updt=0, Y_updt=6.74e-06
%
% (S3LR) cluster=0.0000, comp=5.460e-05, rtime=1.5424
%
% Testing GSSC...
%
% k=1, obj=1.58e+00, obj_updt=-2.99e-08, U_updt=9.85e-06
%
% (GSSC) cluster=0.0000, comp=3.861e-01, rtime=4.2892
%
% Testing LR-GSSC...
%
% k=1, obj=8.51e+00, obj_updt=7.22e+00, U_updt=1.01e+00
% k=2, obj=8.51e+00, obj_updt=2.87e-03, U_updt=4.17e-04
% k=3, obj=8.50e+00, obj_updt=1.94e-03, U_updt=2.86e-04
% k=4, obj=8.50e+00, obj_updt=1.36e-03, U_updt=2.02e-04
% k=5, obj=8.50e+00, obj_updt=9.59e-04, U_updt=1.44e-04
%
% (LR-GSSC) cluster=0.0000, comp=7.586e-02, rtime=14.4941

% generate relatively easy synthetic data example
n = 5; d = 5; D = 25; Ng = 50; sigma = 0.0; ell = 20; seed = 2001;
[X, groups_true, Omega] = generate_scmd_data(n, d, D, Ng, sigma, ell, seed);

Omegac = ~Omega;
Xunobs = X(Omegac);
Xunobs_norm = norm(Xunobs);

test_all = 0;
test_methods = {'ssc-semc', 's3lr'};
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
  [groups, Y, ~] = alt_sc_mc(X, Omega, n, params);
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / Xunobs_norm;
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
  [groups, Y, ~] = alt_sc_mc(X, Omega, n, params);
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / Xunobs_norm;
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
  [groups, Y, ~] = alt_sc_mc(X, Omega, n, params);
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / Xunobs_norm;
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
  [groups, Y, ~] = alt_sc_mc(X, Omega, n, params);
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / Xunobs_norm;
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
  [groups, Y, ~] = alt_sc_mc(X, Omega, n, params);
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / Xunobs_norm;
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
  [groups, Y, ~] = alt_sc_mc(X, Omega, n, params);
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / Xunobs_norm;
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
  [groups, Y, ~] = alt_sc_mc(X, Omega, n, params);
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / Xunobs_norm;
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
  [groups, Y, ~] = alt_sc_mc(X, Omega, n, params);
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / Xunobs_norm;
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
  [groups, Y, ~] = alt_sc_mc(X, Omega, n, params);
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / Xunobs_norm;
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
  [groups, Y, ~] = s3lr(X, Omega, n, params);
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / Xunobs_norm;
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
  params = struct('init', 'pzf-ensc+lrmc', 'squared', 0, 'lr_mode', 0, ...
      'lambda', 1e-3, 'gamma', 1e-3, 'lrmc_final', 0, 'maxit', 50, ...
      'tol', 1e-3, 'prtlevel', prtlevel, 'loglevel', loglevel);
  [groups, Y, ~] = gssc(X, Omega, n, 2*d, params);  % note 2d
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / Xunobs_norm;
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
  params = struct('init', 'pzf-ensc+lrmc', 'squared', 0, 'lr_mode', 1, ...
      'lambda', 1e-3, 'gamma', 2, 'lrmc_final', 0, 'maxit', 50, ...
      'tol', 1e-3, 'prtlevel', prtlevel, 'loglevel', loglevel);
  [groups, Y, ~] = gssc(X, Omega, n, 2*d, params);  % note 2d
  cluster_err = 1-evalAccuracy(groups_true, groups);
  comp_err = norm(Y(Omegac) - Xunobs) / Xunobs_norm;
  rtime = toc(tstart);
  fprintf('\n(%s) cluster=%.4f, comp=%.3e, rtime=%.4f \n\n', method_name, ...
      cluster_err, comp_err, rtime)
end

end
