function scmd_crossval_experiment(n, d, D, Ng, sigma, method_name, ...
    paridx, dataseed, npar, ntrial, prefix, jobidx)

parseed = 2813308004;
% parseed = 2814551858

% set parameter possible parameter values
param_choices.ensc_lambda0 = [5 10 20 40 80 160 320];
param_choices.ensc_gamma = [0.5 0.6 0.7 0.8 0.9 0.99];
param_choices.semc_eta = [0 1e-4 1e-3 1e-2 0.1 1 10];
param_choices.tsc_qfrac = [0.05 0.1 0.15 0.2 0.25 0.3];
param_choices.s3lr_lambda = [1e-2 0.1 1 10 100];
param_choices.s3lr_gamma = [1e-5 1e-4 1e-3 1e-2 0.1];
param_choices.gssc_rfrac = [0.02 0.04 0.1 0.2 0.3 0.4 0.5 0.6];
param_choices.gssc_lambda = [1e-5 1e-4 1e-3 1e-2 0.1];
param_choices.gssc_gamma = [1e-3 1e-2 0.1 1 10 100];
param_choices.srme_mc_alpha = [1e-2 0.1 1 10 100];
param_choices.srme_mc_lambda = [5 10 20 40 80 160];

fprintf(['Running experiment n=%d, d=%d, D=%d, Ng=%d, sigma=%.3f, ' ...
    'method=%s, par=%d, seed=%d, job=%d\n\n'], n, d, D, Ng, sigma, ...
    method_name, paridx, dataseed, jobidx);

out_prefix = sprintf(['%s_n%d_d%d_D%d_Ng%d_sigma%.3f_%s_par%d' ...
    '_seed%d_job%05d'], prefix, n, d, D, Ng, sigma, method_name, paridx, ...
    dataseed, jobidx);

[solver, params] = get_solver_params(method_name, param_choices, paridx, ...
    npar, parseed);

fprintf('Parameter setting: \n\n');
disp(params)

slice_opts = struct('seed_start', dataseed, 'ntrial', ntrial, ...
    'paridx', paridx, 'out_summary', [out_prefix '.csv'], ...
    'out_mat', [out_prefix '.mat'], 'restart', 1, 'prtlevel', 1);
eval_scmd_synth_slice(n, d, D, Ng, sigma, method_name, solver, params, ...
    slice_opts);
end
