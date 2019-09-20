function [solver, params] = get_solver_params(method_name, param_choices, paridx, npar, seed)

method_name = lower(method_name);

if regexp(method_name, '^(alt )?((p?zf\-(s|en)sc\+lrmc)|((lr|lad)mc\+(s|en)sc))$')
  solver = 'alt_sc_mc';
  method_pars = {'ensc_lambda0', 'ensc_gamma'};
  method_par_choices = {param_choices.ensc_lambda0, param_choices.ensc_gamma};
  params = set_params(method_pars, method_par_choices, paridx, npar, seed);
  params.sc_method = 'ensc';
  params.mc_method = 'lrmc';
  if contains(method_name, 'lrmc+')
    params.init = 'lrmc';
  elseif contains(method_name, 'ladmc+')
    params.init = 'ladmc';
  else
    params.init = 'zf';
  end
  if contains(method_name, 'pzf')
    params.ensc_pzf = 1;
  else
    params.ensc_pzf = 0;
  end
  if contains(method_name, 'ssc')
    params.ensc_gamma = 1;
  end
  if contains(method_name, 'alt')
    params.maxit = 20;
  elseif regexp(method_name, '(lad|lr)mc\+')
    params.maxit = 0.5;
  else
    params.maxit = 1;
  end
elseif regexp(method_name, '(s|en)sc\-semc')
  solver = 'alt_sc_mc';
  method_pars = {'ensc_lambda0', 'ensc_gamma', 'semc_eta'};
  method_par_choices = {param_choices.ensc_lambda0, param_choices.ensc_gamma, ...
      param_choices.semc_eta};
  params = set_params(method_pars, method_par_choices, paridx, npar, seed);
  params.init = 'pzf-ensc+lrmc';
  params.sc_method = 'ensc';
  params.ensc_pzf = 0;
  params.mc_method = 'semc';
  params.maxit = 20;
  if contains(method_name, 'ssc')
    params.ensc_gamma = 1;
  end
elseif regexp(method_name, '(alt )?tsc\+lrmc')
  solver = 'alt_sc_mc';
  method_pars = {'tsc_qfrac'};
  method_par_choices = {param_choices.tsc_qfrac};
  params = set_params(method_pars, method_par_choices, paridx, npar, seed);
  params.init = 'zf';
  params.sc_method = 'tsc';
  params.ensc_pzf = 0;
  params.mc_method = 'lrmc';
  if contains(method_name, 'alt')
    params.maxit = 20;
  else
    params.maxit = 1;
  end
elseif regexp(method_name, 's3lr')
  solver = 's3lr';
  method_pars = {'lambda', 'gamma'};
  method_par_choices = {param_choices.s3lr_lambda, param_choices.s3lr_gamma};
  params = set_params(method_pars, method_par_choices, paridx, npar, seed);
  params.init = 'pzf-ensc+lrmc';
  params.alpha = 1;
  params.maxit = 20;
elseif regexp(method_name, '(lr\-)?gssc')
  solver = 'gssc';
  method_pars = {'rfrac', 'lambda', 'gamma'};
  method_par_choices = {param_choices.gssc_rfrac, ...
    param_choices.gssc_lambda, param_choices.gssc_gamma};
  params = set_params(method_pars, method_par_choices, paridx, npar, seed);
  params.init = 'pzf-ensc+lrmc';
  params.optim = 'apg';
  params.squared = 1;
  params.lrmc_final = 0;
  params.maxit = 50;
  if contains(method_name, 'lr-')
    params.lr_mode = 1;
  else
    params.lr_mode = 0;
  end
elseif regexp(method_name, 'srme-mc')
  solver = 'srme_mc_admm';
  method_pars = {'alpha', 'lambda'};
  method_par_choices = {param_choices.srme_mc_alpha, ...
      param_choices.srme_mc_lambda};
  params = set_params(method_pars, method_par_choices, paridx, npar, seed);
  params.init = 'pzf-ensc+lrmc';
  params.L_ell = 'nuc';
  params.E_ell = 'frosquared';
  params.maxit = 500;
else
  error('method %s not implemented', method_name)
end

params.prtlevel = 0;
params.loglevel = 0;
params.tol = 1e-4;
end


function params = set_params(method_pars, method_par_choices, paridx, npar, seed)
par_vals = sample_params(npar, method_par_choices, seed);
params = struct;
for ii=1:length(method_pars)
  params.(method_pars{ii}) = par_vals(paridx, ii);
end
end
