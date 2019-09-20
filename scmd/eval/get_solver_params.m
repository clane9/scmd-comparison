function [solver, params] = get_solver_params(method_name, paridx, npar, seed)

method_name = lower(method_name);

if regexp(method_name, '^(alt )?((p?zf\-(s|en)sc\+lrmc)|((lr|lad)mc\+(s|en)sc))$')
  solver = 'alt_sc_mc';
  par_fields = {'ensc_lambda0', 'ensc_gamma'};
  par_choices = {[5 10 20 40 80 160 320], [0.5 0.6 0.7 0.8 0.9 0.99]};
  params = set_params(par_fields, par_choices, paridx, npar, seed);
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
  par_fields = {'ensc_lambda0', 'ensc_gamma', 'semc_eta'};
  par_choices = {[5 10 20 40 80 160 320], [0.5 0.6 0.7 0.8 0.9 0.99], ...
      [0 1e-4 1e-3 1e-2 0.1 1 10]};
  params = set_params(par_fields, par_choices, paridx, npar, seed);
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
  par_fields = {'tsc_qfrac'};
  par_choices = {[0.05 0.1 0.15 0.2 0.25 0.3]};
  params = set_params(par_fields, par_choices, paridx, npar, seed);
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
  par_fields = {'lambda', 'gamma'};
  par_choices = {[1e-2 0.1 1 10 100], [1e-5 1e-4 1e-3 1e-2 0.1]};
  params = set_params(par_fields, par_choices, paridx, npar, seed);
  params.init = 'pzf-ensc+lrmc';
  params.alpha = 1;
  params.maxit = 20;
elseif regexp(method_name, '(lr\-)?gssc')
  solver = 'gssc';
  par_fields = {'r', 'lambda', 'gamma'};
  par_choices = {[2 5 10 20 30 40 50 60], [1e-5 1e-4 1e-3 1e-2 0.1], ...
      [1e-3 1e-2 0.1 1 10 100]};
  params = set_params(par_fields, par_choices, paridx, npar, seed);
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
  par_fields = {'alpha', 'lambda'};
  par_choices = {[1e-2 0.1 1 10 100], [5 10 20 40 80 160]};
  params = set_params(par_fields, par_choices, paridx, npar, seed);
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


function params = set_params(par_fields, par_choices, paridx, npar, seed)
par_vals = sample_params(npar, par_choices, seed);
params = struct;
for ii=1:length(par_fields)
  params.(par_fields{ii}) = par_vals(paridx, ii);
end
end
