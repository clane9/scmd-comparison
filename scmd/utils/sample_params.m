function par_mat = sample_params(nsamples, par_choices, seed)
% sample_params   Sample parameter settings uniformly from a defined grid.
%
%   par_mat = sample_params(nsamples, par_choices, seed)
%
%   Args:
%     nsamples: number of parameter settings to sample
%     par_choices: cell array of parameter choice vectors
%     seed: seed for rng [default: none]
%
%   Returns:
%     par_mat: nsamples x npar matrix of parameter settings
if nargin >= 3; rng(seed); end

npar = length(par_choices);
par_mat = NaN(nsamples, npar);
for ii=1:npar
  Idx = randi(length(par_choices{ii}), nsamples, 1);
  par_mat(:, ii) = par_choices{ii}(Idx);
end
end
