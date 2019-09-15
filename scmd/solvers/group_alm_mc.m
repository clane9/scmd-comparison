function [Y, history] = group_alm_mc(X, Omega, groups, Y0, params)
% group_alm_mc   group-wise LRMC.
%
%   [Y, history] = group_alm_mc(X, Omega, groups, Y0, params)
%
%   Args:
%     X: D x N incomplete data matrix
%     Omega: D x N logical indicator of observed entries
%     groups: N x 1 cluster assignment.
%     Y0: D x N initial guess for completion.
%     params: Struct containing problem parameters
%       mu: LRMC ALM penalty parameter [default: 1/||X||_{2,1}]
%       alpha: rate for increasing mu after each iteration [default: 1.1]
%       maxit: maximum iterations [default: 500]
%       tol: stopping tolerance [default: 1e-5]
%       prtlevel: 1=basic per-iteration output [default: 0]
%       loglevel: 0=basic summary info, 1=detailed per-iteration info
%         [default: 0]
%
%   Returns:
%     Y: D x N low-rank completion.
%     history: struct containing the following information per group.
%       feas: feasibility, per group, iteration if loglevel > 0.
%       obj: objective, per group, iteration if loglevel > 0.
%       iter, status, conv_cond: number of iterations, termination status,
%         convergence condition at termination. per group if loglevel > 0.
%       rtime: total runtime in seconds, per group if loglevel > 0.
Y = zeros(size(X));
if isempty(Y0); Y0 = X; end
classes = unique(groups);
n = length(classes);
for ii=1:n
  maski = groups == classes(ii);
  [Y(:, maski), grp_history] = alm_mc(X(:, maski), Omega(:, maski), ...
      Y0(:, maski), params);

  history.iter(ii) = grp_history.iter;
  history.status(ii) = grp_history.status;
  history.conv_cond(ii) = grp_history.conv_cond;
  if params.loglevel > 0
    history.obj{ii} = grp_history.obj;
    history.feas{ii} = grp_history.feas;
  else
    history.obj(ii) = grp_history.obj;
    history.feas(ii) = grp_history.feas;
  end
  history.rtime(ii) = grp_history.rtime;
end

if params.loglevel <= 0
  history.iter = max(history.iter);
  history.status = max(history.status);
  history.conv_cond = max(history.conv_cond);
  history.obj = sum(history.obj);
  history.feas = max(history.feas);
  history.rtime = sum(history.rtime);
end
end
