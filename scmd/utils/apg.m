function [x, history] = apg(x0, ffun, rfun, params)
% apg    Generic accelerated proximal gradient descent algorithm
%
%   [x, history] = apg(x0, ffun, rfun, params)
%
%   Minimize an objective f(x) + r(x) using the accelerated proximal gradient
%   descent method.
%
%   Args:
%     x0: Initial guess.
%     ffun, rfun: string function names or function handles.
%       [f, g] = ffun(x) returns function value and optionally gradient.
%       r = rfun(x) returns value of r at x.
%       [r, z] = rfun(u, alpha) returns r value and proximal step from u, with
%         penalty alpha.
%     params: Struct containing parameters for optimization:
%       accel: Whether to do nesterov acceleration [default: 1].
%       maxit: Maximum iterations [default: 500].
%       tol: Stopping tolerance [default: 1e-6].
%       prtlevel: 1=basic per-iteration output [default: 0].
%       loglevel: 1=basic summary info, 2=detailed per-iteration info
%         [default: 1]
%
%   Returns:
%     x: final iterate.
%     history: Struct containing containing fields:
%       iter: Total iterations.
%       status: 0 if stopping tolerance achieved, 1 otherwise.
%       tol: Final stopping tolerance.
%       obj, f, r: objective, f value, r value (per-iteration).
%       update: Relative inf norm change in iterates (per-iteration).
%       alpha: Minimum step size (per-iteration).

if nargin < 4
  params = struct;
end
fields = {'accel', 'maxit', 'tol', 'prtlevel', 'loglevel'};
defaults = {true, 500, 1e-6, 0, 1};
for i=1:length(fields)
  params = setdefaultfield(params, fields{i}, defaults{i});
end

% If f, r functions are strings, convert to function handles.
if ischar(ffun)
  ffun = str2func(ffun);
end
if ischar(rfun)
  rfun = str2func(rfun);
end

% Initialize step size, iterates
alpha = 1; alphamin = 1e-8; alphamax = 1e4; eta = 0.5;
x = x0; xprev = x0;
minupdate = inf;

% Print form str.
printformstr = ['k=%d \t update=%.2e \t obj=%.2e \t f=%.2e \t r=%.2e \t alpha=%.2e \t '...
    'obj_dec=%.2e \t suff_dec=%d \n'];

% Accelerated proximal gradient loop.
status = 1; iter = 0;
while iter <= params.maxit
  iter = iter + 1;
  % Acceleration (Proximal Algorithms, Sect. 4.3)
  if params.accel
    beta = (iter-1) / (iter + 2);
    y = x + beta*(x - xprev);
  else
    y = x;
  end
  xprev = x;

  % Compute f value, gradient.
  [fprev, g] = ffun(y);
  objprev = fprev + rfun(y);

  % Determine step size alpha < 2/L using the simple backtracking strategy.
  if minupdate > 1e-3
    alpha = min(alpha/eta, alphamax);
  end
  while 1
    [~, x] = rfun(y - alpha*g, alpha);
    step = x - y;
    fhat = fprev + frodot(g, step) + (0.5/alpha)*fronormsqrd(step);

    f = ffun(x); r = rfun(x); obj = f + r;
    update = sqrt(fronormsqrd(x - y));
    obj_dec = objprev - obj;
    pred_dec = (0.5/alpha)*update^2;
    suff_dec = obj_dec >= 0.1*pred_dec;

    if (f <= fhat && obj_dec >= 0) || alpha <= alphamin
      break
    else
      alpha = max(eta*alpha, alphamin);
    end
  end

  update = update / max(1, sqrt(fronormsqrd(x)));
  minupdate = min(minupdate, update);

  if params.prtlevel > 0
    fprintf(printformstr, iter, update, obj, f, r, alpha, obj_dec, ...
        suff_dec);
  end

  if params.loglevel > 1
    history.obj(iter) = obj;
    history.f(iter) = f;
    history.r(iter) = r;
    history.update(iter) = update;
    history.alpha(iter) = alpha;
  end

  % Check stopping tolerance.
  if update < params.tol
    status = 0;
    break
  end
end
history.iter = iter; history.status = status; history.tol = update;
end
