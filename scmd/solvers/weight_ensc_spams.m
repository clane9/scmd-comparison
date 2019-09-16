function [C, history] = weight_ensc_spams(X, W, params)
% weight_ensc_spams   Compute self-expression matrix C by weighted elastic-net
%   subspace clustering using SPAMS mexLasso. Solves the problem
%
%   min_C lambda/2 || W \odot (X C - X) ||_F^2
%         + gamma || C ||_1 + (1-gamma)/2 || C ||_F^2
%         s.t. diag(C) = 0
%
%   [C, history] = weight_ensc_spams(X, W, params)
%
%   Args:
%     X, W: D x N data and weight matrices.
%     params: struct containing the following problem parameters.
%       lambda0: least-squares penalty relative to minimum lambda required to
%         guarantee non-zero self-expressions [default: 20].
%       gamma: elastic net tradeoff parameter [default: 0.9].
%       lambda_method: 'nonzero' (individual min lambda per data point) or
%         'fixed' (fixed min lambda for all) [default: 'fixed'].
%       normalize: scale each column of X to have unit l2 norm [default: 1].
%
%   Returns:
%     C: N x N self-expression matrix.
%     history: struct containing the following information.
%       lambda: min lambda * lambda0, either N x 1 if lambda_method ==
%         'nonzero', or 1 x 1.
%       iter, status, conv_cond: always zero, included for consistency.
%       rtime: total runtime in seconds.
tstart = tic;
[D, N] = size(X);

if nargin < 3; params = struct; end
fields = {'lambda0', 'gamma', 'lambda_method', 'normalize'};
defaults = {20, 0.9, 'fixed', 1};
params = set_default_params(params, fields, defaults);

C = zeros(N);
if params.normalize
  X = X ./ repmat(sqrt(sum(X.^2))+eps, [D 1]);
end

% determine minimum lambda required to ensure c_j \neq 0 for all j
%
%   min_{c}  f(c) = lambda/2 || diag(w) (A c - x) ||_2^2
%                   + gamma || c ||_1 + (1 - gamma)/2 || c ||_2^2
%
%   d f (c) = lambda A^T diag(w)^2 (A c - x)
%             + gamma d || c ||_1 + (1 - gamma) c
%
%   0 in d f (0) ==> || A^T diag(w)^2 x ||_\infty \leq gamma / lambda
%
% so, lambda_j > gamma / || X_{-j} diag(w_j)^2 x_j ||_\infty ==> c_j \neq 0.
if isempty(W)
  G = X' * X;
else
  G = X' * (W.^2 .* X);
end
% set diag = 0
G(1:N+1:end) = 0;
lambda_min = params.gamma ./ max(abs(G));
lambda = params.lambda0 * lambda_min;
fixed_method = strcmpi(params.lambda_method, 'fixed');
if fixed_method; lambda = max(lambda); end
history.lambda = lambda;

% solve elastic net per data point
spams_params.numThreads = 1;
history.obj = 0;
for jj=1:N
  x = X(:, jj);
  A = [X(:, 1:(jj-1)) X(:, (jj+1):end)];
  if ~isempty(W)
    w = W(:, jj);
    wmask = w ~= 0; w = w(wmask);
    % handle degenerate case where w is empty
    if isempty(w)
      x = [];
    else
      x = w .* x(wmask);
      A = repmat(w, [1 N-1]) .* A(wmask, :);
    end
  end

  if ~isempty(x)
    if fixed_method; lambdaj = lambda; else; lambdaj = lambda(jj); end
    spams_params.lambda = params.gamma / lambdaj;
    spams_params.lambda2 = (1-params.gamma) / lambdaj;
    c = mexLasso(x, A, spams_params);
    % would be better to keep sparsity, but this is easier for now.
    c = full(c);

    obj = 0.5*lambdaj * sum((A * c - x).^2) ...
        + params.gamma * sum(abs(c)) ...
        + 0.5*(1-params.gamma) * sum(c.^2);
  else
    c = zeros(N-1, 1);
    obj = 0;
  end
  history.obj = history.obj + obj;

  C(:,jj) = [c(1:(jj-1)); 0; c(jj:end)];
end
C = sparse(C);
history.conv_cond = 0; history.iter = 0; history.status = 0;
history.rtime = toc(tstart);
end
