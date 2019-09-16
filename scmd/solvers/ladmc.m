function [Y, history] = ladmc(X, Omega, Y0, params)
% ladmc   low-algebraic dimension matrix completion from Ongie et al., 2018.
%
%   [Y, history] = ladmc(X, Omega, Y0, params)
%
%   Args:
%     X: D x N incomplete data matrix
%     Omega: D x N logical indicator of observed entries
%     Y0: D x N initial guess for completion.
%     params: Struct containing problem parameters
%       mu: LRMC ALM penalty parameter [default: 1/||X^{\otimes 2}||_{2,1}]
%       alpha: rate for increasing mu after each iteration [default: 1.1]
%       maxit: maximum iterations [default: 500]
%       tol: stopping tolerance [default: 1e-5]
%       prtlevel: 1=basic per-iteration output [default: 0]
%       loglevel: 0=basic summary info, 1=detailed per-iteration info
%         [default: 0]
%
%   Returns:
%     Y: D x N low-rank completion.
%     history: struct containing the following information
%       feas: LRMC feasibility, per iteration if loglevel > 0.
%       obj: LRMC objective, per iteration if loglevel > 0.
%       max_sv2: maximum 2nd singular value over all completed tensorized data
%         points. should be small if completion is correct.
%       iter, status, conv_cond: number of iterations, termination status,
%         convergence condition at termination.
%       rtime: total runtime in seconds
%
%   References:
%     Pimentel-Alarcón, D., Ongie, G., Balzano, L., Willett, R., & Nowak, R.
%     (2017, October). Low algebraic dimension matrix completion. In 2017 55th
%     Annual Allerton Conference on Communication, Control, and Computing
%     (Allerton) (pp. 790-797). IEEE.
%
%     Ongie, G., Balzano, L., Pimentel-Alarcón, D., Willett, R., & Nowak, R. D.
%     (2018). Tensor methods for nonlinear matrix completion. arXiv preprint
%     arXiv:1804.10266.
tstart = tic;
[D, N] = size(X);
Omega = logical(Omega);
Omegac = ~Omega;
X(Omegac) = 0;

if nargin < 4; params = struct; end
if isempty(Y0); Y0 = X; end

% tensorize by taking outer product of each column and do lrmc
XX = ord2_tensorize(X);
YY0 = ord2_tensorize(Y0);
OOmega = ord2_tensorize(Omega);
[YY, history] = alm_mc(XX, OOmega, YY0, params);

% pull back into original space by taking top singular vector of each
% tensorized completion
Y = zeros(D, N);
history.max_sv2 = 0;
for jj=1:N
  yy = reshape(YY(:, jj), [D D]);
  [U, S, ~] = svds(yy, 2);
  s = diag(S);
  Y(:, jj) = sqrt(s(1)) * U(:, 1);
  history.max_sv2 = max(history.max_sv2, s(2)/s(1));
end

% ensure constraint satisfied exactly
Y(Omega) = X(Omega);

history.rtime = toc(tstart);
end


function ZZ = ord2_tensorize(Z)
% ord2_tensorize    compute 2nd order tensorization of Z.
[D, N] = size(Z);
% take outer product of each column with itself
ZZ = repmat(reshape(Z, [D 1 N]), [1 D 1]) .* ...
    repmat(reshape(Z, [1 D N]), [D 1 1]);
% flatten first two dims
ZZ = reshape(ZZ, [], N);
end
