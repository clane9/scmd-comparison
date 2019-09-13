function [Y, history] = semc(X, Omega, C, eta)
% semc    Complete missing entries using a self-expression C by solving a
%   constrained least-squares problem
%
%   min_Y  1/2 || Y - YC ||_F^2 + eta/2 || P_{\Omega^c}(Y) ||_F^2
%    s.t.  P_{\Omega}(Y - X) = 0
%
%   can be written as an unconstrained problem over the rows y_i
%
%   min_{y_{\omega_i^c}} 1/2 || (I - C)^T_{\omega_i^c} y_{\omega_i^c}
%                                    + (I - C)^T_{\omega_i} x_{\omega_i} ||_2^2
%                        + eta/2 || y_{\omega_i^c} ||_2^2
%
%   [Y, history] = semc(X, Omega, C, eta)
%
%   Args:
%     X, Omega: D x N data matrix and observed entry mask.
%     C: N x N self-expression matrix.
%     eta: l2 squared regularization parameter [default: 0].
%
%   Returns:
%     Y: D x N data completion
%     history: struct containing the following information.
%       rtime: total runtime in seconds.
%       iter, status: always zero, included for consistency.
%       rc: minimum reciprocal condition number of (I - C)^T_{\omega_i^c} over
%         all rows i.
tstart = tic;
[D, N] = size(X);
Omega = logical(Omega);
Omegac = ~Omega;
C = full(C);

Y = X;
ICT = (eye(N) - C)';
history.rc = 1;
for ii=1:D
  omegai = Omega(ii, :);
  omegaic = Omegac(ii, :);
  xi = X(ii, :)';

  A = ICT(:, omegaic);
  b = - (ICT(:, omegai) * xi(omegai));
  if eta > 0
    Y(ii, omegaic) = (A' * A + eta * eye(size(A, 2))) \ (A' * b);
  else
    Y(ii, omegaic) = A \ b;
  end

  history.rc = min(rcond(A), history.rc);
end
history.iter = 0; history.status = 0; history.rtime = toc(tstart);
end
