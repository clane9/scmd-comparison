function Y = lsqrmd(X, Omega, A, lambda, B1, B2)
% lsqrmd    Solve least-squares problem with missing data constraints
%
%   min_Y 1/2  || YA - B1 ||_F^2 + lambda/2 || Y - B2 ||_F^2
%        s.t.  P_Omega(Y - X) = 0
%
%   Args:
%     X, Omega: D x N data matrix and observed entry mask
%     A: N x M matrix
%     lambda: penalty parameter on second term [default: 0]
%     B1: D x M matrix [default: []]
%     B2: D x N matrix [default: []]
%
%   Returns:
%     Y: D x N data completion
[D, ~] = size(X);
Omega = logical(Omega);
Omegac = ~Omega;
X(Omegac) = 0;

if nargin < 4; lambda = 0; end
if nargin < 5; B1 = []; end
if nargin < 6; B2 = []; end

Y = X; At = A';
for ii=1:D
  omegai = Omega(ii, :);
  % if no observed entries, do nothing
  if sum(omegai) > 0
    omegaic = Omegac(ii, :);

    Ai = At(:, omegaic);
    bi = - At(:, omegai) * X(ii, omegai)';
    if ~isempty(B1)
      bi = bi + B1(ii, :)';
    end

    if lambda > 0
      AitAiI = Ai' * Ai + lambda * eye(size(Ai, 2));
      Aitbi = Ai'*bi;
      if ~isempty(B2)
        Aitbi = Aitbi + lambda * B2(ii, omegaic)';
      end
      Y(ii, omegaic) = AitAiI \ Aitbi;
    else
      Y(ii, omegaic) = Ai \ bi;
    end
  end
end
end
