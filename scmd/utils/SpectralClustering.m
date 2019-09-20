%--------------------------------------------------------------------------
% This function takes an adjacency matrix of a graph and computes the
% clustering of the nodes using the spectral clustering algorithm of
% Ng, Jordan and Weiss.
% A: NxN adjacency matrix
% n: number of groups for clustering
% groups: N-dimensional vector containing the memberships of the N points
% to the n groups obtained by spectral clustering
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
% Modified @ Chong You, 2015
% Modified @ Connor Lane, 2018
%--------------------------------------------------------------------------

function groups = SpectralClustering(A, n)

% disable excessive illconditioning warnings
orig_state = warning;
warning('off','all')

if ~issymmetric(A)
  error('Affinity is not symmetric.')
end
if min(min(A)) < 0
  error('Affinity contains negative values.')
end
N = size(A,1);
MAXiter = 1000; % Maximum number of iterations for KMeans
REPlic = 20; % Number of replications for KMeans

% Normalized spectral clustering according to Ng & Jordan & Weiss
% using Normalized Symmetric Laplacian L = I - D^{-1/2} W D^{-1/2}.
DN = spdiags(1./sqrt(sum(A)+eps)', 0, N, N);
LapN = speye(N) - DN*A*DN;
LapN = 0.5*(LapN + LapN'); % Enforce symmetry.
% Using smallestabs setting is crucial when all eigenvalues are very close to
% zero
[vN, ~] = eigs(LapN, n, 'smallestabs');
normN = sqrt(sum(vN.^2, 2));
kerNS = vN ./ repmat(normN + eps, [1, n]);
groups = kmeans(kerNS,n,'maxiter',MAXiter,'replicates',REPlic);

warning(orig_state);
end
