function [Z, E, stat] = lrr(X, lambda, opt)
% This routine solves the following nuclear-norm optimization problem 
% by using inexact Augmented Lagrange Multiplier, which has been also presented 
% in the paper entitled "Robust Subspace Segmentation 
% by Low-Rank Representation".
%------------------------------
% min |Z|_*+lambda*|E|_2,1
% s.t., X = XZ+E
%--------------------------------
% inputs:
%        X -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors.
if nargin<3
    lambda = 0.1;
    opt.tol =1e-4;
    opt.maxIter =1e6;
    opt.rho =1.1;
end

% parameter settings
tol = opt.tol;
maxIter = opt.maxIter;
% rho = 1.1;
rho =opt.rho;
[d, n] = size(X);

max_mu = 1e30;
%mu = 1e-6;
xtx = X'*X;
inv_x = inv(xtx+eye(n));
%X_norm_two =normest(X);
%X_norm_fro = norm( X, 'fro');
X_norm_inf = norm( X(:), inf);%changed by Zhouchen Lin

%% Initializing optimization variables
% intialize
J = zeros(n,n);
Z = zeros(n,n);
E = sparse(d,n);

Y1 = zeros(d,n);
Y2 = zeros(n,n);

%% Revising the setting of mu:
% temp =inv_x*(xtx-X'*E+J+(X'*Y1-Y2)/mu);
%mu = 1/(normest(temp))
%mu = min(1.25/normest(xtx), 1e-6);
mu = 1.25/normest(X);

%% Start main loop
iter = 0;
%disp(['initial,rank=' num2str(rank(Z))]);
while iter<maxIter
    iter = iter + 1;
    
    %% 1. Updating E
    temp = X-X*Z+Y1/mu;
    E = solve_l1l2(temp,lambda/mu);  
    
    %% 2. Updating J
    temp = Z + Y2/mu;
    if (normest(temp,1e-2) >1/mu)
        [U,sigma,V] = svd(temp,'econ');
     %   [U,sigma,V] = lansvd(temp,30,'L');
        sigma = diag(sigma);
        svp = length(find(sigma>1/mu));
        if svp>=1
            sigma = sigma(1:svp)-1/mu;
        else
            svp = 1;
            sigma = 0;
        end
        J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    %if (norm(J,'fro')>1e1) 
    %    disp('J is active.');
    %end
    end
    
    %% 3. Updating Z
    Z = inv_x*(xtx-X'*E+J+(X'*Y1-Y2)/mu);
    
    %% 4. Checking convergence conditions
    xmaz = X-X*Z;    
    leq1 = xmaz-E;
    leq2 = Z-J;
    %leq2 = (Z-J)/norm(X,'fro');
    % whether the two conditions should be normalized by norm(X, 'fro')? 
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2)))) / X_norm_inf;
    if iter==1 || mod(iter,50)==0 || stopC<tol
        rankZ =rank(Z,1e-3*normest(Z,2));
        stat.iter =iter;
        stat.rank =rankZ;
        stat.stopALM =stopC;
        %disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e')  ',rank='  num2str(rankZ)  ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    %% 5. Updating \mu
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end

function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end


function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end