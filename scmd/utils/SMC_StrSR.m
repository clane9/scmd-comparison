% SMC_StrSR.m
% Description: This code is for the step I in S3LR, i.e. 
%                       Structured Matrix Completion and Structured Sparse Representation, which is solving the
%                       following problem by linearized alternating direction method:
%       arg min   \sum ||D^i ||_*  +  lambda*||E||_1 +  || (Gamma * Theta + gamma0* 11^T) \odot Z||_1 +  tau*||N||_fro
%       s.t. P_\Omega(X) = P_\Omega (D + N), 
%            D = D*Z + E, diag(Z) =0
%      Note: 
%       1) If affine, then  1^T =1^T Z.
%       2) If 'relax==1':  the structured nuclear norm, i.e.,   \sum ||D^i ||_*  is used.
%       3) If 'relax==2': the structured nuclear norm is further relaxed as  k ||D||_* (for  \sum ||D^i ||_* <=  k ||D||_*)
%
%   [Z, D, E, N, stat] =SMC_StrSR(X, Omega,Theta, lambda, Gamma, gamma0, tau, opt, X_init, Z_init,  relax, affine)
%
%    Inputs:  X - incomplete data matrix, d*n data matrix, d is the data dimension, and n is the number of data vectors             
%                 Omega - logical matrix to indicate the observation entries;
%                 Theta  - the "structure" matrix computed via spectral clustering,  where Theta_ij =1 if i and j lie in different subspaces and 0, otherwise.
%                 X_init -  the initialization of X, i.e. the result of last iteration
%                 Z_init -  the initialization of Z, i.e. the result of last iteration
%                 k        -  the number of subspaces
%                 relax ={1, 2} where '1' if  the relaxation of structured nuclear norm is only for the clustering part, and '2' if for both D and Theta.     
%                 affine ={1, 0} if the affine representation is needed
%
%                 lambda: regularization parameter to trade-off between error and regularization
%                 tau: when X is noise-free on observed entries, \tau is set as 'Inf'
%
%                opt.tol: precision to terminate iterations, 1e-6 for toy data and 1e-2 for ExYaleB 
%                opt.maxIter: 1e6 for toy data and 5e2 for real world data set
%             
%        Outputs:
%             D:  The recovered data
%             E:  The estimated error for LRR
%             N:  the noise on observed entries
%             Z:  the low-rank representation coefficient matrix
%             stat.iter
%             stat.rank
%             stat.stopALM
%
% Copyright by Chun-Guang Li
% Date: Sept. 18, 2013
% Modified by July 31, 2015
function [Z, D, E, N, stat] =SMC_StrSR(X, Omega,Theta, lambda, Gamma, gamma0, tau, opt, X_init, Z_init,  relax, affine)
if nargin< 6
    gamma0 =0;
end 
if nargin< 7
    tau =Inf; % by default, X is noise-free.
end
if nargin < 8
    opt.tol =1e-4;
    opt.maxIter =1e6;
    opt.rho =1.1;
    opt.mu_max =1e4;
    opt.norm_sr ='1';
    opt.norm_mc ='1';
end   
if nargin < 9
    X_init =zeros(size(X));
end
if nargin < 10    
    Z_init =zeros(size(X,2));
end
if nargin < 11    
    relax =1;
end
if nargin < 12    
    affine =0;
end

%% FOR TEST ONLY:
TEST_CODE = 0;
if (TEST_CODE)
    err1 =[];err2 =[];err3 =[];err4 =[];
    h=figure;
end

[d, n] = size(X);

%% Initializing optimization variables
D = X_init; %instead of zeros(d,n);
D_old = D;
E = sparse(d,n);
N = E;
In = eye(n);
Z = Z_init; % instead of zeros(n);
Z_old = Z;
Y1 = zeros(d,n);
Y2 = Y1;
Zeros =zeros(d,n);
e_zeros =zeros(1,n);

P_Omega_star_P_Omega =logical(Omega);
P_Omega = find(Omega);
%P_Theta_star_P_Theta =logical(Theta);

X_norm_inf = norm( X(:), inf);

%% Find idx from Theta
k =rank(double(1-Theta));
thresh =0.8;
if  (relax ==2) || (sum(double(Theta(:))) <0.5)
    idx = ones(1,n);
    k = max(idx);
else
    idx = find_idx_from_Theta(1-Theta, k, thresh);
end

%% Start main loop
iter = 0;
mu = opt.mu;
rho =opt.rho;
mu_max = opt.mu_max;
tol = opt.tol; %tol =0.05; epsilon =1e-3;
maxIter = opt.maxIter;
while iter < maxIter
    iter = iter + 1;

    %% Solving D and N by linearized ADM: Structured Matrix Completion
    %% 1. updating D
    Z_norm_inf =norm(Z(:), inf);
    if (sum(Omega(:)) == d * n)        
        D = X; % If there is no missing entries, i.e. |Omega| = n*d, it is smart to let D = X;
    else        
        I_Z = In - Z;
        I_Z_norm_two =(normest(I_Z,0.1))^2;

        eta =1.01 + I_Z_norm_two;
        temp =D + N - X - Y1/mu;
        Zeros(P_Omega) = temp(P_Omega_star_P_Omega);
        temp = D - (Zeros + Y2*( In - Z')/mu +(D - D*Z -E)*(In - Z'))/eta;

        for ii =1:k
            idx_ii =(idx ==ii);
            tmp_ii =temp(:,idx_ii);

            [U,sigma,V] = svd(tmp_ii,'econ');    

            sigma = diag(sigma);
            svp = length(find(sigma >= 1/(mu*eta)));
            if svp >= 1
                sigma = sigma(1:svp) - 1/(mu*eta);
            else
                svp = 1;
                sigma = 0;
            end        
            D(:,idx_ii) = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
        end
        
    end
    
   %% 2. updating N (However N may not be activated forever: in noise free case tau is Inf and it noise case we prefer to take a relaxion ||P_\Omega(D-X)|| rather than introducing N)
    eta =1;
    switch opt.norm_mc
        case '21'
            temp =(N + D - X - Y1/mu)/eta;
            Zeros(P_Omega) = temp(P_Omega_star_P_Omega);
            temp = N-Zeros;
            N = solve_l1l2(temp, tau/(mu*eta));

        case 'fro' % Need a check...
            temp =(N + D - X - Y1/mu)/eta;
            Zeros(P_Omega) = temp(P_Omega_star_P_Omega);
            N = (mu/(2*tau + mu*eta)) *(eta*N - Zeros);

        case '1'
                temp =(N + D - X - Y1/mu)/eta;
                Zeros(P_Omega) = temp(P_Omega_star_P_Omega);
                temp = N - Zeros;
                N = max(0,temp - tau/(mu*eta)) + min(0,temp + tau/(mu*eta));
    end
    
    %% Solving Z and E by linearized ADM: Structured Sparse Representation
    
    D_norm_inf =norm(D(:), inf);
    
    D_norm_two =normest(D, 0.1);
    xi =D_norm_two^2 +1.01;    

    %% 3. updating Z: simultaneously and separately 
    temp = Z - ( D' * ( D*Z + E - D - Y2/mu ) ) /xi;

    %% 1) without structured sparse penalty entries: gamma0
    % Z = temp;        
    % Z = max(0 , temp - gamma0/(mu * xi) ) + min(0 , temp + gamma0/(mu * xi));    

    %% THE FOLLOWING STEP MAKEs S3LR converges much better than without 1/Gamma. 
     %         %Z = max(0 , temp - gamma0/(Gamma * mu * xi) ) + min(0 , temp + gamma0/(Gamma * mu * xi)); 
     %         Z = max(0 , temp - gamma0/(mu * xi) ) + min(0 , temp + gamma0/(mu * xi));
     %% 2) with the structured sparse penalty, i.e. structured sparsifying Z
     %         % temp = Z - ( D' * ( D*Z + E - D - Y2/mu ) ) / xi;
     %         temp = max(0 , temp - Gamma/(mu*xi) ) + min(0 , temp + Gamma/(mu*xi));
     %         Z(P_Theta_star_P_Theta) = temp(P_Theta_star_P_Theta);         

    Z = max(0, temp - ( Gamma * Theta + gamma0) / (mu * xi) ) + min(0, temp +  ( Gamma * Theta + gamma0) /(mu* xi) );       

    Z = Z - diag(diag(Z)); % to force Zii = 0
    
    %% 4. updating E
    temp = D - D*Z + Y2/mu;
    switch opt.norm_sr
        case '21'
            E = solve_l1l2(temp,lambda/mu); 

        case 'fro'
            E = (mu /(2*lambda + mu))*temp;

        case '1'
            E = max(0,temp - lambda/mu) + min(0,temp + lambda/mu);
    end
    if (affine)
        E(1,:) = e_zeros; % used for the problem with affinity constraint when augmenting D with a full-one row
    end
    
    %% Checking convergence conditions:   
    temp = X -D -N;
    leq1 = temp(P_Omega);        
    stopC1 =max(abs(leq1(:)))/X_norm_inf;
    leq2 = D - D*Z - E;
    stopC2 =max(abs(leq2(:)))/D_norm_inf; 
   
    %% Checking whole looping convergence conditions:
    leq3 = Z_old - Z;
    leq4 = D_old - D;
    stopC3 =max(abs(leq3(:)))/Z_norm_inf;  
    stopC4 =max(abs(leq4(:)))/D_norm_inf; 
    
    %% FOR TEST ONLY:
    if (TEST_CODE) 
        err1 =[err1, stopC1];
        plot(err1,':+r');hold on; drawnow;
        figure(h);
        err2 =[err2, stopC2];
        plot(err2,'-ob'); drawnow;hold on;
        err3 =[err3, stopC3];figure(h);plot(err3,'-r');hold on; drawnow;
        err4 =[err4, stopC4];figure(h);plot(err4,'-b'); drawnow;hold off;
        
        if iter==1 || mod(iter,50)==0 || (stopC1<tol && stopC2 < tol) 
            rankD =rank(D,1e-3*norm(D,2));
            disp(['iter ' num2str(iter) ', mu=' num2str(mu,'%2.1e'), 'mu*eta=  ',num2str(mu*eta),', rankD=' num2str(rankD) ', stopLADM=' num2str(stopC1,'%2.3e')]);
            stat.iter =iter;
            stat.rank1 =rankD;
            stat.stopLALM1 =stopC1;
            rankZ =rank(Z,1e-3*norm(Z,2));
            disp(['iter ' num2str(iter) ', mu=' num2str(mu,'%2.1e'), 'mu*eta=  ',num2str(mu*eta),', rankZ=' num2str(rankZ) ', stopLALM=' num2str(stopC2,'%2.3e')]);
            stat.rank2 =rankZ;
            stat.stopLALM2 =stopC2;
        end
    end
    
    %% Checking convergence conditions:
    if (stopC1 < tol) && (stopC2<tol) ||(stopC3 < tol) && (stopC4<tol) || mu > mu_max %1e4%mu >1e6  % protect the over-looping|| mu >1e3;
        break;
    else
        
   %% Updating Y1, Y2, and mu
        temp =X - D - N;
        Zeros(P_Omega) = temp(P_Omega_star_P_Omega);
        Y1 = Y1 + mu * Zeros;
        Y2 = Y2 + mu * (D - D*Z -E);
               
        %% Updating strategy for the penalty paramter \mu
        mu = mu * rho;
        
        %% Adaptive updating strategy for the penalty paramter \mu
        %if ( mu * max( eta^0.5 * max(abs(D - D_old)), xi^0.5 * max(abs(Z - Z_old)))/X_norm_inf <= epsilon)
        %    mu = mu * rho;
        %end
        D_old =D;
        Z_old =Z;
        
    end    
end

function [idx]= find_idx_from_Theta(M,k,thresh)
if nargin<3
    thresh=0.5;
end
logic_M =(M >thresh);
n =size(M,2);
id =1:max(k,n);
% Find the k connected components in the graph
markers =ones(1,n);
%idx_cell =cell(1,k);
idx =zeros(1,n);
c =0;
%% Detect the biggest groups and assign........
for i=1:n
    if (markers(i)) % find the index which are still turn-on.
        c =c+1;
        j_idx =find(logic_M(i,:)==1);
        idx(j_idx) =id(c);
        %idx_cell{c} =j_idx;
        markers(j_idx) =0; % turn-off all markers which are connecting with i
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