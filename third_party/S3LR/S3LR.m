% S3LR.m
% Description: This code is for High-Rank Matrix Completion with Structured Sparse and Structured Low-Rank approach which is
% alternatively solving the following models:
%       While (~not converge)
%          Step 1:     arg min   \sum ||D||_*,i + lambda*||E||_1 + || (Gamma * Theta + gamma0* 11^T) \odot Z||_1 +  tau*||N||_fro
%                      s.t. P_\Omega(X) = P_\Omega (D + N), 
%                           D = D*Z + E
%                          (if affine, then  1^T =1^T Z)
%          Step 2:    spectral (assemble) clustering to find \Theta
%      end
%
%    [acc_i, mc_err_i, Z, X_fill] =S3LR(X, Omega, idx, X0, lambda, Gamma, gamma0,  affine, t_max, opt, T, nu, tau, lambda_max, Gamma_max)
%
%    Inputs:  X - incomplete data matrix;                
%                 Omega - logical matrix to indicate the observation entries;
%                 idx - data points groundtruth label;    
%                 X0  - complete data matrix
%                 relax ={1, 2} where '1' if  the relaxation of structured nuclear norm is only for the clustering part, and '2' if for both D and Theta.     
%                 affine ={1, 0} if the affine representation is needed.      
%                 t_max - the maximal number of outer loop of S3LR
%                 nu1 -  an increasing factor for lambda, nu1 >=1, e.g. 1.1, 1.2, 1.5, 2, which is used to increase the lambda
%                 nu2 -  an increasing factor for Gamma, nu1 >=1, e.g. 1.1, 1.2, 1.5, 2, which is used to increase the Gamma
%                           By default, nu1=nu2=1. 
%
% Copyright by Chun-Guang Li      
% Date: Sept 18, 2013
function [acc_i, mc_err_i, Z, X_fill, iterStatus] =S3LR(X, Omega, idx, X0, lambda, Gamma, gamma0, relax, affine, t_max, opt, T, nu1, nu2, tau, lambda_max, Gamma_max)
if nargin < 7
    gamma0 =0;
end
if nargin < 8        
    relax =1;
end
if nargin < 9        
    affine =0;
end
if nargin < 10        
    t_max =50; % for S3LR's outer loop
end
if nargin < 11        
    opt.tol =1e-4;
    opt.maxIter =1e6;
    opt.rho =1.1;
    opt.mu_max =1e4;
    opt.norm_sr ='1';
    opt.norm_mc ='1';
end
if nargin < 12        
    T =1; % T=1 for spectral clustering and T>2 for spectral assumble clustering
end
if nargin < 13
    nu1 =1.2; % or nu1 =1.0-2.0,  e.g. 1.1, 1.2, 1.5, 2, which is used to increase the lambda and Gamma
end
if nargin < 14
    nu2 =1.0; % or nu2 =1
end
if nargin < 15
    tau =Inf;
end
if nargin < 16
    lambda_max =Inf; 
    Gamma_max =Inf;
end

%% 
if isfield(opt,'assemble')
    SC_assemble =opt.assemble;
else
    SC_assemble =0;
end
% if isfield(opt,'multiSC')
%     SC_multi =1;
% end
% if isfield(opt,'warmSC')
%     SC_warm =1;
% end

%% Initialization
nbcluster =max(idx);  % the number of clusters is given, or it need to be estimated.
Theta = zeros(size(X,2));
Z = zeros(size(X,2));
X = zeros(size(X)) + X;
X_fill = X; %zeros(size(X));
E = zeros(size(X));

X_norm_two =normest(X, 0.1);
opt.mu =1.25/X_norm_two;

%% FOR TEST ONLY
TEST_CODE =0;
if (TEST_CODE)
    h =figure('name','HRMC'); set(gcf,'Position',[50,200,900,440]);
    g =figure;    
    
    mc_err =[];
    Z_off_diag =[];
    delta_D =[];
    delta2_D =[];
    %delta_mc_err =[];
    E_norm =[];    
end

X_fill_old = X_fill;
Theta_old =ones(size(X,2));
deltaD_old =2;
E_old = X +zeros(size(X));

lambda_t =lambda;
Gamma_t =Gamma;

obj = zeros(1, t_max); 
[~, s, ~] =svd(X_fill,'econ');   
obj(1) = sum(diag(s)) + sum(sum( ( Gamma_t * Theta + gamma0).*(abs(Z)))) + lambda_t * sum(sum(abs(E)));    
mcERR =zeros(1, t_max); 
mcERR(1) =norm(X0 - X_fill,'fro')/norm(X0,'fro');
cluACC =zeros(1, t_max); 
cluACC(1) =1/nbcluster;

if (TEST_CODE==1)
    h1 =figure;
    h2=figure;
    h3=figure;
end

%% main loop of S3LR
t=0;
while (t < t_max)     
    
    t = t+1;
    % parameters updating
    lambda_t =min(lambda_t * nu1, lambda_max); % lambda0 =0.05, lambda_max =0.15, rho=1.1 : ExYaleB_640: 32x32, 16x16
    Gamma_t = min(Gamma_t * nu2, Gamma_max);% beta0 =0.1, beta_max =2, rho=1.2 : ExYaleB_640_1024, 32x32,16x16
    
    %% Step I: Structured Matrix Completion and Structured Sparse Representation
    [Z, X_fill, E] =SMC_StrSR(X, Omega,Theta, lambda_t, Gamma_t, gamma0, tau, opt, X_fill, Z,  relax, affine);
    
    % colormap jet;image(abs(Z)*500); colorbar;
    
    %% Step II: Structure Estimation via Spectral Assemble Clustering
        
    %% without warmstart
    %grps = SpectralClustering(CKSym, max(idx));  missrate = Misclassification(grps, idx');
    
    %% with 'warmstart' which we initialize the next kmeans with the previous clustering result
    % it call our modified kmeans.
    % if (SC_warm)
    %     grps = SpectralClustering(CKSym, nbcluster, grps); missrate = Misclassification(grps, idx'); 
    % end
    
    if (SC_assemble)
        rho =1;CKSym = BuildAdjacency(thrC(Z,rho));
        [acc_i, Theta, ~, ~] = SpectrAssembleClustEvaluation(CKSym, idx, nbcluster, T);
    else
    % [acc_i, ~, Theta] =SpectrClustEvaluation(CKSym, idx, nbcluster, T);
    % [acc_i, ~, Theta] =SpectrClustEvaluation(Z, idx, nbcluster, T);
    
    %% with 'warmstart' from the previous 'Theta':
    % if (SC_multi_warm)
        [acc_i, ~, Theta] =SpectrClustEvaluation(Z, idx, nbcluster, T, 1e-40, 1-Theta);
    end
    
    Theta =1-Theta;    
    if (~affine)
        mc_err_i = norm(X0 - X_fill,'fro')/norm(X0,'fro');
    else
        tmp = X0 - X_fill;
        mc_err_i = norm(tmp(2:end, :),'fro')/norm(X0(2:end, :),'fro');
    end
    
    [~, s, ~] =svd(X_fill,'econ');
    obj(t+1) = sum(diag(s)) + sum(sum( ( Gamma_t * Theta + gamma0).*(abs(Z)))) + lambda_t * sum(sum(abs(E)));        
    mcERR(t+1) =mc_err_i;
    cluACC(t+1) =acc_i;
    if (TEST_CODE ==1)
        figure(h1);plot(obj(2:end),':vr','LineWidth',2); drawnow;   
        figure(h2);plot(mcERR(2:end),':sg','LineWidth',2); drawnow;     
        figure(h3);plot(cluACC(2:end),':ob','LineWidth',2); drawnow;   
    end

    %if (TEST_CODE)
        disp(['t = ',num2str(t),', acc = ',num2str(acc_i),', mc_err = ',num2str(mc_err_i), ', obj. : ', num2str(obj(t+1))]);
    %end
    
    %% Stop criterion checking    
    deltaD =norm(X_fill - X_fill_old,'fro')/norm(X_fill,'fro');
    stop_criterion_D =deltaD_old / (deltaD + (deltaD < 1e-20));
    stop_criterion_theta = norm(Theta -Theta_old, 'fro')/(norm(Theta,'fro') + (norm(Theta,'fro')<1e-20));
    stop_criterion_E = norm(E,'fro') - norm(E_old,'fro');
    stop_criterion_E0 = norm(E,'fro');
    stop_criterion_Z =norm( Theta.*Z, 'fro');

    %% FOR TEST ONLY:
    if (TEST_CODE)
        if (mod(t,2)==1)
            figure(h);
            subplot(4,4,(t+1)/2); 
            imshow(abs(Z)*150);colorbar;colormap jet;
        end        
        mc_err =[mc_err, mc_acc_i];figure(g);plot(mc_err,':og','LineWidth',2); drawnow;hold on; 
        Z_off_diag =[Z_off_diag, stop_criterion_Z];figure(g);plot(Z_off_diag,'-sb','LineWidth',2); drawnow;hold on;
        delta_D =[delta_D, deltaD];figure(g);plot(delta_D,'-vr','LineWidth',2); drawnow;hold on;
        delta2_D =[delta2_D, stop_criterion_D];figure(g);plot(delta2_D,':^m','LineWidth',2); drawnow;hold on;    
        delta_mc_err = -([mc_err, mc_err(end)] - [1,mc_err])./[1,mc_err];  figure(g);
        plot(delta_mc_err,':vg','LineWidth',2); drawnow;hold on;   
        E_norm =[E_norm, stop_criterion_E0];plot(E_norm,':^k','LineWidth',2); drawnow; hold off;     
    end
    
    if (TEST_CODE)
        disp(['delta D: ',num2str(deltaD),', stop_D: ',num2str(stop_criterion_D),...
        ', stop_dealtTheta: ', num2str(stop_criterion_theta),...
        ', stop_|Theta.*Z|: ', num2str(stop_criterion_Z),...
        ', stop E:',num2str(stop_criterion_E),' norm(E): ',num2str(norm(E,'fro'))]);
    end
    
    %if ( (stop_criterion_Z < 1e-8) && (stop_criterion_D < 1.0) && (stop_criterion_theta <1e-8 || deltaD <1e-8)|| stop_criterion_E >1e-20)
    %if ( (stop_criterion_Z < 1e-8) && (stop_criterion_theta <1e-8 || deltaD <1e-8)|| stop_criterion_E >1e-20)
    %if ( (stop_criterion_Z < 1e-8) && (stop_criterion_theta <1e-8 || deltaD <1e-8) && norm(E,'fro') <1e-8)        
    if (~affine)
        tol_Z =1.5e-3;
        tol_D =1e-5;
    else
        tol_Z =5e-3;
        tol_D =5e-3;
    end
    % 
    %if ((stop_criterion_Z < tol_Z) || (stop_criterion_Z < tol_Z) && (stop_criterion_theta <1e-4 || deltaD < tol_D))            
    if ((stop_criterion_Z < tol_Z) || (deltaD < tol_D))            

            disp(['* * * * * * STOP @ t = ',num2str(t),'. acc = ',num2str(acc_i),', mc_err = ',num2str(mc_err_i)]);
            
            break;
    end
    deltaD_old =deltaD;
    X_fill_old =X_fill;
    Theta_old = Theta;
    E_old =E;
end
if (TEST_CODE)
    str ={'MC err';'|Z|_{1,\Theta}';'\delta (D)';'\Delta (D)';'\delta (MC err)';'norm(E)'};
    legend(str);
end
iterStatus.obj = obj;
iterStatus.mcERR = mcERR;
iterStatus.cluACC =cluACC;
%% MC error curves nLevel=11
% figure;
% zero_fill =ones(1,18)*0.708; 
% plot(zero_fill,':k','LineWidth',2); hold on; 
% lrmc =ones(1,17)*0.514;
% plot(lrmc,'-r','LineWidth',2); hold on;
% mc_err =[0.708, mc_err];plot(mc_err,'-ob','LineWidth',2); hold on; 
% legend({'zerofill';'low-rank MC';'S^3LR';});