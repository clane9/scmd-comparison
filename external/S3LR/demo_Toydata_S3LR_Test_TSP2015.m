% demo_Toydata_S3LR_Test_TSP2015.m
% Description: This demo compares the performance of the following algorithms for Matrix Completion in a Union of Subspaces
% with the following algorithms:
%         1. Structured Sparse Structured Low-Rank (S3LR)
%         2. MC + LRR
%         3. MC + SSC
%         4. Zero-filled + SSC
%         5. zero-filled + LRR
%         6. S3LR: S3LR-QZ
%         7. k-SMC: S3LR-QD, i.e. Alternating SMC + Subspace clustering
%
%  We generate nS subspaces which of dimension d_i <= d, and having n samples per subspace.  We randomly choose a proportion
%  of data points entries to be missing value.
%
% CHUNGUANG LI
% Date: April 14, 2014
% Modified by August 1, 2015
function [] = demo_Toydata_S3LR_Test_TSP2015()
% addpath('../lrr');
% addpath('../inexact_ladm_mc');
% addpath('../local_comp_represent');
% addpath('../toy_data');
% addpath('../inexact_alm_mc/PROPACK');
% addpath('../inexact_alm_mc');
% addpath('../');
% addpath('../Evaluation_on_MotionSegmentation/SSC_ADMM_v1.1');

%% Data Setting
n = 10; % number of data points in each subspace
nS =20; % number of subspaces
%d_arr = [1:5]; % The dimensions of each subspace
d_arr = 5*ones(1,nS);%[5 5 5 5 5]; % The maximum dim of subspace
D = 100; % dim of observation
nRound = 10; % repeat times of random trials: 5, 10
missingLevel =19; % from 1 to 21 : completely observed to completely missing, at interval of 5%.

%% 
newTest =0;

if (newTest==1)
    data = generate_data(n,d_arr,D,nS,nRound,missingLevel);
    Str =['save data_Noiseless_Incompleted_LowRank_n',num2str(n),'_mind',num2str(min(d_arr)),'_maxd',num2str(max(d_arr)),'_D',num2str(D),'_nS',num2str(nS),'_nRound',num2str(nRound),'_missingLevel',num2str(missingLevel),'.mat'];
else
    Str =['load ','data_Noiseless_Incompleted_LowRank_n',num2str(n),'_mind',num2str(min(d_arr)),'_maxd',num2str(max(d_arr)),'_D',num2str(D),'_nS',num2str(nS),'_nRound',num2str(nRound),'_missingLevel',num2str(missingLevel),'.mat'];
end
eval(Str);

acc1=zeros(1,missingLevel); % zero-filled X + SSC
acc2=zeros(1,missingLevel); % zero-filled X + LRR
acc3=zeros(1,missingLevel); % low-rank matrix completion (with ||E||_2,1) + SSC 
acc4=zeros(1,missingLevel); % low-rank matrix completion (with ||E||_2,1) + LRR
acc5=zeros(1,missingLevel); % low-rank matrix completion with ||E||_1 + SSC
acc6=zeros(1,missingLevel); % low-rank matrix completion with ||E||_1 + LRR
acc7=zeros(1,missingLevel); % low-rank matrix completion with ||E||_F + SSC
acc8=zeros(1,missingLevel); % low-rank matrix completion with ||E||_F + LRR
%  acc24=zeros(1,missingLevel); % S3LR - QD (SMC + StrSSC)
acc25=zeros(1,missingLevel); % S3LR -QZ

% acc26=zeros(1,missingLevel); % k-SMC (Alt.MC+SC): SR
% acc27=zeros(1,missingLevel); % k-SMC (Alt.MC+SC): LRR
% acc28=zeros(1,missingLevel); % k-SMC (Alt.MC+SC): SSC_fro
% acc29=zeros(1,missingLevel); % k-SMC (Alt.MC+SC): SSC_outlier

n_alg =29;
Acc_arr =zeros(n_alg,missingLevel);
Acc_std_arr =zeros(n_alg,missingLevel);
Acc_std_ii_arr =zeros(n_alg,missingLevel);
MC_Acc_arr =zeros(n_alg,missingLevel);
MC_Acc_std_arr =zeros(n_alg,missingLevel);
cell_iter =cell(1, missingLevel);

%%
x = 0:5:5*(missingLevel-1);
rstStr =[Str(6:end-4),'results',datestr(now,30),'S3LR.mat'];

tic;
for i = 1:2:missingLevel
    acc =0; mc_acc =0; acc_std =0; mc_acc_std =0;acc_std_ii=0;
    lambda =my_lambda_new(x,1, i);
%     [acc, ~, mc_acc, acc_std, mc_acc_std] = test_seg(data,i,lambda,1); % SSC on zero-filled data
    acc1(i) =acc;
    disp(['acc1 = ',num2str(acc1)])   
    Acc_arr(1,i)=acc;
    MC_Acc_arr(1,i)=mc_acc;
    Acc_std_arr(1,i)=acc_std;
    Acc_std_ii_arr(1,i)=acc_std_ii;
    MC_Acc_std_arr(1,i)=mc_acc_std;
    acc =0; mc_acc =0; acc_std =0; mc_acc_std =0;acc_std_ii=0;
    lambda =my_lambda_new(x,2, i);
  % [acc, ~, mc_acc, acc_std, mc_acc_std] = test_seg(data,i,lambda,2); % LRR on zero-filled data
    acc2(i) =acc;
    disp(['acc2 = ',num2str(acc2)])  
    Acc_arr(2,i)=acc;
    MC_Acc_arr(2,i)=mc_acc;
    Acc_std_arr(2,i)=acc_std;
    Acc_std_ii_arr(2,i)=acc_std_ii;
    MC_Acc_std_arr(2,i)=mc_acc_std;

    lambda =my_lambda_new(x,3, i);
    acc =0; mc_acc =0; acc_std =0; mc_acc_std =0;acc_std_ii=0;
    %[acc, ~, mc_acc, acc_std, mc_acc_std] = test_seg(data,i,lambda,3); % MC21 +SSC
    acc3(i) =acc;
    disp(['acc3 = ',num2str(acc3)])
    Acc_arr(3,i)=acc;
    MC_Acc_arr(3,i)=mc_acc;
    Acc_std_arr(3,i)=acc_std;
    Acc_std_ii_arr(3,i)=acc_std_ii;
    MC_Acc_std_arr(3,i)=mc_acc_std;
    acc =0; mc_acc =0; acc_std =0; mc_acc_std =0;acc_std_ii=0;
    lambda =my_lambda_new(x,4, i);
    %[acc, ~, mc_acc, acc_std, mc_acc_std] = test_seg(data,i,lambda,4); % MC21 +LRR
    acc4(i) =acc;
    disp(['acc4 = ',num2str(acc4)])
    Acc_arr(4,i)=acc;
    MC_Acc_arr(4,i)=mc_acc;
    Acc_std_arr(4,i)=acc_std;
    Acc_std_ii_arr(4,i)=acc_std_ii;
    MC_Acc_std_arr(4,i)=mc_acc_std;    
    
    lambda =my_lambda_new(x,5, i);
    %[acc, ~, mc_acc, acc_std, mc_acc_std] = test_seg(data,i,lambda,5); % MC1 +SSC
    acc5(i) =acc;
    disp(['acc5 = ',num2str(acc5)])
    Acc_arr(5,i)=acc;
    MC_Acc_arr(5,i)=mc_acc;
    Acc_std_arr(5,i)=acc_std;
    Acc_std_ii_arr(5,i)=acc_std_ii;
    MC_Acc_std_arr(5,i)=mc_acc_std;    
    acc =0; mc_acc =0; acc_std =0; mc_acc_std =0;acc_std_ii=0;
    lambda =my_lambda_new(x,6, i);
    %[acc, ~, mc_acc, acc_std, mc_acc_std] = test_seg(data,i,lambda,6); % MC1 +LRR
    acc6(i) =acc;
    disp(['acc6 = ',num2str(acc6)])
    Acc_arr(6,i)=acc;
    MC_Acc_arr(6,i)=mc_acc;
    Acc_std_arr(6,i)=acc_std;
    Acc_std_ii_arr(6,i)=acc_std_ii;
    MC_Acc_std_arr(6,i)=mc_acc_std;    
    
    acc =0; mc_acc =0; acc_std =0; mc_acc_std =0;acc_std_ii=0;
    lambda =my_lambda_new(x,7, i);
   % [acc, ~, mc_acc, acc_std, mc_acc_std] = test_seg(data,i,lambda,7); % MCfro +SSC
    acc7(i) =acc;
    disp(['acc7 = ',num2str(acc7)])
    Acc_arr(7,i)=acc;
    MC_Acc_arr(7,i)=mc_acc;
    Acc_std_arr(7,i)=acc_std;
    Acc_std_ii_arr(7,i)=acc_std_ii;
    MC_Acc_std_arr(7,i)=mc_acc_std;    
    acc =0; mc_acc =0; acc_std =0; mc_acc_std =0;acc_std_ii=0;
    
    lambda =my_lambda_new(x,8, i);
   %[acc, ~, mc_acc, acc_std, mc_acc_std] = test_seg(data,i,lambda,8); % MCfro +LRR
    acc8(i) =acc;
    disp(['acc8 = ',num2str(acc8)])
    Acc_arr(8,i)=acc;
    MC_Acc_arr(8,i)=mc_acc;
    Acc_std_arr(8,i)=acc_std;
    Acc_std_ii_arr(8,i)=acc_std_ii;
    MC_Acc_std_arr(8,i)=mc_acc_std;  
           
    lambda =my_lambda_new(x,24, i);
  % [acc, ~, mc_acc, acc_std, mc_acc_std] = test_seg(data,i,lambda, 24); % SLR - QD (SMC + StrSSC)
    acc24(i) =acc;
    disp(['acc24 = ',num2str(acc24)])
    Acc_arr(24,i)=acc;
    MC_Acc_arr(24,i)=mc_acc;
    Acc_std_arr(24,i)=acc_std;
    Acc_std_ii_arr(24,i)=acc_std_ii;
    MC_Acc_std_arr(24,i)=mc_acc_std; 
    
    lambda =my_lambda_new(x,25, i);
    [acc, ~, mc_acc, acc_std, mc_acc_std, Cell_iterStatus] = test_seg(data,i,lambda, 25); % SLR with Missing Value <first update Z then D>
    acc25(i) =acc;
    disp(['acc25 = ',num2str(acc25)])
    Acc_arr(25,i)=acc;
    MC_Acc_arr(25,i)=mc_acc;
    Acc_std_arr(25,i)=acc_std;
    Acc_std_ii_arr(25,i)=acc_std_ii;
    MC_Acc_std_arr(25,i)=mc_acc_std;     
    cell_iter{1, i} =Cell_iterStatus;
            
    save(rstStr, 'cell_iter','missingLevel','x','acc1', 'acc2','acc3','acc4','acc5','acc6','acc7','acc8',...
        'acc24','acc25','Acc_arr','MC_Acc_arr','Acc_std_arr','MC_Acc_std_arr','Acc_std_ii_arr');
    
    disp(['**Iteration: ' num2str(i) 'Estimated remainding time: ' num2str((missingLevel-i)*toc/60) ' minutes']); 
    tic;
end

figure;
x = 5*(1-1):5:5*(missingLevel-1);
subplot(121);
plot(x,acc1,':r',x,acc2,':b',x,acc3,'-dr',x,acc4,'-db',x,acc5,'-sr',x,acc6,'-sb',x,acc7,'-or',x,acc8,...
     '-ob',x,acc24,'-ob',x,acc25,'-og',x,acc26,'-+r',x,acc27,'-+g',x,acc28,'-+b',x,acc29,'-*b','LineWidth',2);
 legend('zero-fill SSC', 'zero-fill LRR','MC_{2,1} +SSC','MC_{2,1} +LRR','MC_{1} +SSC',...
     'MC_{1} +LRR','MC_{fro} +SSC','MC_{fro} +LRR','S^3LR(QD)','S^3LR(QZ)');
xlabel('percentage of missing value');
ylabel('segmentation accuracy');
subplot(122);
plot(x,MC_Acc_arr(1,:),':r',x,MC_Acc_arr(2,:),':b',x,MC_Acc_arr(3,:),'-dr',x,MC_Acc_arr(4,:),'-db',...
    x,MC_Acc_arr(5,:),'-sr',x,MC_Acc_arr(6,:),'-sb',x,MC_Acc_arr(7,:),'-or',x,MC_Acc_arr(8,:),...
     '-ob', x,MC_Acc_arr(24,:),'-+b',x,MC_Acc_arr(25,:),'-+r', 'LineWidth',2);
 legend('zero-fill SSC', 'zero-fill LRR','MC_{2,1} +SSC','MC_{2,1} +LRR','MC_{1} +SSC',...
     'MC_{1} +LRR','MC_{fro} +SSC','MC_{fro} +LRR','S^3LR(QD)','S^3LR(QZ)');
xlabel('percentage of missing value');
ylabel('error of matrix completion');

function [data] = generate_data(n,d_arr,D,nS,nRound,missingLevel)
% n = 50; % number of data points in each subspace
% d = 10; % dim of subspace
% D = 100; % dim of observation
% nS =10; % number of subspaces
% nRound = 10; % repeat times of random trials

[U,~,~] = svd(rand(D));
cids = [];
U_cell =cell(1,nS);
X_cell =cell(1,nS);
R = orth(rand(D));
X =[];
for ii=1:nS
    d =max(d_arr);
    if (ii==1)
        U_cell{ii} =U(:,1:d);
    else
        U_cell{ii} =R*U_cell{1,ii-1};
    end
    %% Generate subspace with different dimenion d_ii which is specified by d
    d_ii =d_arr(ii);
    tmp =U_cell{ii};
    X_cell{ii} =tmp(:, 1:d_ii)*rand(d_ii,n);
    
    cids = [cids,ii*ones(1,n)];
    X =[X,X_cell{ii}];
end

data.X = X;
data.cids = cids;
data.Xs = cell(missingLevel,nRound);% in 5% interval of proportion
data.Omega = cell(missingLevel,nRound);% in 5% interval of proportion
%data.X0fill = cell(21,nRound);% in 5% interval of proportion
nX = size(X,2);
%norm_x = sqrt(sum(X.^2,1));
%norm_x = repmat(norm_x,D,1);
for i=1:missingLevel
%    gn = norm_x.*randn(D,nX);
    for j=1:nRound
        m =D;
        n =nX;        
        Omega =zeros(m,n);
        p =ceil((1-(i-1)*5/100)*m*n);
        [I, J, col, omega] = myRandsample(m, n, p);        
        Omega(omega) =1;
        V = XOmega(X, I, J, col);
        X_Omega = spconvert([I,J,V; m,n,0]);
        % sparse matrix X_Omega
        data.Xs{i,j} = X_Omega;
        data.Omega{i,j} =Omega;
        %data.X0fill{i,j} =X.*Omega;
    end
end

function [acc, X_fill, mc_acc,acc_std, mc_acc_std, Cell_iterStatus] = test_seg(data,id,lambda,flag,X_fill)
%id = 2*id+1;
idx0 = data.cids;
nbcluster = length(unique(idx0));
X_c = data.Xs(id,:);
Omega_c =data.Omega(id,:); 
%X_0fill = data.X0fill(id,:);
X =data.X;
clear data;

opt.tol =1e-5;
opt.maxIter =1e3;
opt.rho =1.1;
opt.mu_max = 1e6;

%%
lambda_mc =Inf;

nRound =size(X_c,2);
mc_acc =0;
acc_tmp =zeros(1,nRound);
mc_err_tmp =zeros(1,nRound);
acc = 0;
%acc_std =0;
T =11;
Cell_iterStatus =cell(1, nRound);
for i=1:nRound
    Xi = X_c{i};
    Omega_i =Omega_c{i};
    %Xi_0 =X_0fill{i};
    [m,n] =size(Omega_i);
    switch flag
        case 1 % zero-fill SSC
            X_fill = zeros(m,n) + Xi;
            mc_acc_i = norm(X-X_fill,'fro')/norm(X,'fro');          
            alpha =lambda;
            r = 0; affine = false; outlier = true; rho = 1;
            [missrate,~] = SSC_compacc(X_fill, r, affine, alpha, outlier, rho, idx0');
            acc_i = 1 - missrate;                        
            
        case 2 % zero-fill LRR
            X_fill = zeros(m,n) + Xi;
            L = lrr(X_fill,lambda, opt);
            %L = fast_lrr_ladm(X_fill,lambda, opt);
            mc_acc_i = norm(X-X_fill,'fro')/norm(X,'fro');
            [acc_i,~] =SpectrClustEvaluation(L,idx0,nbcluster,T);
            
        case 3 %MC +SSC
            if nargin< 5          
                if (sum(Omega_i(:)) < 1+m*n)
                    %[X_fill, E, stat] = inexact_alm_mc(Xi, 1e-4);
                    Omega_i =logical(Omega_i);
                    [X_fill] = inexact_ladm_mc(Xi, Omega_i,lambda_mc,opt);
                    %[X_fill, E, stat] = inexact_ladm_mc(Xi, Omega_i,lambda,opt);
                else
                    X_fill =Xi;
                end

            end
            L = sr(X_fill,lambda,opt);
            mc_acc_i = norm(X-X_fill,'fro')/norm(X,'fro');
            [acc_i,~] =SpectrClustEvaluation(L,idx0,nbcluster,T);
        case 4 %MC +LRR
            if nargin< 5          
                if (sum(Omega_i(:)) < 1+m*n)
                    %[X_fill, E, stat] = inexact_alm_mc(Xi, 1e-4);
                    Omega_i =logical(Omega_i);
                    [X_fill] = inexact_ladm_mc(Xi, Omega_i,lambda_mc,opt);
                else
                    X_fill =Xi;
                end
            end
            L = lrr(X_fill,lambda, opt);
            mc_acc_i = norm(X-X_fill,'fro')/norm(X,'fro');
            [acc_i,~] =SpectrClustEvaluation(L,idx0,nbcluster,T);
            
        case 5 %MC_E1 +SSC
            if nargin< 5          
                if (sum(Omega_i(:)) < 1+m*n)
                    %[X_fill, E, stat] = inexact_alm_mc(Xi, 1e-4);
                    Omega_i =logical(Omega_i);
                    [X_fill] = inexact_ladm_mc_E1(Xi, Omega_i,lambda_mc,opt);
                else
                    X_fill =Xi;
                end
            end
            L = sr(X_fill,lambda,opt);
            mc_acc_i = norm(X-X_fill,'fro')/norm(X,'fro');
            [acc_i,~] =SpectrClustEvaluation(L,idx0,nbcluster,T);
            
        case 6 %MC_E1 +LRR
            if nargin< 5          
                if (sum(Omega_i(:)) < 1+m*n)
                    %[X_fill, E, stat] = inexact_alm_mc(Xi, 1e-4);
                    Omega_i =logical(Omega_i);
                    [X_fill] = inexact_ladm_mc_E1(Xi, Omega_i,lambda_mc,opt);
                else
                    X_fill =Xi;
                end
            end
            L = lrr(X_fill,lambda, opt);
            mc_acc_i = norm(X-X_fill,'fro')/norm(X,'fro');
            [acc_i,~] =SpectrClustEvaluation(L,idx0,nbcluster,T);
            
        case 7 %MC_E_fro +SSC
            if nargin< 5          
                if (sum(Omega_i(:)) < 1+m*n)
                    %[X_fill, E, stat] = inexact_alm_mc(Xi, 1e-4);
                    Omega_i =logical(Omega_i);
                    [X_fill] = inexact_ladm_mc_E_fro(Xi, Omega_i,lambda_mc,opt);
                else
                    X_fill =Xi;
                end
            end
            mc_acc_i = norm(X-X_fill,'fro')/norm(X,'fro');
            alpha =lambda;
            r = 0; affine = false; outlier = true; rho = 1;
            [missrate,~] = SSC_compacc(X_fill, r, affine, alpha, outlier, rho, idx0');
            acc_i = 1 - missrate;
            
        case 8 %MC_E_fro +LRR
            if nargin< 5          
                if (sum(Omega_i(:)) < 1+m*n)
                    %[X_fill, E, stat] = inexact_alm_mc(Xi, 1e-4);
                    Omega_i =logical(Omega_i);
                    [X_fill] = inexact_ladm_mc_E_fro(Xi, Omega_i,lambda_mc,opt);
                else
                    X_fill =Xi;
                end
            end
            L = lrr(X_fill,lambda, opt);
            mc_acc_i = norm(X-X_fill,'fro')/norm(X,'fro');
            [acc_i,~] =SpectrClustEvaluation(L,idx0,nbcluster,T);        
            
    case 24

        %% S3LR (QD): SMC + StrSSC
        opt.norm_mc ='1';
        opt.mu_max = 1e8;
        opt.tol =1e-5;
        opt.maxIter =2e3;
        opt.norm_mc ='1';
        opt.sc_method ='ssc_outliers'; % e.g., 'ssc', 'lrr', 'ssc_outliers'
        opt.lambda =lambda;

        t_max =30; %30;
        opt.iter_max =5; %10; %  iter_max is for loop in SSCE
        T =1;
        affine =0;
        tau = Inf; % (1e0) 1e1, 1e2, 1e3 
        [acc_i, mc_acc_i, ~, X_fill, iterStatus] = k_SMC(Xi, Omega_i, idx0, X, t_max, opt, T, affine, tau);  
        Cell_iterStatus{1, i} =iterStatus; 
          
        case 25
            %% Using LRMC for initialization
        if nargin ==5       
            Xi = Xi_fill;
        end
        Gamma =0.02;
        gamma0 =0.02 * sum(sum(Omega_i))/(size(Omega_i,1)*size(Omega_i,2));
         %gamma0 =0.1; %gamma0 =0.02 * sum(sum(Omega))/(size(Omega,1)*size(Omega,2));
         %gamma0 =0.025 * sum(sum(Omega_i))/(size(Omega_i,1)*size(Omega_i,2));
         %gamma0 =1 * sum(sum(Omega_i))/(size(Omega_i,1)*size(Omega_i,2));
         
         relax =1; % {1, 2}
         affine =0;

        opt.norm_sr ='1';
        opt.norm_mc ='1';
        %opt.mu_max = 1e8;
        opt.mu_max = 1e8;
        opt.tol =1e-5;
        opt.maxIter =2e3; 

        opt.assemble =1; % {1, 0}
        
        t_max =30; %10;%6 %30;
        T =11; %11;%19;
        nu1 =1.0; %1/1.5;%1.5; % 2;
        nu2 =1.0; %1.2;%1.5; % 2;
        % tau = Inf; % (1e0) 1e1, 1e2, 1e3 
        % lambda_max = Inf; Gamma_max = Inf;
        [acc_i, mc_acc_i, ~, X_fill, iterStatus] =S3LR(Xi, Omega_i, idx0, X, lambda, Gamma, gamma0, relax, ...
              affine, t_max, opt, T, nu1, nu2);      
          Cell_iterStatus{1, i} =iterStatus; 
            
        otherwise
            disp('Unknown Algorithm!');
    end
    mc_acc = mc_acc + mc_acc_i;
    mc_err_tmp(i) =mc_acc_i;
    acc = acc + acc_i;
    acc_tmp(i) =acc_i;
end
acc = acc/nRound;
mc_acc =mc_acc/nRound;
mc_acc_std =std(mc_err_tmp);
acc_std = std(acc_tmp);
%acc_std_ii =acc_std/nRound;
disp(['acc = ' num2str(acc),',  acc_std = ' num2str(acc_std),' mc_err = ' num2str(mc_acc),', mc_err_std = ' num2str(mc_acc_std)]);


function y = XOmega(X,I,J,col)
y = zeros(length(I), 1);
for k = 1:length(col)-1
    j = J(col(k)+1);
    Xj = X(:,j)';
    idx = col(k)+1:col(k+1);
    y(idx) = Xj(I(idx));
end

% function X_zerofill = ZeroFill_X_Omega(X,m,n)
% X_zerofill = zeros(m,n) + X;