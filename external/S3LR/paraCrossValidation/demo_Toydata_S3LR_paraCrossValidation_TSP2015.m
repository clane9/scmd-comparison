% demo_Toydata_S3LR_paraCrossValidation_TSP2015.m
%% paraCrossValidation.m
% Description: This code is for finding the proper parameters (e.g. \gamma0, lambda, and k) in S3LR with cross-validation
% strategy, in which a small proportion entries (e.g. 5%) are held-out and evaluate the matrix completion error on these
% held-out entries in order to determine the proper parameter.
%
%       Step 1: randomly sample p% entries from Omega as the held-out entries, denoted as Omega0
%       Step 2: perform matrix completion on Omega \ Omega0
%       Step 3: evaluate the completion error on Omega0
%       Step 4: change parameter and repeat Step 3 and Step 4
%
%
% CHUNGUANG LI
% Date: Feb. 12, 2014
% Modified: Oct. 31,2014, Jan 27, 2015
function [] = demo_Toydata_S3LR_paraCrossValidation_TSP2015()
addpath('../lrr');
addpath('../inexact_ladm_mc');
addpath('../local_comp_represent');
addpath('../toy_data');
addpath('../inexact_alm_mc/PROPACK');
addpath('../inexact_alm_mc');
addpath('../');
addpath('../Evaluation_on_MotionSegmentation/SSC_ADMM_v1.1');
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

missLevel_arr =[30]; % percentage of missing entries
numCases =length(missLevel_arr);

%% lambda0 and gamma0:
lambda_arr =[1 2 3 4 5 10 20 30 50];
lambda_arr =[5 10 15 20 30];
gamma_arr =[0.01 0.02 0.03 0.04];

num_lambda =length(lambda_arr);
num_gamma =length(gamma_arr);

ACC =zeros(num_lambda, num_gamma, numCases);
ACC_STD =zeros(num_lambda, num_gamma, numCases);

MC_ERR =zeros(num_lambda, num_gamma, numCases);
MC_ERR_STD =zeros(num_lambda, num_gamma, numCases);

MC_ERR_OUT =zeros(num_lambda, num_gamma, numCases);
MC_ERR_OUT_STD =zeros(num_lambda, num_gamma, numCases);

%%
x = 0:5:5*(missingLevel-1);
rstStr =[Str(6:end-4),'_paraCV_results_',datestr(now,30),'S3LR.mat'];
tic;

for ii = 1: numCases
    missingLevel_ii = fix( missLevel_arr(ii) / 5) + 1;    
    for jj =1: num_lambda
        for kk =1: num_gamma
            ob_rate = 1 - 5* (missingLevel_ii - 1)/100;
            lambda0 =lambda_arr(jj)*(ob_rate^5);
            gamma0 = gamma_arr(kk);
            [acc, ~, mc_err, acc_std, mc_err_std, ~, mc_err_out, mc_err_out_std] = test_seg(data, missingLevel_ii, lambda0, gamma0); % SLR with Missing Value <first update Z then D>
    
            ACC(jj, kk, ii) =acc;
            ACC_STD(jj, kk, ii) =acc_std;
            MC_ERR(jj, kk, ii) =mc_err;
            MC_ERR_STD(jj, kk, ii) =mc_err_std;
            MC_ERR_OUT(jj, kk, ii) =mc_err_out;
            MC_ERR_OUT_STD(jj, kk, ii) =mc_err_out_std;
            
            save(rstStr,'missLevel_arr','missingLevel','x','ACC','ACC_STD','MC_ERR','MC_ERR','MC_ERR_STD','MC_ERR_OUT',...
                'lambda_arr', 'gamma_arr', 'MC_ERR_OUT_STD');
    
            disp(['**iter: ii =', num2str(ii), ' jj =', num2str(jj),' kk=', num2str(kk), ': remainding time: ', ...
                num2str((numCases - ii +1) * ( num_lambda - jj + 1) * ( num_gamma - kk)*toc / 60), ' min.']);
            tic;
        end
    end
end

% load str_save_results;
% surf(ACC(:,:,1));figure;surf(MC_ERR(:,:,1));figure;surf(MC_ERR_OUT(:,:,1));
% show_paraCrossValidation(lambda_arr, gamma_arr, MC_ERR_OUT, 'MC ERR OUT');

function [data] = generate_data(n,d_arr,D,nS,nRound, missingLevel)
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
nX = size(X,2);

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

function [acc, X_fill, mc_err,acc_std, mc_err_std, Cell_iterStatus, mc_err_out, mc_err_out_std] = test_seg(data, id, lambda0, gamma0)
idx0 = data.cids;
X_c = data.Xs(id,:);
Omega_c =data.Omega(id,:); 
X =data.X;
clear data;

opt.tol =1e-5;
opt.maxIter =1e3;
opt.rho =1.1;
opt.mu_max = 1e6;

%%
nRound =size(X_c,2);
mc_err =0;
mc_err_out =0;
acc_tmp =zeros(1,nRound);
mc_err_tmp =zeros(1,nRound);
mc_err_out_tmp =zeros(1,nRound);
acc = 0;
Cell_iterStatus =cell(1, nRound);
for i=1:nRound
    Xi = X_c{i};
    Omega_i =Omega_c{i};
    %Xi_0 =X_0fill{i};
    [m,n] =size(Omega_i);
    
    
    %% parameter selection via cross validation on p% held-out entries
    %       Step 1: randomly sample p% entries from Omega_i as the held-out entries, denoted as Omega_i_out
    %       Step 2: perform matrix completion on Omega_i \ Omega_i_out, denoted as Omega_i_in
    %       Step 3: evaluate the completion error on Omega_i_out
    %       Step 4: change parameter and repeat Step 3
    
    %% paraCrossValidation
    tmp_idx =find(Omega_i >0);
    tmp_idx_perm = randperm(length(tmp_idx));
    
    % held-out entries to perform completion and evaluate completion errors
    cv_percent =0.05;
    Omega_i_out =zeros(m, n);    
    Omega_i_out( tmp_idx( tmp_idx_perm( 1 : fix(m * n * cv_percent) ) ) ) =1;
    
    % left observed entries to perform completion
    Omega_i_in =zeros(m, n);
    Omega_i_in( tmp_idx( tmp_idx_perm( fix(m * n * cv_percent) + 1 : end) ) ) =1;
    
    
    % Using LRMC for initialization
    % Xi = Xi_fill;  
    lambda =lambda0;
    Gamma =gamma0;
    gamma0 =gamma0 * sum(sum(Omega_i_in))/(size(Omega_i_in,1)*size(Omega_i_in,2));

     relax =1; % {1, 2}
     affine =0;

    opt.norm_sr ='1';
    opt.norm_mc ='1';
    opt.mu_max = 1e8;
    opt.tol =1e-5;
    opt.maxIter =2e3; 

    opt.assemble =1;

    t_max =30; %10;%6 %30;
    T =11;%11;%19;
    nu1 =1.0; %1/1.5;%1.5; % 2;
    nu2 =1.0;%1.2;%1.5; % 2;
    % tau = Inf; % (1e0) 1e1, 1e2, 1e3 
    % lambda_max = Inf; Gamma_max = Inf;
    [acc_i, mc_err_i, ~, X_fill, iterStatus] =S3LR(Xi, Omega_i_in, idx0, X, lambda, Gamma, gamma0, relax, affine, t_max, opt, T, nu1, nu2);     

    tmp =Xi - X_fill;         
    Omega_i_out =logical(Omega_i_out);
    mc_err_out_i = norm(tmp(Omega_i_out),'fro') / (norm(Xi,'fro'));

    Cell_iterStatus{1, i} =iterStatus; 
       
    %% Record middle results:
    mc_err = mc_err + mc_err_i;    
    mc_err_out = mc_err_out + mc_err_out_i;
    mc_err_tmp(i) =mc_err_i;
    mc_err_out_tmp(i) =mc_err_out_i;
    acc = acc + acc_i;
    acc_tmp(i) =acc_i;
    
end
acc = acc/nRound;
mc_err =mc_err / nRound;
mc_err_out =mc_err_out / nRound;
mc_err_std =std(mc_err_tmp);
mc_err_out_std =std(mc_err_out_tmp);
acc_std = std(acc_tmp);
disp(['acc = ' num2str(acc),',  acc_std = ' num2str(acc_std),' mc_err = ' num2str(mc_err),', mc_err_std = ' num2str(mc_err_std),...
    ' mc_err_out = ' num2str(mc_err_out),', mc_err_out_std = ' num2str(mc_err_out_std)]);

function y = XOmega(X,I,J,col)
y = zeros(length(I), 1);
for k = 1:length(col)-1
    j = J(col(k)+1);
    Xj = X(:,j)';
    idx = col(k)+1:col(k+1);
    y(idx) = Xj(I(idx));
end