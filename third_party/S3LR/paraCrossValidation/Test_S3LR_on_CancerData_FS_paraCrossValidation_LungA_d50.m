% Test_S3LR_on_CancerData_FS_paraCrossValidation_LungA_d50.m
%  This code is used to determine the proper parameters used in S3LR in order to evaluate the peformance of S3LR on the
%  dimension reduced data set, in which the dimesion is reduced by feature selection. The feature selection is to reduce the
%  measurements and to make the problem of high rank.
%
% Revised: August 26, 2015
% by Chun-Guang Li
function [] =Test_S3LR_on_CancerData_FS_paraCrossValidation_LungA_d50()
addpath('../lrr');
addpath('../inexact_ladm_mc');
addpath('../local_comp_represent');
addpath('../toy_data');
addpath('../inexact_alm_mc/PROPACK');
addpath('../inexact_alm_mc');
addpath('../');
addpath('../Evaluation_on_MotionSegmentation/SSC_ADMM_v1.1');
newTEST =0;
%DEBUG = 1;
str_load ='load LungA_local_zeromean_unitL2.mat';  % alpha =10, err: 0.05
eval(str_load);
m =50;
%FS_opt ='random';
FS_opt ='PCA_corr';

%% Generate data with missing values:
nRound =10;
missing_rates =[0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50];

%lambda_arr =[0.001, 0.01, 0.1, 0.5, 1, 2,3,4, 5, 6, 8, 10, 12, 15, 20,30,50,100, 200, 500];
%gamma_arr =[0.001, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5];

if (newTEST) 
    X =feature_selection(X, m, FS_opt);
    [data] = generate_missing_data(X, missing_rates,nRound);
    str_save_data =[str_load(6:end-4), '_FS_',FS_opt,'_m',num2str(m),'_data_nRound_',num2str(nRound),'_missingrate_',num2str(length(missing_rates)),'.mat'];
    save(str_save_data, 'data', 'index', 'X','FS_opt', 'nRound', 'missing_rates');
else
    str_save_data =[str_load(6:end-4), '_FS_',FS_opt,'_m',num2str(m),'_data_nRound_',num2str(nRound),'_missingrate_',num2str(length(missing_rates)),'.mat'];
    load(str_save_data, 'data', 'index', 'X', 'nRound', 'missing_rates');
end

str_save_results =[str_load(6:end-4), '_FS_m',num2str(m),'_paraCV_results',datestr(now,30),'.mat'];

missLevel_arr =[2  3]; % percentage of missing entries from "missing_rates"

%% lambda0 and gamma0:    
lambda_arr =[0.01, 0.1, 0.5, 1, 2,3,4, 5, 6, 8, 10, 12, 15, 20,30,50,100, 200];
gamma_arr =[0.001, 0.002, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5];
DEBUG=1;

if (DEBUG==1)
    % lambda_arr =[0.002, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0 2.0];
    % gamma_arr =[0.02, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2];        
    lambda_arr =[0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1.0 2.0];
    gamma_arr =[0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5];    
    missLevel_arr =[2 3]; % percentage of missing entries from "missing_rates"
end

numCases =length(missLevel_arr);
num_lambda =length(lambda_arr);
num_gamma =length(gamma_arr);

ACC =zeros(num_lambda, num_gamma, nRound, numCases);
%ACC_STD =zeros(num_lambda, num_gamma, numCases);

MC_ERR =zeros(num_lambda, num_gamma, nRound, numCases);
%MC_ERR_STD =zeros(num_lambda, num_gamma, numCases);

MC_ERR_OUT =zeros(num_lambda, num_gamma, nRound, numCases);
%MC_ERR_OUT_STD =zeros(num_lambda, num_gamma, numCases);

for ii =1 : numCases
    
    miss_ii =missLevel_arr(ii);
    
    for jj =1: nRound
    
        %X_i = data.X_Omega{miss_ii, jj};        
        Omega_i =data.Omega{miss_ii, jj};                
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
        
        tic        
        %% Evaluate S3LR:
        for kk=1:length(lambda_arr)
            lambda0 =lambda_arr(kk);
            for ll =1: length(gamma_arr)
                gamma0 =gamma_arr(ll);
                X_i =X .* Omega_i_in;
                [acc, mc_err, X_fill] = test_alg(X_i, Omega_i_in, index, X, lambda0, gamma0);
                
                ACC(kk, ll, jj, ii) =acc;
                MC_ERR(kk, ll, jj, ii) =mc_err;

                tmp =X - X_fill;         
                Omega_i_out =logical(Omega_i_out);
                mc_err_out = norm(tmp(Omega_i_out),'fro') / (norm(X,'fro'));
                MC_ERR_OUT(kk, ll, jj, ii) =mc_err_out;
                
                disp(['*@ miss rate: ',num2str(missing_rates(miss_ii)*100),'%, lambda0=',num2str(lambda0),', gamma0=', num2str(gamma0),...
                    ',  S3LR acc: ', num2str(acc), ',    mc err: ', num2str(mc_err), ', mc err out: ', num2str(mc_err_out)]);
                  
                toc;
            end
        end
        
        disp('=======================')
        toc;
        save(str_save_results, 'MC_ERR','ACC','MC_ERR_OUT','lambda_arr','gamma_arr','cv_percent','missLevel_arr','missing_rates');
    end
end
% load str_save_results;
% show_paraCrossValidation(lambda_arr, gamma_arr, MC_ERR_OUT, 'MC ERR OUT');
% show_paraCrossValidation(lambda_arr, gamma_arr, MC_ERR, 'MC ERR');
% show_paraCrossValidation(lambda_arr, gamma_arr, ACC, 'ACC (%)');

function [acc_i, mc_err_i, X_fill] = test_alg(Xi, Omega_i, index, X, lambda0, gamma0, Xi_fill)
%
idx0 = index;

opt.tol =1e-5;
opt.maxIter =1e3;
opt.rho =1.1;
opt.mu_max = 1e6;
   
%% Using LRMC for initialization
if nargin >=7       
    Xi = Xi_fill;
else
    Xi =Xi + zeros(size(Xi));
end

Gamma = gamma0;
% Gamma =0.02;
% gamma0 =0.02 * sum(sum(Omega_i))/(size(Omega_i,1)*size(Omega_i,2));

relax =1; % {1, 2}
affine =0;

opt.norm_sr ='1';
opt.norm_mc ='1';
opt.mu_max = 1e8;
opt.tol =1e-5;
opt.maxIter =2e3; 

t_max =5; %6 %30;
T =11; % 11
nu1 =1.0; %1/1.5;%1.5; % 2;
nu2 =1.0;%1.2;%1.5; % 2;
% tau = Inf; % (1e0) 1e1, 1e2, 1e3 
% lambda_max = Inf; Gamma_max = Inf;
[acc_i, mc_err_i, ~, X_fill] =S3LR(Xi, Omega_i, idx0, X, lambda0, Gamma, gamma0, relax, affine, t_max, opt, T, nu1, nu2);