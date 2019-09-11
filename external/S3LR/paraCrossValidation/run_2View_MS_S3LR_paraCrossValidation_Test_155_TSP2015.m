% run_2View_MS_S3LR_paraCrossValidation_Test_155_TSP2015.m
% Modified by Chunguang LI
% August 17, 2015
%--------------------------------------------------------------------------
% This is the main function to run the SSC algorithm for the motion
% segmentation problem on the Hopkins 155 dataset.
%
% cd to the main folder containing the Hopkins 155 sequences
%
% avgmissrate1: the n-th element contains the average clustering error for 
% sequences with n motions (using 2F-dimensional data)
% avgmissrate2: the n-th element contains the average clustering error for 
% sequences with n motions (using 4n-dimensional data)
% medmissrate1: the n-th element contains the median clustering error for 
% sequences with n motions (using 2F-dimensional data)
% medmissrate2: the n-th element contains the median clustering error for 
% sequences with n motions (using 4n-dimensional data)
%--------------------------------------------------------------------------
addpath('../../lrr');
addpath('../../GPCA');
addpath('../../inexact_ladm_mc');
addpath('../../local_comp_represent');
addpath('../../inexact_alm_mc/PROPACK');
addpath('../../inexact_alm_mc');
%clc
clear all
close all
% addpath('../Hopkins155_Add_Seq_MissingData');
%cd '../Hopkins155_Add_Seq_MissingData';
%results_fn =['S3LR_MS_seq12_2view_resultsComparison_150points_',datestr(now,30),'.mat'];
cd '../Hopkins155';
results_fn =['S3LR_MS_seq155_2view_paraCV_results_150points_TSP2015_',datestr(now,30),'.mat'];
addpath('../SSC_ADMM_v1.1');

maxNumGroup = 5;
for ii = 1:maxNumGroup
    num(ii) = 0;
end

nTrials =10; % repeat 20 random trials
%miss_val_p =[0.05, 0.1, 0.15, 0.2, 0.25, 0.30];
miss_val_p =[0.10 0.20]; % percentage of missing entries
numCases =length(miss_val_p);

%% lambda0 and gamma0:
%alpha = 50; %800;
lambda_arr =[1e-3 2e-3 3e-3 4e-3 5e-3 6e-3 7e-3 8e-3 1e-2]; % alpha_arr
gamma_arr =[0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1];
Ns =150;
% DEBUG=0;
% %% DEBUG: 
% if (DEBUG ==1)    
%     nTrials =1; %10, repeat 20 random trials
%     lambda_arr =[1e-3 2e-3]; % alpha_arr
%     gamma_arr =[0.01 0.02 0.1];
%     miss_val_p =[0.10 0.20]; % percentage of missing entries
%     Ns =30;
%     %Gamma =0.05;
%     %gamma0 =0.05;%0.05;%0.05; 0.1; 0.02
%     %lambda =0.005;%0.01; % 0.001, 0.01        
% end
num_lambda =length(lambda_arr);
num_gamma =length(gamma_arr);

ACC =zeros(num_lambda, num_gamma, numCases, nTrials, 156);
%ACC_STD =zeros(num_lambda, num_gamma, numCases, nTrials, 156);

MC_ERR =zeros(num_lambda, num_gamma, numCases, nTrials, 156);
%MC_ERR_STD =zeros(num_lambda, num_gamma, numCases, nTrials, 156);

MC_ERR_OUT =zeros(num_lambda, num_gamma, numCases, nTrials, 156);
%MC_ERR_OUT_STD =zeros(num_lambda, num_gamma, numCases, nTrials, 156);

for kk =1:size(miss_val_p,2)
    for ii =1:nTrials   % repeat 20 random trials       
        ll =1; 
        d = dir;
        for i = 1 : length(d)            
        
            if ( (d(i).isdir == 1) && ~strcmp(d(i).name,'.') && ~strcmp(d(i).name,'..') )
                filepath = d(i).name;
                eval(['cd ' filepath]);

                f = dir;
                foundValidData = false;
                for j = 1:length(f)
                    if ( ~isempty(strfind(f(j).name,'_truth.mat')) )
                        ind = j;
                        foundValidData = true;
                        break
                    end
                end
                eval(['load ' f(ind).name]);
                cd ..

                if (foundValidData)
                    disp(['# Sequence: ', num2str(ll)]);
                    %n = max(s);
                    N = size(x,2);
                    F = size(x,3);
                    D = 2*F;
                    %X = reshape(permute(x(1:2,:,:),[1 3 2]),D,N);
                    X_view1 =x(:,:,1);
                    X_viewN =x(:,:,end);
                    X_2view =zeros(9, N);
                    for j_point =1: N
                        A =X_view1(:, j_point);
                        B =X_viewN(:, j_point);
                        X_2view(:, j_point) =kron(A, B);                        
                    end
                    X = X_2view;
                    p =miss_val_p(kk);
                    idx =1:size(X,2);
                    Ns =min(150, size(X,2)); % 150
                    %Ns =size(X,2);
                    X =X(:, idx(1:Ns));
                    s =s(idx);
                    n =max(s(1:Ns));
                    %counter_n_N_F(ll,1) =n;
                    %counter_n_N_F(ll,2) =N;
                    %counter_n_N_F(ll,3) =F;
                    
                    %if (0==1)
                    X_max =max(X(:));
                    X = X./X_max;

                    R_num =rand(size(X));
                    Omega =(R_num > p);
                    X_Omega =X.* Omega;
                    Omega =logical(Omega);
                    
                    
                    %% parameter selection via cross validation on p% held-out entries
                    %       Step 1: randomly sample p% entries from Omega_i as the held-out entries, denoted as Omega_i_out
                    %       Step 2: perform matrix completion on Omega_i \ Omega_i_out, denoted as Omega_i_in
                    %       Step 3: evaluate the completion error on Omega_i_out
                    %       Step 4: change parameter and repeat Step 3

                    [m,n] =size(Omega);
                    %% paraCrossValidation
                    tmp_idx =find(Omega >0);
                    tmp_idx_perm = randperm(length(tmp_idx));

                    % held-out entries to perform completion and evaluate completion errors
                    cv_percent =0.05;
                    Omega_out =zeros(m, n);    
                    Omega_out( tmp_idx( tmp_idx_perm( 1 : fix(m * n * cv_percent) ) ) ) =1;

                    % left observed entries to perform completion
                    Omega_in =zeros(m, n);
                    Omega_in( tmp_idx( tmp_idx_perm( fix(m * n * cv_percent) + 1 : end) ) ) =1;

                    
                    for ii_lambda =1 : num_lambda
                        for jj_gamma =1 : num_gamma
                            
                            tic
                            lambda0 =lambda_arr(ii_lambda);
                            gamma0 =gamma_arr(jj_gamma);
                            Gamma =gamma0;
                            %gamma0 =0.02 * sum(sum(Omega))/(size(Omega,1)*size(Omega,2));    
                            
                            %% Perform S3LR on Omega_i_in:                            
                            %augmX = [ones(1,size(X_Omega,2));X_Omega];                    
                            %augmOmega =[ones(1,size(X_Omega,2));Omega];
                            
                            X_Omega_in =X.* Omega_in;
                            Omega_in =logical(Omega_in);
                            
                            augmX = [ones(1,size(X_Omega_in,2));X_Omega_in];                    
                            augmOmega =[ones(1,size(X_Omega_in,2));Omega_in];
                                                    
                            tau =5e5; % (1e0) 1e1, 1e2, 1e3       
                            
                            %opt.norm_lrr ='fro';
                            %opt.norm_mc ='fro';                
                            opt.norm_sr ='fro';
                            %opt.norm_sr ='1';
                            opt.norm_mc ='1';
                            opt.mu_max = 1e8;
                            opt.maxIter =1e6; 
                            opt.tol =1e-5;
                            opt.rho =1.1;
                            Xc =[ones(1,size(X_Omega_in,2));X];
                            
                            relax =1; 
                            affine =1;
                            t_max =20;
                            %t_max =1;
                            T =11;
                            nu1 =1.0;%2.0;%1.05; %nu =1.2;  % 
                            nu2 =1.0;%2.0;%1.05; %nu =1.2;  % 
                            
                            [acc, mc_err, ~, X_fill, iterStatus]=S3LR(augmX, augmOmega, s(1:Ns)', Xc, lambda0, ...
                                Gamma, gamma0, relax, affine, t_max, opt, T, nu1, nu2); %
                            
                            tmp =Xc - X_fill;
                            Omega_out =logical(Omega_out);
                            mc_err_out = norm(tmp(Omega_out),'fro') / (norm(Xc,'fro'));

                            ACC(ii_lambda, jj_gamma, kk, ii, ll) =acc;
                            MC_ERR(ii_lambda, jj_gamma, kk, ii, ll) =mc_err;
                            MC_ERR_OUT(ii_lambda, jj_gamma, kk, ii, ll) =mc_err_out;
                            
                            disp(['**iter: ii lambda =', num2str(ii_lambda), ' jj gamma=', num2str(jj_gamma),...
                                ' seq. ll=', num2str(ll) ]);
                            
                        end
                    end
                    ll =ll+1;
                    
                    eval(['cd ' filepath]);
                    cd ..
                end   
            end
        end

        %L = [2 3];
        %for i = 1:length(L)
        %    j = L(i);
        %    avgmissrate1(j) = mean(missrateTot1{j});
        %   medmissrate1(j) = median(missrateTot1{j});
            %avgmissrate2(j) = mean(missrateTot2{j});
            %medmissrate2(j) = median(missrateTot2{j});
        %end
        save(results_fn, 'ACC','MC_ERR','MC_ERR_OUT','cv_percent', 'miss_val_p','nTrials','lambda_arr',...
                'gamma_arr','T','nu1','nu2', 'opt','t_max');
    end
end
% load str_save_results;
% show_paraCrossValidation(lambda_arr, gamma_arr, MC_ERR_OUT, 'MC ERR OUT');