% run_nView_MS_S3LR_Test_155_TSP2015.m
% Modified by Chunguang LI
% August 18, 2015
%--------------------------------------------------------------------------
% This is the main function to run the SSC algorithm for the motion
% segmentation problem on the Hopkins 155 dataset.
%
% cd to the main folder containing the Hopkins 155 sequences
% add the path to the folder "SSC_motion_face" containing these m-files
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
results_fn =['S3LR_MS_seq155_nView_resultsComparison_150points_TSP2015_',datestr(now,30),'.mat'];
addpath('../SSC_ADMM_v1.1');
%alpha = 50; %800;
miss_val_p =[0.05, 0.1, 0.15, 0.2, 0.25, 0.30];
%miss_val_p =[0.15];
alpha_arr =[700, 700,700, 700, 700, 700];

maxNumGroup = 5;
for ii = 1:maxNumGroup
    num(ii) = 0;
end
nTrials =10; % repeat 20 random trials

nAlg_MC =4; % ZF, PF, LR, S3LR, 
MC_ERR =ones(156, length(miss_val_p), nTrials, nAlg_MC);
nAlg_ACC =12 + 1; % 3 x 3 + 2 + 1 + 1
ACC =zeros(156, length(miss_val_p), nTrials, nAlg_ACC);
iter_cell =cell(156, length(miss_val_p), nTrials);

for kk =1:size(miss_val_p,2)
    for ii =1:nTrials   % repeat 20 random trials       
        ll =1; 
        d = dir;
        for i = 1:length(d)            
        
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
                    X = reshape(permute(x(1:2,:,:),[1 3 2]),D,N);                    
                    p =miss_val_p(kk);
                    %idx =randperm(size(X,2));
                    idx =1:size(X,2);
                    Ns =min(150, size(X,2)); % 150
                    X =X(1:12, idx(1:Ns));                    
                    
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
                    Omega_i =logical(Omega);

                    %% Algorithms to Compare:
                    % data completion: 
                    % 0. ZF
                    % 1. PF
                    % 2. LRMC
                    % 3. S3LR                                    
                    
                    % motion segmentation:
                    % 1. ZF + SSC
                    % 2. ZF + LRR
                    % 3. LRMC + SSC
                    % 4. LRMC + LRR
                    % 5. LRMC + GPCA
                    % 6. S3LR
                    % 7. S3LR + SSC
                    % 8. S3LR + LRR
                    % 9. S3LR + GPCA
                    % 10. PF + SSC
                    % 11. PF + LRR
                    % 12. PF + GPCA
                    % 13. MFEM
                    
                    %% 1. ZF
                    X_fill =X_Omega;
                    mc_err = norm(X_fill - X,'fro')/norm(X,'fro');
                    disp(['zero fillled error: ', num2str(mc_err)]);
                    %err_zeroMC(kk, ll, ii) = mc_err;                
                    MC_ERR(ll, kk, ii, 1) =mc_err;
                    
                    %% 1.1 ZF +SSC                  
                    alpha =alpha_arr(kk);%alpha = 50;
                    r = 0; affine = true; outlier = true; rho=0.7;      
                    [missrate1,~] = SSC(X_fill,r,affine,alpha,outlier,rho,s(1:Ns));
                    disp(['zero_fill SSC missrate1: ', num2str(missrate1)]);                    
                    %err_zeroSSC(kk, ll, ii) = missrate1;
                    ACC(ll, kk, ii, 1) =1 - missrate1;
                    
                    %% 1.2 ZF + LRR             
                    augmX = [ones(1,size(X_fill,2));X_fill];
                    lambda_lrr =2.5; % 2, 2.5, 5
                    [acc_lrr] =lrr_SC_aff(augmX, s(1:Ns)', lambda_lrr);
                    missrate1_lrr = 1 - acc_lrr;
                    disp(['zero_fill LRR missrate1: ', num2str(missrate1_lrr)]);            
                    %err_zeroLRR(kk, ll, ii) = missrate1_lrr;
                    ACC(ll, kk, ii, 2) =1 - missrate1_lrr;
                    
                    %% 2. LRMC: Low-Rank Matrix Completion
                    lambda_mc =Inf;                    
                    opt.tol =1e-5;
                    opt.maxIter =1e6;
                    opt.rho =1.1;
                    opt.mu_max = 1e8;
                    
                    [X_fill, E, stat] = inexact_ladm_mc_E_fro(X_Omega, Omega,lambda_mc,opt);
                    mc_err = norm(X-X_fill,'fro')/norm(X,'fro');
                    disp(['MC error: ', num2str(mc_err)]);
                    %err_MC(kk, ll, ii) = mc_err;
                    MC_ERR(ll, kk, ii, 2) =mc_err;

                    %% 2.1 LRMC + SSC
                    alpha =alpha_arr(kk);%alpha = 50;                    
                    r = 0; affine = true; outlier = false; rho=0.7;         
                    [missrate1,~] = SSC(X_fill,r,affine,alpha,outlier,rho,s(1:Ns));
                    disp(['SSC missrate1: ', num2str(missrate1)]);
                    %err_MCSSC(kk, ll, ii) =missrate1;
                    ACC(ll, kk, ii, 3) =1 - missrate1;
                    
                    %% 2.2 LRMC + LRR
                    augmX = [ones(1,size(X_fill,2));X_fill];
                    %lambda =2.5;
                    [acc_lrr] =lrr_SC_aff(augmX, s(1:Ns)', lambda_lrr);
                    missrate1_lrr = 1-acc_lrr;
                    disp(['LRR missrate1: ', num2str(missrate1_lrr)]);
                    %err_MCLRR(kk, ll, ii) =missrate1_lrr;
                    ACC(ll, kk, ii, 4) =1 - missrate1_lrr;
                    
                    %% 2.3 LRMC + GPCA
                    grps = gpca_pda_spectralcluster(X_fill, max(s(1:Ns)));
                    missrate = Misclassification(grps',s(1:Ns));
                    disp(['LRMC + GPCA missrate: ', num2str(missrate)]);
                    ACC(ll, kk, ii, 5) =1 - missrate;                    

                    %% 3. S3LR
                    augmX = [ones(1,size(X_Omega,2));X_Omega];
                    augmOmega =[ones(1,size(X_Omega,2));Omega];
                    %gamma0 =0.02 * sum(sum(Omega))/(size(Omega,1)*size(Omega,2));
                    
                    Gamma =0.05;
                    gamma0 =0.05;%0.05;%0.05; 0.1; 0.02
                    tau =5e5; % (1e0) 1e1, 1e2, 1e3       
                    
                    lambda =0.005;%0.01; % 0.001, 0.01
                    %opt.norm_lrr ='fro';
                    %opt.norm_mc ='fro';                    
                    
                    opt.norm_sr ='fro';
                    %opt.norm_sr ='1';
                    opt.norm_mc ='1';
                    opt.mu_max = 1e8;
                    opt.maxIter =1e6; 
                    opt.tol =1e-5;
                    opt.rho =1.1;            
                    Xc =[ones(1,size(X_Omega,2));X];
                    
                     relax =1; 
                     affine =1;
                     t_max =20;
                     %t_max =1;
                     T =11;
                     nu1 =1.0;%2.0;%1.05; %nu =1.2;  % 
                     nu2 =1.0;%2.0;%1.05; %nu =1.2;  % 
                       [acc_slr, mc_acc_slr, ~, X_fill, iterStatus]=S3LR(augmX, augmOmega, s(1:Ns)', Xc, lambda, Gamma, gamma0, relax, ...
                          affine, t_max, opt, T, nu1, nu2);                  
                      
                    missrate1_slr = 1-acc_slr;
                    disp(['S3LR C missrate1: ', num2str(missrate1_slr)]);
                    %err_S3LR(kk, ll, ii) =missrate1_slr;
                    %err_S3LRMC(kk, ll, ii) =mc_acc_slr;
                    MC_ERR(ll, kk, ii, 3) =mc_acc_slr;
                    ACC(ll, kk, ii, 6) = 1 - missrate1_slr;
                    
                    iter_cell{ll, kk, ii} =iterStatus; %cell(156, length(miss_val_p), nTrials, nAlg_MC);
                    
                    %% 3.1 S3LR for MC + SSC
                    alpha =alpha_arr(kk);
                    r = 0; affine = true; outlier = false; rho=0.7;         
                    [missrate1,C1] = SSC(X_fill(2:end, :),r,affine,alpha,outlier,rho,s(1:Ns));
                    disp(['S3LR for MC + SSC missrate1: ', num2str(missrate1)]);
                    %err_MCSSC2(kk, ll, ii) =missrate1;
                    ACC(ll, kk, ii, 7) = 1 - missrate1;
                    
                    %% 3.2 S3LR for MC + LRR
                    augmX = [ones(1,size(X_fill,2)) ; X_fill(2:end, :)];
                    %lambda =2.5;
                    [acc_lrr] =lrr_SC_aff(augmX, s(1:Ns)', lambda_lrr);
                    missrate1_lrr = 1-acc_lrr;
                    disp(['S3LR for MC + LRR missrate1: ', num2str(missrate1_lrr)]);
                    %err_MCLRR2(kk, ll, ii) =missrate1_lrr;
                    ACC(ll, kk, ii, 8) = 1 - missrate1_lrr;
                    
                    %% 3.3 S3LR for MC  + GPCA
                    grps = gpca_pda_spectralcluster(X_fill, max(s(1:Ns)));
                    missrate = Misclassification(grps',s(1:Ns));
                    disp(['S3LR for MC + GPCA missrate: ', num2str(missrate)]);
                    ACC(ll, kk, ii, 9) =1 - missrate;     

                    %% 4. PF for MC
                    mode = 'PF'; %SVD, PF
                    rr = 5;
                    [X_fill, ~, ~] = PowerFactorization(X_Omega, rr, mode);                    
                    %mc_err = norm(X-X_fill,'fro')/norm(X,'fro');
                    mc_err = nan;
                    disp(['PF for MC error: ', num2str(mc_err)]);
                    MC_ERR(ll, kk, ii, 4) =mc_err;
                    
                    %% 4.1 PF for MC + SSC
                    alpha =alpha_arr(kk);
                    r = 0; affine = true; outlier = false; rho=0.7;         
                    [missrate1,C1] = SSC(X_fill(2:end, :),r,affine,alpha,outlier,rho,s(1:Ns));
                    disp(['PF for MC + SSC missrate1: ', num2str(missrate1)]);
                    %err_MCSSC2(kk, ll, ii) =missrate1;
                    ACC(ll, kk, ii, 10) = 1 - missrate1;
                    
                    %% 4.2 PF for MC + LRR
                    augmX = [ones(1,size(X_fill,2)) ; X_fill(2:end, :)];
                    %lambda =2.5;
                    [acc_lrr] =lrr_SC_aff(augmX, s(1:Ns)', lambda_lrr);
                    missrate1_lrr = 1-acc_lrr;
                    disp(['PF for MC + LRR missrate1: ', num2str(missrate1_lrr)]);
                    %err_MCLRR2(kk, ll, ii) =missrate1_lrr;
                    ACC(ll, kk, ii, 11) = 1 - missrate1_lrr;
                    
                    %% 4.3 PF for MC  + GPCA
                    grps = gpca_pda_spectralcluster(X_fill, max(s(1:Ns)));
                    missrate = Misclassification(grps',s(1:Ns));
                    disp(['PF for MC + GPCA missrate: ', num2str(missrate)]);
                    ACC(ll, kk, ii, 12) =1 - missrate;                       
                    
                    
                    %% 5. MFEM
                    
%                     data =x(:, 1:min(150, size(x,2)), 1:min(6, size(x,3))); 
%                     %data =x(:, 1:end, 1:end); 
%                     k =max(s);
%                     sigmaX = 0.05;
%                     invPsi = ones(size(data, 2)*2);
%                     %A = factorizationWithEM( data, k, sigmaX, invPsi);
%                     T =10;
%                     tmp_err =zeros(1,T);
%                     tmp_INDEX =zeros(length(s), T);
%                     for t =1: T
%                         [index, err, A] = MultiFactorEM( data, k, sigmaX, invPsi);
%                         missrate = Misclassification(index,s);
%                         %tmp_err(t) =err;
%                         tmp_err(t) =missrate;
%                         tmp_INDEX(:, t) =index;
%                         disp(['-----------------  missrate: ', num2str(missrate),'  , err: ', num2str(err)]);    
%                     end
%                     [~, t_star] = min(tmp_err);
%                     missrate2 = Misclassification(tmp_INDEX(:, t_star),s);
%                     disp(['------------------------- - - # Sequence: ', num2str(n_seq)]);
%                     disp(['* * * MFEM missrate  2: ', num2str(missrate2), ',  reconstruction err: ', num2str(tmp_err(t_star))]);    
%                     disp(['------------------------'])                            
                    
                    
                    %r = 4*n; affine = true; outlier = false; rho = 0.7;
                    %[missrate2,C2] = SSC(X,r,affine,alpha,outlier,rho,s);

                    %num(kk, n, ii) = num(kk, n, ii) +1;
                    %missrateTot1{kk, n}(num(kk, n), ii) = missrate1;
                    %num(n) = num(n) + 1;
                    %missrateTot1{n}(num(n)) = missrate1;
                    %missrateTot2{n}(num(n)) = missrate2;
                    %end
                    ll =ll+1;
                    
                    eval(['cd ' filepath]);
                    %save SSC_MS.mat missrate1 missrate1_slr  C1 alpha  %C2 missrate2
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
        
            save(results_fn, 'ACC','MC_ERR', 'miss_val_p','nTrials','alpha','lambda_lrr',...
                'gamma0','Gamma','T','nu1','nu2', 'lambda', 'opt','t_max');
    end
end