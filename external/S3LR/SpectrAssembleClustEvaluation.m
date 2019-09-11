% SpectrAssembleClustEvaluation.m
%   Run spectral embedding and repeat k-means for T times, and take the assembled results
%
% Spectral assembled clustering evaluation
%
%  2013-02-8
%  By Li Chun-Guang
% Revised: Sept. 19, 2013
function [acc_assemble,M,acc,acc_std,centroid] =SpectrAssembleClustEvaluation(L,idx0,k,T,epsilon, thresh, centroid)
if nargin < 7
    warmstart =0;
else
    warmstart =1;
end
if nargin < 6
    thresh =0.6;
end
if nargin<5
    epsilon =1e-40;
end
if nargin<4
    epsilon =1e-40;
    T =20;
end
if nargin<3
    epsilon =1e-40;
    T =20;
    k=1;
end
acc_tmp =zeros(1,T);
idx_arr =zeros(T,size(L,2));
acc =0;
acc_std =0;
acc_assemble =0;
%% Check full zero L
if norm(L,'fro') > epsilon
    L = abs(L) + abs(L');
    %% check the full-zero columns
    sumL =sum(L,1);
    tmp_idx =find(sumL < epsilon, 1);
    if (~isempty(tmp_idx))
        L(:,tmp_idx) =1e-4*abs(randn(size(L,1),length(tmp_idx)));
        L = (L + L') / 2;
    end
    %% Building graph Laplacian and do Spectral Embedding
    %L = BuildAdjacency(thrC(L,0.7));
    D = diag(sum(L,2).^(-1./2)); 
    L = eye(size(L,1)) - D*L*D;
    [U,~,~] = svd(L);
    V = U(:,end - k + 1 : end);
    %% Repeat k-means T times
    %idx =0;
    %warmstart =0;
    for ii =1 :T
        %idx = clu_ncut(L,k);     
        %idx = kmeans(V, k,'emptyaction','singleton','replicates',20,'display','off');
        %% if we use 'warmstart', the result will never be worse than 'sample' because my midification in kmeans.        
        if (~warmstart)
            [idx, centroid]= kmeans(V, k, 'Start', 'sample','emptyaction','singleton','replicates',20,'display','off'); 
        else
            [idx, centroid] = kmeans(V, k, 'Start', 'warmstart', 'initialcentroid', centroid,'emptyaction','singleton','replicates',20,'display','off'); 
            %idx = kmeans(kerNS,n,'Start', 'warmstart', 'initiallabel', idx, 'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
        end
        acc_tmp(ii) = compacc((idx)' , idx0);
        idx_arr(ii,:) = (idx)';
    end
    %% Assemble the T results via majority voting
    %thresh =0.6;
    [idx_assemble,M] =Clustering_assemble(idx_arr,size(L,2),k,thresh);
    if (max(idx0)<10)
        missrate = Misclassification(idx_assemble',idx0);
        acc_assemble =1 - missrate;
    else
        acc_assemble = compacc(idx_assemble,idx0);
    end
    
%     %% Test the optimal 'thresh'
%      acc_test =[];
%      for i=1:100
%          thre =0.01*i;
%          [idx_assemble,M] =Clustering_assemble(idx_arr,size(L,2),k,thre);
%          acc_test = [acc_test , compacc(idx_assemble,idx0)];
%      end
%      figure;plot(acc_test);
    
    acc = acc + mean(acc_tmp);
    acc_std =acc_std + std(acc_tmp);
    %disp(num2str(mean(acc_assemble)));
else
    acc = acc + 0;
    acc_std =acc_std + 1;
    M =zeros(size(L,2));
    disp('failed in the bad condition of L...');
    
end