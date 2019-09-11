% feature_selection.m
%
%
function hatX =feature_selection(X, m, opt)

% 
n =size(X, 1);
N =size(X, 2);
hatX =zeros(m, N);

switch opt
    
    case 'random'
        ind = randperm(n);
        hatX = X(ind(1:m), :);
        
        
    case 'PCA_corr'
        % select the most correlated dimentions via PCA
        % 1. Do PCA to find the PCs of X'*X
        % 2. Select the rows which are the most correlated to those PCs.
        % 3. Test the singular values
        A = X' * X;
         [U,S,V ]=svd(A, 'econ');
         P =U(:, 1:m);
         
         for j =1:m
             corr_j = X* P(:, j);
             corr_j =abs(corr_j);
             [~, idx ] =max(corr_j);
             hatX(j, :) = X(idx(1), :);
             X(idx(1), :) =[];
         end
        
        
    case 'greedy_correlation'
        ind = randperm(n);
        x =X(ind(1), :);
        hatX(1,:) = x;
        X(ind(1), :) =[];
        for j =2:m
            accumulate_coef =[];
            for i =1:size(X, 1)
                coef =X(i, :) * hatX';
                coef =abs(coef);
                accumulate_coef = [accumulate_coef, sum(coef)];
            end
            [~, idx] =min(accumulate_coef);
            hatX(j,:) = X(idx(1), :);
            X(idx(1), :) = [];
        end
        
    case 'DR_PCA'
        [U,~,~] =svd(X, 'econ');
        hatX = U(:, 1:m)'* X;
        
end