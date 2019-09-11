% generate_missing_data.m
%
% Chunguang Li
% July 25, 2015
function [data] = generate_missing_data(X, missing_rates,nRound)
%
%

if (length(missing_rates) > 1)
    m =length(missing_rates);
    
    for jj =1: nRound
        
        rand_num =rand(size(X));
        
        for ii =1: m
            
            p =missing_rates(ii);
            
            % generate the observation data with missing value
            Omega =(rand_num > p);        
            
            data.X_Omega{ii, jj} =X.* Omega;                
            data.Omega{ii, jj} =logical(Omega);
            data.missing_rate(ii) =p;
            data.X = X;
            
        end
        
    end
    
end