% my_lambda.m
%
%
%% For Alg 15 and 16
%lambda =20.*(ob_rate.^5);
%y =[20.0000 15.4756 11.8098 8.8741 6.5536 4.7461 3.3614 2.3206 1.5552 ...
%    1.0066 0.6250 0.3691 0.2048 0.1050 0.0486 0.0195 0.0064 0.0015 0.0002];
%% For Alg 12 and 14
%lambda =10.*(ob_rate.^3)+1;
%% For Alg 3 to 8
%lambda =10.*(ob_rate.^3)+0.4
%% For Alg 1 lambda =2;
%% For alg 2 lambda =0.1 and 0.5 [1:6, 7:19]
function y =my_lambda_new(x,alg, missing_level_i)
%x =0:5:90;
miss_rate = x/100;
ob_rate =1 - miss_rate;
%lambda0 =20;
%lambda_arr =[0.05 0.1 0.5 1 2 5 10 20 50 100];
switch alg
    case {0, 24} % zero-fill SSC
        lambda_arr =5*ones(1,size(x,2));
    case {2} % zero-fill LRR
        lambda_arr =[0.1*ones(1,6),0.5*ones(1,size(x,2)-6)];
% For alg 2 lambda =0.1 and 0.5 [1:6, 7:19]
    case {3,4,5,6,8,9, 26} % 24
        lambda_arr =10.*(ob_rate.^3)+0.4;
%     case 3 %MC +SSC
%     case 4 %MC +LRR
%     case 5 %MC_E1 +SSC
%     case 6 %MC_E1 +LRR
%     case 7 %MC_E_fro +SSC
%     case 8 %MC_E_fro +LRR

    case {12,14} % lrr_mv_zd
        lambda_arr =10.*(ob_rate.^3)+1;
        
    case {15,16,17,18,19,20,21,22,23,25,27} % frr_mv, frr_mv_zd    
        lambda_arr =20.*(ob_rate.^5);
        
    case {1,7, 28, 29, 30}
        lambda_arr =5.*ones(size(ob_rate));
        
    otherwise
        disp('error parameters!');
end
y =lambda_arr(missing_level_i);