% show_paraCrossValidation.m
%  This code is to show the MC ERR OUT as a function of lambda and gamma.
%  
%  Note that there are two paramters to be determined in S3LR: \lambda and \gamma0 where \Gamma is set to equal to \gamma0
%  (as \alpha is set to 1). To find the lowest matrix completion error, we check the matrix completion error on the extra held-out
%  p% (e.g. 5%) entries.  By doing so, we can find the parameters to yield the best completion error and the better
%  clustering accuracy. Note that the parameters to yield the lowest completion error might not yield the best clustering
%  accuracy at the same time. 
%
%
% CHUNGUANG LI
% May 12, 2015; Revised July 28, 2017.
function []=show_paraCrossValidation(lambda_arr, gamma_arr, MC_ERR_OUT, strTitle, h, strColormap)
%
if (nargin < 4)
    strTitle = 'MC ERR OUT'; 
end
if (nargin < 5)
    h = figure; 
end
if (nargin < 6)
    strColormap = 'jet'; 
end
%% Show MC ERR OUT
figure(h); 
colormap(strColormap);

%imagesc(MC_ERR_OUT(:,:,1,1)); 
surf(MC_ERR_OUT(:,:,1,1)); 

xlabel('\gamma');
%ylabel('$\frac{k d}{ n}$','interpreter','latex');
ylabel('\lambda');
xlim([1 length(gamma_arr)]);
XTickLabel= gamma_arr;

set(gca,'XTickLabel',XTickLabel,'XLimMode','manual');
ylim([1 length(lambda_arr)]);
YTickLabel= lambda_arr;
set(gca,'YTickLabel', YTickLabel,'YLimMode','manual');
zlabel(strTitle);
colorbar;

% figure; 
% colormap jet;
% surf(MC_ERR(:,:,1,1));
% ylabel('\gamma');
% %ylabel('$\frac{k d}{ n}$','interpreter','latex');
% xlabel('\lambda');
% XTickLabel= lambda_arr;
% set(gca,'XTickLabel',XTickLabel,'XLimMode','manual');
% YTickLabel= gamma_arr;
% set(gca,'YTickLabel', YTickLabel,'YLimMode','manual');
% zlabel('MC ERR (%)');
% colorbar;

% figure; 
% colormap hot;
% surf(ACC(:,:,1,1));
% ylabel('\gamma');
% %ylabel('$\frac{k d}{ n}$','interpreter','latex');
% xlabel('\lambda');
% XTickLabel= lambda_arr;
% set(gca,'XTickLabel',XTickLabel,'XLimMode','manual');
% YTickLabel= gamma_arr;
% set(gca,'YTickLabel', YTickLabel,'YLimMode','manual');
% zlabel('ACC (%)');
% colorbar;

