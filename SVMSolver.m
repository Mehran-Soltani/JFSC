function [alpha, objDual, b, svIdx] = SVMSolver(K, y, reglambda)
% SVM interface
% Input:
%   K: kernel matrix
%   y: label for training data
%   alpha0: initial value for variables alpha in dual domain
%   para: SVM related parameters
% Output:
%   alpha: optimal variable in dual domain
%   objDual: optimal value for objective function in dual domain
%   b: bias of SVM
%   svIdx: index for support vectors


% if nargin<3
    wPos = 1;
    wNeg = 1;
%     reglambda = 1;
% else
%     wPos = para.weightPosSamp;
%     wNeg = para.weightNegSamp;
%     reglambda = para.regLambda;
% end
alpha_eps = 1e-3;

y = y(:);
%% SVM solver - libSVM
n = size(K, 1);
K1 = [(1:n)', K];
opts = sprintf('-t 4 -c %g -w1 %g -w-1 % g -q', reglambda, wPos, wNeg);
model = svmtrain(y, K1, opts);
alpha = zeros(n, 1);
alpha(model.SVs) = abs(model.sv_coef);
alpha(alpha<alpha_eps*reglambda) = 0;
c = alpha.*y;
objDual = sum(alpha(:)) - c'*K*c/2;

b = -model.rho;
svIdx = model.SVs;



