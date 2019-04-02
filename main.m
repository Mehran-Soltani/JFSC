%%%%%%%%%%%% main optimization algorithm

%% training data and the corresponding labels
% form of the input : N*d data with N number of samples and d  number of
% features 
%% prepare data (Normalization to [0,1] interval)
%% load data 
clc
clear
load Spect_data(534)
load k
norm_data = prep_data(Spect_data(:,2:5));

X = norm_data;
labels = Spect_data(:,6);
nfold = 10;
accuracy1 = zeros(1,nfold);
[train,test] = kfold_ind(X, labels,nfold);
kernelpar = 1;
k = KernelGradient_opt(X,X,kernelpar);                          % a N*N cell containing km(xi,xj)
% save ('k')

for i = 1:nfold
    i
    Xtrain = train{1,i};
    ytrain = train{2,i};
    k1 = k(train{3,i},train{3,i});
    
    Xtest = test{1,i};
    ytest = test{2,i};
    k2 = k(train{3,i},test{3,i});
%%%%%%%%%%%%%%%%%%%%----------Optimization problem----------%%%%%%%%%%%%%%%%%%%

%% optimization parameters
num_kernels = 5;
[n,num_feas] = size(Xtrain);
alpha_n = zeros (num_feas , num_kernels); % initial value of alpha (kernel coefficients)(d*Kapa)
t = 0 ;                                 % number of iterations
regHyper = 20;                             % regularization hyperparameter(C) 
lambda = 1;      



while t<100
  t = t+1;  
 %% kernel construction 
 Alpha = alpha_n(:); 
 K =kernel_matrix_opt(k1, Alpha);
 %% Use SVMsolver for updating w,b
     [beta, objDual, b, svIdx] = SVMSolver(K, ytrain, regHyper);
     % compute w matrix (1*d)
     c = beta.*ytrain;
     w = sum(repmat(c,1,num_feas).*Xtrain,1); 
     % compute f(x) = sigma(beta*y*k)+b for all n samples (1*n vector)
     f_vec = sum(repmat(c,1,n).*K,1) + b; 
     % compute loss function
     L = regHyper*sum(max(0,1-ytrain'.*f_vec));
     
     %%%%%%%%%%%-------- optimization of the second subproblem----------%%%%%%%%%% 
     
     Sec_subprob =@(Alpha,ytrain,c,n,k1 , b,lambda)(regHyper*sum(max(0,1-ytrain'.*(sum(repmat(c,1,n).*kernel_matrix(k1, Alpha),1) + b))) + ...
         lambda * norm(Alpha,1));
     f = @(Alpha) Sec_subprob(Alpha,ytrain,c,n,k1 , b,lambda);
     up = 1000*ones(5*num_feas,1); low = zeros(5*num_feas,1); % upper bound and lower bound of qalpha
     alpha_n = gradient_projection (f,Alpha,up,low,20,beta,n,c,ytrain,k1,b,lambda,regHyper,num_feas);
     alpha_n = max(0,alpha_n); % guarantee positivity of the alpha values
     % cost function (SVM regularization + loss function + feature selection regularization ) 
     M = norm(w,2)/2 + regHyper*sum(max(0,1-ytrain'.*(sum(repmat(c,1,n).*kernel_matrix(k1, alpha_n),1) + b))) + ...
         lambda * norm(alpha_n,1);
     norm_alpha = norm((Alpha-alpha_n),2)/(norm(Alpha,2)*norm(alpha_n,2));
     
    if (norm_alpha<1e-3 | M<1e-6)
        
        break 
    end
    
end

% testing y_predict =sign(sum(beta*y*kernel)+b)
KK = kernel_matrix(k2, alpha_n); % kernel matrix between train and test data
m = size(ytest,1);
y_predict = sign(sum(repmat(c,1,m).*KK,1) + b);
accuracy1(i) = mean(y_predict == ytest');
save('accuracy1') 

end 

Mean_acc = mean(accuracy)

save('accuracy') 