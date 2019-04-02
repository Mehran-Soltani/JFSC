function K =additive_kernel(X, alpha)

% K is an N*N matrix which calculates the kernel function betweein all
% samples . K(i,j) is the kernel between i'th and j'th sample
% inputs : X : training data with N samples and d features
%          alpha : kernel coefficients d*kappa (kappa : number of kernels)
% output : K is an N*N matrix which shows the similarity between training
%           samples
% 

[N,d ] = size(X);
K = zeros(N,N);





end