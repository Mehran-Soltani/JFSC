function K =kernel_matrix_opt(k, alpha)

% K is an N*N matrix which calculates the kernel function betweein all
% samples . K(i,j) is the kernel between i'th and j'th sample
% inputs : k : N*N cell constructed by KernelGradient 
%          alpha : kernel coefficients d*kappa (kappa : number of kernels)
%          
% output : K is an N*N matrix which shows the similarity between training
%           samples
% 
%  must be symmetric

[N] = size(k,1);
K = zeros(N,N);


for i = 1:N
    for j= i:N
         K(i,j) = sum(alpha.*k{i,j});
         K(j,i) = K(i,j);
    end
end

end