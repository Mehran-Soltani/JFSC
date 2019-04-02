function k = KernelGradient_opt(X_1,X_2,kernelpar)
[N,d] = size(X_1);
[M,p] = size(X_2);
if p~=d
   error('inputs must have equal dimensions') 
end
k = cell(N,M);
for i = 1:N
    for j = i:M
        X1 = X_1(i,:);
        X2 = X_2(j,:);
       h = zeros(d,5) ;
         for m = 1:d
             h (m , :) = [Kernel_Cal( X1(m), X2(m), 'rbf', kernelpar , 'Sq_norm' ),...
                 Kernel_Cal( X1(m), X2(m), 'rbf', kernelpar , 'Chi_Sq' ),...
                 Kernel_Cal( X1(m), X2(m), 'rbf', kernelpar , 'emd' ),...
                 Kernel_Cal( X1, X2, 'Linear', kernelpar , 0 ),...
                 Kernel_Cal( X1, X2, 'HIK' , kernelpar , 0 )];
         end
         k{i,j} = h(:);
         k{j,i} = h(:);
    end
end
end