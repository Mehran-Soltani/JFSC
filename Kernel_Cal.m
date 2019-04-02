function K = Kernel_Cal( X1, X2, kernel, kernelpar , dist )
 % Kernel_func  : Compute Support Vector Machine kernel function
 % Computing valid kernels
 %   kernel = 'rbf'
 %       radial basis function, common length scale for all inputs is
 %       kernelpar(1), scaled with the number of inputs nin
 %       K = exp(-D(X1i,X2i)/(kernelpar*nin))
 %              a) D(X1,X2) = sum((X1-X2)^2) 'Sq_norm'
 %              b) D(X1,X2) = (1/2)*sum((X1-X2)^2/(X1+X2)) 'Chi_Sq'
 %              c) D(X1,X2)=sum(abs(cdf(X1)-cdf(X2))) 'EMD'
 %   kernel = 'linear' 
 %       inner product
 %   kernel = 'HIK'
 %       histogram intersection kernel 

 %% Dimensions should match
 
 if (size(X1,2)~=size(X2,2))
    error('X1 & X2 differ in dimensionality!!');
 end
[N1, d] = size(X1);
[N2, nin] = size(X2);
 
%% 
 switch kernel
     case 'rbf'
        switch dist 
            case 'Sq_norm'
                K = exp(-pdist2(X1,X2,'sqeuclidean')/(nin*kernelpar(1)));
            case 'Chi_Sq'
                K = exp(-pdist2(X1,X2,'chisq')/(nin*kernelpar(1)));
            case 'emd'
                K = exp(-pdist2(X1,X2,'emd')/(nin*kernelpar(1)));
            otherwise
                error ('Unknown distance function')
        end
     case 'Linear'
           K = X1*X2';
     case 'HIK'
         if N1~=N2
             error('You cant use these Kernel for more than two samples')
         end
         m = min(X1,X2);
         K = sum(m,2);
     otherwise 
         error('Unknown kernel function')
 end
 
end