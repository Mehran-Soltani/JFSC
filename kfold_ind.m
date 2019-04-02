function [train,test] = kfold_ind(xdata,ydata,k)
indices = crossvalind('kfold',length(ydata),k);
train = cell (3,k); test= cell(3,k);
for i = 1:k
indtrain = find(indices ~=i);
train{1,i} = xdata(indtrain, :);train{2,i}=ydata(indtrain); train{3,i} = indtrain;
indtest = find(indices == i);
test{1,i} = xdata(indtest,:); test{2,i}= ydata(indtest); test{3,i} = indtest;
end
end