function [ clusters, assigns ] = spectralCluster( X, K, iters, tp, normalize)
%SPECTRALCLUSTER 此处显示有关此函数的摘要
%   此处显示详细说明
num = size(X, 1);
switch tp
    case 0
        W = exp(-pdist(X)/280);
        W = squareform(W);
    case 1
        % 非对称k近邻图
        k = 5;
        W = exp(-pdist(X)/280);
        W = squareform(W);
        [~, inds] = sort(W, 2);
        for i = 1:num
            W(i, inds(i, 1:num-k-1)) = 0;
        end
end

D = diag(sum(W, 2));
L = D - W;

if normalize == 1
    L = pinv(D) * L;
end
% Find smallest M eigen vectors,
% which represenet every points' affinities to 
% M pseudo connected components
M = 5;
[U, ~] = eigs(L, M, 'sm');
[clusters, assigns] = kmeans(U, K, iters);  
end

