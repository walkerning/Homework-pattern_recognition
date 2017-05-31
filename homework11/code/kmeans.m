function [ clusters, assigns ] = kmeans( X, K, iters )
%KMEAN 此处显示有关此函数的摘要
%   此处显示详细说明
[num, dim] = size(X);
% randomly initialize assigns
approx_n_per_cluster = num / K;
assigns = arrayfun(@(c) c * ones(1, approx_n_per_cluster), (1:K), 'UniformOutput',...
                   false);
assigns = cat(2, assigns{:});
assigns = assigns(randperm(num));
clusters = zeros(K, dim);
for iter=1:iters
    for c=1:K
        clusters(c, :) = mean(X(assigns==c, :), 1);
    end
    avg_square_dist = 0;
    for n=1:num
        [square_dist, ass] = min(sum((repmat(X(n, :), K, 1) - clusters).^2, 2));
        assigns(n) = ass;
        avg_square_dist = avg_square_dist + square_dist;
    end
    avg_square_dist = avg_square_dist / num;
    fprintf(1, 'iter %d: average square distance: %f\n', iter, avg_square_dist);
end
end

