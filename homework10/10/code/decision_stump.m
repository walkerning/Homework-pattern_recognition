function [k, a, d] = decision_stump(X, y, w)
% decision_stump returns a rule ...
% h(x) = d if x(k) <= a, -d otherwise,
%
% Input
%     X : n * p matrix, each row a sample
%     y : n * 1 vector, each row a label
%     w : n * 1 vector, each row a weight
%
% Output
%     k : the optimal dimension
%     a : the optimal threshold
%     d : the optimal d, 1 or -1

% total time complexity required to be O(p*n*logn) or less
%%% Your Code Here %%%
[~, p] = size(X);
max_dim_score = 0;
for dim=1:p % 对于所有feature维度循环
    [sorted_X, ind] = sort(X(:, dim)); % 按照这一维度的feature排序
    sorted_wy = y(ind) .* w(ind); % 得到[w_k * y_k] 数组
    cumsum_front = cumsum(sorted_wy); % 得到\sum_{k<=j}[w_k * y_k]数组
    whole_sum = sum(sorted_wy);
    pos_scores = whole_sum - cumsum_front * 2; % 得到F_np
    [dim_score, max_ind] = max(abs(pos_scores)); % 得到最佳划分index max_ind
    if dim_score > max_dim_score
        max_dim_score = dim_score;
        k = dim;
        a = sorted_X(max_ind); % a = x[max_ind]
        if pos_scores(max_ind) > 0 % 判断是F_np里找到最大还是F_pn里找到最大
            d = -1;
        else
            d = 1;
        end
    end
end
%%% Your Code Here %%%
end