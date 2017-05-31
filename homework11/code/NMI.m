function [ score ] = NMI( true_labels, assign_labels, K )
%NMI 此处显示有关此函数的摘要
%   此处显示详细说明
n = length(true_labels);
s_inds = cell(K, 1);
n_s = zeros(K, 1);
t_inds = cell(K, 1);
n_t = zeros(K, 1);
for s=1:K
    s_inds{s} = assign_labels == s;
    n_s(s) = sum(s_inds{s});
end
for t=1:K
    t_inds{t} = true_labels == t;
    n_t(t) = sum(t_inds{t});
end
upper = 0;
for s=1:K
    for t=1:K
        st_inds = s_inds{s} & t_inds{t};
        n_st = sum(st_inds);
        if n_st > 0
            upper = upper + n_st * log(n * n_st / n_s(s) / n_t(t));
        end
    end 
end
score = upper / sqrt(sum(n_s(n_s>0) .* log(n_s(n_s>0) / n)) *...
    sum(n_t(n_t>0) .* log(n_t(n_t>0) / n)));
end

