function [ pc_e, u_e, sigma_e ] = em_gmm( samples, num_comp, max_iter )
%EM_GMM 用EM算法解高斯混合模型
%       输入:
%           samples为nsamples x dim的输入samples
%           num_comp为混合模型数量
%       返回:
%           pi_e为1 x num_comp为每个component的prior
%           u_e 为dim x num_comp为每个component的均值
%           sigma_e 为dim x dim x num_comp为每个component的协方差矩阵
%       num_comp选择策略可以考虑用根据方差的split-merge在外层做

[nsamples, dim] = size(samples);
% Initialize parameters randomly
pc_e = rand(1, num_comp);
pc_e = pc_e / sum(pc_e); % normalize

u_e = randn(dim, num_comp);

epsilon = 0.1;
sigma_e = zeros(dim, dim, num_comp);
for y=1:num_comp
    Q = rand(dim, dim);
    % avoid singular initialization
    sigma_e(:, :, y) = Q' * Q + epsilon * eye(dim);
end

for iter=1:max_iter
    % E-step: calculate pc = P(comp | x)
    pc = zeros(nsamples, num_comp);
    for y=1:num_comp
        pc(:, y) = mvnpdf(samples, u_e(:, y)', sigma_e(:, :, y));
    end
    pc = pc ./ repmat(sum(pc, 2), 1, num_comp);
    %size(pc)
    % M-step: MLE for Q
    s_y = sum(pc, 1);
    pc_e = s_y / nsamples;
    u_e_t = reshape(sum(repmat(pc, 1, 1, dim) .*...
              repmat(reshape(samples, nsamples, 1, dim),...
                             1, num_comp, 1), 1), num_comp, dim) ./...
                             repmat(s_y', 1, dim);
    for y=1:num_comp
        centered_samples = samples - repmat(u_e_t(y, :), nsamples, 1);
        sigma_e(:, :, y) = centered_samples' * diag(pc(:, y)) *...
            centered_samples / s_y(y);
    end
    u_e = u_e_t';
end
end

