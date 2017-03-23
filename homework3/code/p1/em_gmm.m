function [ pc_e, u_e, sigma_e ] = em_gmm( samples, num_comp, max_iter )
%EM_GMM ��EM�㷨���˹���ģ��
%       ����:
%           samplesΪnsamples x dim������samples
%           num_compΪ���ģ������
%       ����:
%           pi_eΪ1 x num_compΪÿ��component��prior
%           u_e Ϊdim x num_compΪÿ��component�ľ�ֵ
%           sigma_e Ϊdim x dim x num_compΪÿ��component��Э�������
%       num_compѡ����Կ��Կ����ø��ݷ����split-merge�������

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

