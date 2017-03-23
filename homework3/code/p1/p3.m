clear; close all;
setting;
[samples1, samples2] = gen_sample(2 * N, prior(1), u1, sigma1, u2, sigma2);
samples = [samples1; samples2]; % pretend we don't know the label
[pc_e, u_e, sigma_e] = em_gmm(samples, 2, 500);

%% calculate and save the prob mesh
[~, ind] = sort(u_e(1, :), 2);
u_e = u_e(:, ind);
sigma_e = sigma_e(:, :, ind);
p_em1 = reshape(mvnpdf(points, u_e(:, 1)', sigma_e(:, :, 1)), height, width);
p_em2 = reshape(mvnpdf(points, u_e(:, 2)', sigma_e(:, :, 2)), height, width);
save('p_em.mat', 'p_em1', 'p_em2');