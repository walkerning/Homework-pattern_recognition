clear; close all;
setting;
[samples1, samples2] = gen_sample(N, prior(1), u1, sigma1, u2, sigma2);
%% gauss window
gauss_sigma = 200 / N;

% Q: different sigma for samples1 and samples2?
p_gauss1 = gauss_window(gauss_sigma, mesh_x, mesh_y, samples1);
p_gauss2 = gauss_window(gauss_sigma, mesh_x, mesh_y, samples2);

[pe_gauss, ind_gauss] = min(cat(3, p_gauss1, p_gauss2), [], 3);
gauss_err_rate = sum(sum(pe_gauss .* prior(ind_gauss))) * delta;
save('p_gauss.mat', 'p_gauss1', 'p_gauss2');
figure(1);
surf(mesh_x, mesh_y, p_gauss2);

%% rect window
a = 800 / N;

p_rect1 = rect_window(a, mesh_x, mesh_y, samples1);
p_rect2 = rect_window(a, mesh_x, mesh_y, samples2);

[pe_rect, ind_rect] = min(cat(3, p_rect1, p_rect2), [], 3);
rect_err_rate = sum(sum(pe_rect .* prior(ind_rect))) * delta;
save('p_rect.mat', 'p_rect1', 'p_rect2');
figure(2);
surf(mesh_x, mesh_y, p_rect2);