% classifier using EM-estimated 
setting;
a = 800 / N;
gauss_sigma = 200 / N;

em_err_rates = zeros(1, RUN);
rect_err_rates = zeros(1, RUN);
gauss_err_rates = zeros(1, RUN);
for run=1:RUN
    [samples1, samples2] = gen_sample(2 * N, prior(1), u1, sigma1, u2, sigma2);
    samples = [samples1; samples2]; % pretend we don't know the label
    [pc_e, u_e, sigma_e] = em_gmm(samples, 2, 500);

    % calculate error rate
    [~, ind] = sort(u_e(1, :), 2);
    u_e = u_e(:, ind);
    sigma_e = sigma_e(:, :, ind);

    p_em1 = reshape(mvnpdf(points, u_e(:, 1)', sigma_e(:, :, 1)), height, width);
    p_em2 = reshape(mvnpdf(points, u_e(:, 2)', sigma_e(:, :, 2)), height, width);
    [p_em_min, ind] = min(cat(3, p_rect1, p_rect2), [], 3);
    em_err_rates(run) = sum(sum(p_em_min .* prior(ind))) * delta;
    
    % 
    p_rect1 = rect_window(a, mesh_x, mesh_y, samples1);
    p_rect2 = rect_window(a, mesh_x, mesh_y, samples2);

    [pe_rect, ind_rect] = min(cat(3, p_rect1, p_rect2), [], 3);
    rect_err_rates(run) = sum(sum(pe_rect .* prior(ind_rect))) * delta;
    
    %
    p_gauss1 = gauss_window(gauss_sigma, mesh_x, mesh_y, samples1);
    p_gauss2 = gauss_window(gauss_sigma, mesh_x, mesh_y, samples2);

    [pe_gauss, ind_gauss] = min(cat(3, p_gauss1, p_gauss2), [], 3);
    gauss_err_rates(run) = sum(sum(pe_gauss .* prior(ind_gauss))) * delta;
end
em_err_rate = mean(em_err_rates);
em_err_var = var(em_err_rates);
gauss_err_rate = mean(gauss_err_rates);
gauss_err_var = var(gauss_err_rates);
rect_err_rate = mean(rect_err_rates);
rect_err_var = var(rect_err_rates);
save('err_rate.mat', 'em_err_rates', 'em_err_rate', 'em_err_var',...
      'gauss_err_rates', 'gauss_err_rate', 'gauss_err_var',...
      'rect_err_rates', 'rect_err_rate', 'rect_err_var');