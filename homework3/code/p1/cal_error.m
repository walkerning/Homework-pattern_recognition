setting;
load('p_rect.mat');
load('p_gauss.mat');
load('p_em.mat');

p_true1 = reshape(mvnpdf(points, u1, sigma1), height, width);
p_true2 = reshape(mvnpdf(points, u2, sigma2), height, width);

err_rect1 = sum(sum((p_rect1 - p_true1).^2)) * delta;
err_rect2 = sum(sum((p_rect2 - p_true2).^2)) * delta;

err_gauss1 = sum(sum((p_gauss1 - p_true1).^2)) * delta;
err_gauss2 = sum(sum((p_gauss2 - p_true2).^2)) * delta;

err_em1 = sum(sum((p_em1 - p_true1).^2)) * delta;
err_em2 = sum(sum((p_em2 - p_true2).^2)) * delta;