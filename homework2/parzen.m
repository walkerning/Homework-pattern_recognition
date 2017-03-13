clc; clear all; close all;
N = 10000;
RUN = 10;
delta_x = 0.001;
xrange = -5:delta_x:5;

%% Different parameters for the same N
% as = [10, 1, 0.1, 0.01, 0.001];
% [err_rect_mean, err_rect_var] = eval_error(@rect_window, as, N, delta_x, xrange, RUN);
% sigmas = [1, 0.1, 0.01, 0.001];
% [err_gauss_mean, err_gauss_var] = eval_error(@gaussian_window, sigmas, N, delta_x, xrange, RUN);

%% Different N for the same parameters
% a = 0.1;
% Ns = [10, 100, 1000, 10000, 100000];
% err_rect_diff_n_mean = zeros(1, length(Ns));
% err_rect_diff_n_var = zeros(1, length(Ns));
% for n=1:length(Ns)
%     N = Ns(n);
%     [err_rect_diff_n_mean(n), err_rect_diff_n_var(n)] = eval_error(...
%         @rect_window, (a), N, delta_x, xrange, RUN);
% end
% sigma = 
% for run=1:RUN
%     samples = gen_sample(N);
%     sort_samples = sort(samples);
% 
%     p = zeros(1, length(xrange));
%     for m=1:length(p)
%         p(m) = 0.2 * normpdf(xrange(m), -1, 1) + 0.8 * normpdf(xrange(m), 1, 1);
%     end
%     %err_rect(run, :) = zeros(1, length(as));
% 
%     for k=1:length(as)
%         % rectangle window
%         a = as(k);
%         p_rect = rect_window(a, xrange, sort_samples);
% 
%         %figure(10+k);
%         %plot(xrange, p_rect);
% 
%         % compute error
%         err_rect_mat(run, k) = sum((p_rect - p).^2) * delta_x;
%     end
% end
% % calculate mean and variance of err
% err_rect_mean = sum(err_rect_mat, 1) / RUN;
% err_rect_var = sum((err_rect_mat - repmat(err_rect_mean, RUN, 1)).^2, 1)/RUN;

%% Gaussian
%err_gauss = sum((p_gauss - p(m)).^2) * delta_x;
%figure(2);
%plot(xrange, p_gauss);
% gaussian window
% sigma = 0.1;
% p_gauss = gaussian_window(sigma, xrange, sort_samples);