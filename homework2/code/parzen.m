clc; clear all; close all;
N = 10000;
RUN = 10;
delta_x = 0.01;
xrange = -5:delta_x:5;

%% Different parameters for the same N
% rect window
as = [10, 1, 0.1, 0.01, 0.001];
[err_rect_mean, err_rect_var] = eval_error(@rect_window, as, N, delta_x, xrange, RUN);
save('err_rect.mat', 'err_rect_mean', 'err_rect_var');

% gauss window
sigmas = [0.5, 0.1, 0.01, 0.001];
[err_gauss_mean, err_gauss_var] = eval_error(@gaussian_window, sigmas, N, delta_x, xrange, RUN/2);
save('err_gaussian.mat', 'err_gauss_mean', 'err_gauss_var');

%% Different N for the same parameters
% rect window
a = 0.1;
Ns = [10, 100, 1000, 10000, 100000];
err_rect_diff_n_mean = zeros(1, length(Ns));
err_rect_diff_n_var = zeros(1, length(Ns));
for n=1:length(Ns)
    N = Ns(n);
    [err_rect_diff_n_mean(n), err_rect_diff_n_var(n)] = eval_error(...
        @rect_window, (a), N, delta_x, xrange, RUN);
end

% gauss window
sigma = 0.1;
Ns = [10, 100, 1000, 10000, 100000];
err_gauss_diff_n_mean = zeros(1, length(Ns));
err_gauss_diff_n_var = zeros(1, length(Ns));
for n=1:length(Ns)
    N = Ns(n);
    [err_gauss_diff_n_mean(n), err_gauss_diff_n_var(n)] = eval_error(...
        @gaussian_window, (sigma), N, delta_x, xrange, RUN/2);
end
