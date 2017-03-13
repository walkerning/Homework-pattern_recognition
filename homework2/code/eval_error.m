function [ err_mean, err_var ] = eval_error( win_func, params, N, delta_x, xrange, RUN )
%EVAL_ERROR 此处显示有关此函数的摘要
%   此处显示详细说明

err_mat = zeros(RUN, size(params, 2));

for run=1:RUN
    samples = gen_sample(N);
    sort_samples = sort(samples);

    p = zeros(1, length(xrange));
    for m=1:length(p)
        p(m) = 0.2 * normpdf(xrange(m), -1, 1) + 0.8 * normpdf(xrange(m), 1, 1);
    end
    %err_rect(run, :) = zeros(1, length(as));

    for k=1:length(params)
        % rectangle window
        param = params(:, k);
        p_rect = win_func(param, xrange, sort_samples);

        figure(10+k);
        plot(xrange, p_rect);

        % compute error
        err_mat(run, k) = sum((p_rect - p).^2) * delta_x;
    end
end
% calculate mean and variance of err
err_mean = sum(err_mat, 1) / RUN;
err_var = sum((err_mat - repmat(err_mean, RUN, 1)).^2, 1)/RUN;
end

