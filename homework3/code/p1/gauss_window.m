function [ prob ] = gauss_window( sigma, x_mesh, y_mesh, samples)
%GAUSS_WINDOW 
    [height, width] = size(x_mesh);
    prob = zeros(height, width);
    [nsample, dim_size] = size(samples);
    % 这里dim_size 肯定为 2
    points = [reshape(x_mesh, [], 1), reshape(y_mesh, [], 1)];
    for n=1:nsample
        prob = prob + reshape(mvnpdf(points, samples(n, :),...
            eye(dim_size) * sigma), height, width);
    end
    prob = prob / nsample;
end

