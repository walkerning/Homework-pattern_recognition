function [ prob ] = rect_window( a, x_mesh, y_mesh, samples )
%RECT_WINDOW
    [height, width] = size(x_mesh);
    sz = height * width;
    prob = zeros(height, width);
    nsample = size(samples, 1);
    % 这里dim_size 肯定为 2
    points = [reshape(x_mesh, [], 1), reshape(y_mesh, [], 1)];
    for n=1:nsample
        prob = prob + reshape(all(abs(points - repmat(samples(n, :),...
                                                  sz, 1)) < a, 2),...
                              height, width);
    end
    prob = prob / nsample / a / a / 4;

end

