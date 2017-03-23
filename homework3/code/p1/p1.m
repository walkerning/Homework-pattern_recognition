clear; close all;
setting;
for h=1:height
    for w=1:width
        [p, ind] = min([mvnpdf([mesh_x(h,w), mesh_y(h, w)], u1, sigma1),...
            mvnpdf([mesh_x(h,w), mesh_y(h, w)], u2, sigma2)]);
        err_rate = err_rate + prior(ind) * p;
    end
end
err_rate = err_rate * delta;

%% parzen
