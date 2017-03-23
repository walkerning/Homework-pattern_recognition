clear; close all;
load('hw5em2.mat');
figure(1);
plot(data2(:, 1), data2(:, 2), '.');
[nsamples, dim] = size(data2);
eps = 1e-4;
RUN = 5;
best_ll = zeros(1, 4);
best_param = cell(1, 4);
best_history = cell(1,4);
ms = (2:5);
for m=2:5
    for run=1:RUN
        [param, history, ll] = em_mix_modify(data2, m, eps);
        if (run == 1)
            best_ll(m-1) = ll(length(ll));
        else
            if (ll(length(ll)) > best_ll(m-1))
                best_ll(m-1) = ll(length(ll));
                best_param{m-1} = param;
                best_history{m-1} = history;
            end
        end
    end
end
% BIC
BIC = (ms * (1 + dim + dim*(dim+1)/2) - 1) * log(nsamples) - 2 * best_ll;
BIC_fix = (ms * (1 + dim) + dim*(dim+1)/2 - 1) * log(nsamples) - 2 * best_ll;
%1.0e+03 *
%   -4.2311   -4.1884   -4.0890   -4.0940