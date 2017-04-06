clear; close all;
load mnist;

digits = (0:9);
% choose the first
indexes = cell(1, 10);
for i = 0:9
   indexes{i+1} = find(train_Y==i);
end
nums = cellfun(@(x) length(x), indexes);
Ns = [10, 100, 1000, min(nums)];%, 500, min(nums)];
Ks = [10];
test_num = size(test_X, 1);

for ind_k=1:length(Ks)
    K = Ks(ind_k);
    for ind=1:length(Ns)
        start_time = tic;
        N = Ns(ind);
        num_of_points = N * 10;
        labels = arrayfun(@(num) num * ones(1, N), digits, 'UniformOutput',...
            false);
        labels = cat(2, labels{:});
        points = zeros(num_of_points, 784);
        for cls=1:10
            points((cls-1)*N+1:cls*N, :) = train_X(indexes{cls}(1:N), :);
        end
        error_num = 0;
        for ind_x=1:test_num
            dist = sum((repmat(test_X(ind_x, :), num_of_points, 1) - points).^2, 2);
            [~, I] = sort(dist);
            k_labels = labels(I(1:K));
            occurs = findoccur(k_labels, digits);
            [max_occur, max_ind] = max(occurs);
            if digits(max_ind) ~= test_Y(ind_x);
                error_num = error_num + 1;
            end
        end
        elapsed = toc;
        fprintf(1, 'number of points per class: %d; K: %d: error rate: %f. elapsed: %f\n',...
            N, K, error_num/test_num, elapsed);
    end
end