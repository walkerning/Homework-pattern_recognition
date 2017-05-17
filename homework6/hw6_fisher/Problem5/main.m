clear; close all;
load orl_faces;

CLS = 5;
WHOLE_NUM = CLS * 10;
train_data = zeros(WHOLE_NUM, 4096);
% only cope with 10 classes
inds = cell(1, CLS);
means = zeros(4096, CLS);
nums = zeros(1, CLS);
vars = cell(1, CLS);
for ind=1:CLS
    inds{ind} = find(label == ind);
    nums(ind) = length(inds{ind});
    means(:, ind) = mean(data(inds{ind}, :), 1)';
    vars{ind} = cov(data(inds{ind}, :));
    train_data((ind-1) * 10+1:ind*10, :) = data(inds{ind}, :);
end

% one-versus-one
disfun_num = CLS * (CLS - 1) / 2;
one_ws = zeros(4096, disfun_num);
disfun_ind = 0;
eps=1e-5;
for c1 = 1:CLS
    for c2=c1+1:CLS
        disfun_ind = disfun_ind + 1;
        Sw = vars{c1} + vars{c2} + eps * eye(4096);
        one_ws(:, disfun_ind) = Sw \ (means(:, c1) - means(:, c2));
    end
end

% multi-class
mSw = cov(train_data);
%whole_mean = sum(repmat(nums, 4096, 1) .* means, 2) / sum(nums);
mSb = cov(means');
tmpM = mSw \ mSb;
[V, D] = eig(tmpM);
% pick top 10 features
multi_ws = V(:, 1:CLS);