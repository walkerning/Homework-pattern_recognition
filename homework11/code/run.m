close all; clear; clc

load('../mnist.mat');
seed=12345;
rng(seed);
indexes = cell(1, 10);
for i = 1:10
  indexes{i} = find(train_Y==(i-1));
end
nums = cellfun(@(x) length(x), indexes);
train_num = size(train_X, 1);
test_num = size(test_X, 1);
K = 10;
ITERS = 100;
N = 100;

% train data number used
[points, labels] = subsample(train_X, (1:10), indexes, N);
[kmeans_clusters, kmeans_assigns] = kmeans(points, K, ITERS);
kmeans_nmi_score = NMI(labels, kmeans_assigns, K);
[~, spectral_assigns] = spectralCluster( points, K, ITERS, 1, 0);
spectral_nmi_score = NMI(labels, spectral_assigns, K);