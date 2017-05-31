function [ points, labels ] = subsample( X, clses, indexes, N )
%SUBSAMPLE 此处显示有关此函数的摘要
%   此处显示详细说明
dim = size(X, 2);
labels = arrayfun(@(num) num * ones(1, N), clses, 'UniformOutput',...
											false);
labels = cat(2, labels{:});
points = zeros(N, dim);
for cls=1:length(clses)
   points((cls-1)*N+1:cls*N, :) = X(indexes{cls}(1:N), :);
end

end