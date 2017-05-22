function e = adaboost_error(X, y, k, a, d, alpha)
% adaboost_error: returns the final error rate of a whole adaboost
% 
% Input
%     X     : n * p matrix, each row a sample
%     y     : n * 1 vector, each row a label
%     k     : iter * 1 vector,  selected dimension of features
%     a     : iter * 1 vector, selected threshold for feature-k
%     d     : iter * 1 vector, 1 or -1
%     alpha : iter * 1 vector, weights of the classifiers
%
% Output
%     e     : error rate      

%%% Your Code Here %%%
n = length(y);
legal_inds = k > 0;
k = k(legal_inds);
a = a(legal_inds);
d = d(legal_inds);
alpha = alpha(legal_inds);

ps = ((X(:, k) <= repmat(a', n, 1)) - 0.5) .* repmat(d', n, 1) * 2;
p = sign(ps * alpha);
e = (sum(p ~= y)) / n;
%%% Your Code Here %%%
end