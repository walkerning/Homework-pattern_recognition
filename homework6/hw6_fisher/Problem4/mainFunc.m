clear; clc; close all;
load least_sq.mat;

%% Step 1: Data preprocessing
eps = 1e-10;
dataTrain = train_small; % select the training data
% train = train_mid;
% train = train_large;

X = dataTrain.X;
y = dataTrain.y;
n = size(y, 1);
Lambda = 0.01: 0.01: 2.0; % a series of L1-norm penalty 
w_0 = pinv(X' * X) * (X' * y); % least-square estimation without L1-norm ...
                               % is supposed to be a good initial 
                               
%% Step 2: Train weight vectors with different penalty constants
W = least_sq_multi(X, y, Lambda, w_0); % each column a weight vector  

%% Step 3: plot different errors versus lambda
L = length(Lambda);
err_Lambda = zeros(L, 5); % each row a different lambda
for l = 1: L
  w = W(:, l);
  %%% Your code here %%%
  % training error multiplying 1/2
  err_Lambda(l, 1) = sum((y - X * w).^2) / (2 * n);

  % L1 regularization penalty
  err_Lambda(l, 2) = Lambda(l) * sum(abs(w));

  % minimized objective
  err_Lambda(l, 3) = err_Lambda(l, 1) + err_Lambda(l, 2);

  % L0 norm: non-zero parameters  
  err_Lambda(l, 4) = sum(abs(w) > eps);
  
  % test error
  test_X = test.X;
  test_y = test.y;
  err_Lambda(l, 5) = sum((test_y - test_X * w).^2) / 2 / size(test_y, 1);
end

figure;
plot(Lambda, err_Lambda(:, 1));
title('training error vs lambda');

figure;
plot(Lambda, err_Lambda(:,2));
title('L1 regularization penalty vs lambda');

figure;
plot(Lambda, err_Lambda(:,3));
title('objective vs lambda');

figure;
plot(Lambda, err_Lambda(:, 4)');
title('number features vs lambda');

figure;
plot(Lambda, err_Lambda(:, 5));
title('test error vs lambda');

[min_test_error, ind] = min(err_Lambda(:, 5));
fprintf(1, 'min test error: %f; lambda: %f; feature number: %d\n',...
    min_test_error, Lambda(ind), err_Lambda(ind, 4));
