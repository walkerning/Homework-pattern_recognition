clc; close all; clear;
load 'ada_data.mat';
maxIter = 300;
[e_train, e_test] = adaboost(X_train, y_train, X_test, y_test, maxIter);
figure(1);
plot(1:maxIter, e_train);
hold on;
plot(1:maxIter, e_test)
legend('train error', 'test error');
title('error - iteration curve');
xlabel('iterations');