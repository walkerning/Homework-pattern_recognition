close all;
load 'coeff.mat';
figure(10)
%figure(1);
subplot(2, 3, 3);
histogram(coeff1);

hold on;
plot([effective_th1, effective_th1], [0, 200], 'r');
title('p = 1');

subplot(2, 3, 1);
histogram(coeff_25);

hold on;
plot([effective_th_25, effective_th_25], [0, 200], 'r');
title('p = 0.25');

subplot(2, 3, 2);
histogram(coeff_5);

hold on;
plot([effective_th_5, effective_th_5], [0, 200], 'r');
title('p = 0.5');

subplot(2, 3, 4);
histogram(coeff1_25);

hold on;
plot([effective_th1_25, effective_th1_25], [0, 200], 'r');
title('p = 1.25');

subplot(2, 3, 5);
histogram(coeff1);

hold on;
plot([effective_th1_5, effective_th1_5], [0, 200], 'r');
title('p = 1.5');
