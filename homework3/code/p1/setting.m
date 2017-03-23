N = 1000;
RUN = 20;
u1 = [-1, 0]; sigma1 = [1,0;0,1];
u2 = [1, 0]; sigma2 = [2,0;0,1];
prior1 = 0.5;
prior = [prior1, 1 - prior1];
step_size = 0.05;
X_MIN = -5; X_MAX = 8;
Y_MIN = -4; Y_MAX = 4;
err_rate = 0;
[mesh_x, mesh_y] = meshgrid((X_MIN:step_size:X_MAX), (Y_MIN:step_size:Y_MAX));
points = [reshape(mesh_x, [], 1), reshape(mesh_y, [], 1)];

delta = step_size * step_size;
[height, width] = size(mesh_x);