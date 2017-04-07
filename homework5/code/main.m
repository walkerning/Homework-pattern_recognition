clear; close all;
load mnist;

digits = (0:9);
indexes = cell(1, 10);
for i = 0:9
  indexes{i+1} = find(train_Y==i);
end
nums = cellfun(@(x) length(x), indexes);
train_num = size(train_X, 1);
test_num = size(test_X, 1);

% train data number used
Ns = [1000];

% K
Ks = [1];

% only consider minkoski dist: 2, 3. ignored if tangent_dist == 1
powers = [3]; 

% use tangent dist or not
tangent_dist = 0;

% rotate transform or not
rotate = 0; 
% how to computate the pixel value at each grid after rotation; only useful when rotate ==1
interpolation = 'nearest';

%  vertical movement or horizontal movement; ignored if rotate == 1
vert = 1; 
if tangent_dist == 1
	 if rotate == 1
		 fprintf(1, 'use tangent dist: rotation transform: %s\n', interpolation);
	 else
		 if vert == 1
			 fprintf(1, 'use tangent dist: vertical movement transform\n');
		 else
			 fprintf(1, 'use tangent dist: horizontal movement transform\n');
		 end
	 end
end

% preprocess using per-feature weight or not
preprocess = 0;

%% preprocess data
if preprocess==1
	ori_train_X = train_X;
	ori_test_X = test_X;
	varis = zeros(10, 784);
	means = zeros(10,784);
	for i=1:10
		data = train_X(indexes{i}, :);
		means(i, :) = mean(data, 1);
		varis(i, :) = var(data, 0, 1);
	end

	p = 1; % 0.5, 0.25
	intravar_regular = 1e-3;
	coeff = var(means, 0, 1);
	intravar = sum(varis, 1);
	coeff = ((intravar + 9*coeff) ./ (intravar + intravar_regular)) .^ p;
	
	% process data by multiplty different weights `coeff` for each dimension
	train_X = repmat(coeff, train_num, 1) .* train_X;
	test_X = repmat(coeff, test_num, 1) .* test_X;

	% effective feature
	%effective_th = 0.4642; % 0.1^(1/3);
	effective_th = 0.5; % (1/8)^(1/3)
	%effective_th = 0.63; % (1/4)^(1/3)
	coeff_1norm = max(coeff);
	effective_num = sum(coeff/coeff_1norm > effective_th);
	fprintf(1, 'preprocess data: p: %f; effective: %d\n', p, effective_num);
end

%% calculate
for ind_power=1:length(powers)
  power = powers(ind_power);
  for ind=1:length(Ns)
		tic;
    N = Ns(ind);
    num_of_points = N * 10;
    labels = arrayfun(@(num) num * ones(1, N), digits, 'UniformOutput',...
											false);
    labels = cat(2, labels{:});
    points = zeros(num_of_points, 784);
    for cls=1:10
      points((cls-1)*N+1:cls*N, :) = train_X(indexes{cls}(1:N), :);
    end
		% tangent precaculate
		if tangent_dist == 1
			reshape_tan = reshape(points, num_of_points, 28, 28);
			if rotate == 1
				for ind_rotate=1:num_of_points
					im = reshape(reshape_tan(ind_rotate, :, :), 28, 28);
					T(ind_rotate, :) = reshape((imrotate(im, 10, interpolation, 'crop') - imrotate(im, -10, interpolation, 'crop')), 1, 784);
				end
			else
				if vert == 1
					T = reshape_tan(:, :, 3:28) - reshape_tan(:, :, 1:26);
					% 1 0 1 template
					T = reshape(cat(3, zeros(num_of_points, 28, 1), T, zeros(num_of_points, 28, 1)),...
											num_of_points, 784);
				else
					T = reshape_tan(:, 3:28, :) - reshape_tan(:, 1:26, :);
					T = reshape(cat(2, zeros(num_of_points, 1, 28), T, zeros(num_of_points, 1, 28)),...
											num_of_points, 784);
				end
			end
			norm_T = sum(T .^ 2, 2);
		end
    error_num = zeros(1, length(Ks));
    for ind_x=1:test_num
			if tangent_dist == 1
				 dist = tandist(test_X(ind_x, :), points, T, norm_T); 
			else
				if mod(power, 2) == 0
					dist = sum((repmat(test_X(ind_x, :), num_of_points, 1) - points).^power, 2);
				else
					dist = sum(abs(repmat(test_X(ind_x, :), num_of_points, 1) - points).^power, 2);
				end
			end
      [~, I] = sort(dist);
			for ind_k=1:length(Ks)
				K = Ks(ind_k);
				k_labels = labels(I(1:K));
				occurs = findoccur(k_labels, digits);
				[max_occur, max_ind] = max(occurs);
				if digits(max_ind) ~= test_Y(ind_x)
					error_num(ind_k) = error_num(ind_k) + 1;
				end
      end
		end
		for ind_k=1:length(Ks)
      fprintf(1, 'power: %d; number of points per class: %d; K: %d: error rate: %f\n',...
							power, N, Ks(ind_k), error_num(ind_k)/test_num);
		%. elapsed: %f\n',...
		%							N, Ks(ind_k), error_num/test_num, elapsed);
		end
	end
end
