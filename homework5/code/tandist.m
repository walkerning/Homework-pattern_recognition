function [dist] = tandist ( point, tanpoint, T, norm_T )
	num_points = size(tanpoint, 1);
	diffx = tanpoint - repmat(point, num_points, 1);
	T_diffx = sum(T .* diffx, 2);
	dist = sum(diffx.^2, 2) - (T_diffx.^2)./norm_T;
end
