function [ p_e ] = gaussian_window( sigma, xrange, sort_samples )
  %GAUSSIAN_WINDOW 此处显示有关此函数的摘要
  %   此处显示详细说明

  N = length(sort_samples);
  p_e = zeros(1, length(xrange));

  support_threshold = 1e-3;
  a = 2 * norminv(1-support_threshold, 0, sigma);
  fprintf(1, 'sigma: %f, support width: %f\n', sigma, a);
  min_support_ind = 1;
  for m=1:length(xrange)
    k = min_support_ind;
    x = xrange(m);
    %fprintf(1, 'x: %d, %f, support start at : %f\n', m, x, k);
    flag = false;
    while k < N
      if sort_samples(k) >= x - a/2
        if ~flag
            min_support_ind = k;
            flag = true;
        end
        if sort_samples(k) < x + a/2
            %fprintf(1, 'plus %f at %d\n',  normpdf(x, sort_samples(k), sigma), m);
          p_e(m) = p_e(m) + normpdf(x, sort_samples(k), sigma);
        else
          break;
        end
      end
      k = k + 1;
    end
  end
  p_e = p_e/N;
end
