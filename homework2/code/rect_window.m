function [ p_e ] = rect_window( a, xrange, sort_samples )
  %RECTANGLE 此处显示有关此函数的摘要
  %   此处显示详细说明

  N = length(sort_samples);
  p_e = zeros(1, length(xrange));
  min_support_ind = 1;
  for m=1:length(xrange)
    k = min_support_ind;
    x = xrange(m);
    flag = false;
    %fprintf(1, 'calculate %d, %f.  minsupport %d, %f\n', m, x, k, sort_samples(k));
    while k < N && sort_samples(k) < x + a/2
      if sort_samples(k) >= x - a/2
	if ~flag
	  min_support_ind = k;
	  flag = true;
	end
      %              if sort_samples(k) < x + a/2
      %                  p_e(m) = p_e(m) + 1/a;
      %              else
      %                  break;
      %              end
      end
      k = k + 1;
    end
    if flag
      p_e(m) = (k - min_support_ind)/a;
    end
  end
  p_e = p_e/N;

end

