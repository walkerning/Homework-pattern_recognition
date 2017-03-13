function [ p_e ] = gaussian_window( sigma, xrange, sort_samples )
%GAUSSIAN_WINDOW �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

N = length(sort_samples);
p_e = zeros(1, length(xrange));
for m=1:length(xrange)
    x = xrange(m);
    for k=1:N
        p_e(m) = p_e(m) + normpdf(x, sort_samples(k), sigma);
    end
end
p_e = p_e/N;
end

