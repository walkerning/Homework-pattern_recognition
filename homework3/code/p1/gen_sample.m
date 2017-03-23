function [ samples1, samples2 ] = gen_sample( n, prior1, u1, sigma1, u2, sigma2 )
%GEN_SAMPLE 
%   generate samples

%samples = zeros(2, n);
y = rand(1, n) < prior1;
num1 = sum(y);
samples1 = mvnrnd(u1, sigma1, num1);
samples2 = mvnrnd(u2, sigma2, n - num1);

end

