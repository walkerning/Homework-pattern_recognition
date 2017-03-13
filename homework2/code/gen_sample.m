function [ samples ] = gen_sample( n )
%GEN_SAMPLE 
%   generate sample from 0.2N(-1, 1) + 0.8N(1,1)

samples = zeros(1, n);
for m=1:n
    num = randn();
    if rand(1) < 0.2
        samples(m) = num - 1;
    else
        samples(m) = num + 1;
    end
end

end

